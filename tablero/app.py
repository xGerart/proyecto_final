"""Tablero de monitoreo, lote por lote: sistema, modelo, deriva, alertas y reentrenamientos.

Lee de Snowflake (PSET2.MLOPS): MONITOREO, ALERTAS, REENTRENAMIENTOS y LOTES_ESTADO.
  streamlit run tablero/app.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src import mlops

INDIGO, TEAL, AMBAR, GRIS, ROJO = "#544ddb", "#1fa396", "#e9820c", "#8a8f98", "#d64545"

st.set_page_config(page_title="Monitoreo · ventas por hora", layout="wide")


@st.cache_data(ttl=30)
def cargar():
    mon = mlops.leer("select * from MONITOREO")
    alertas = mlops.leer("select * from ALERTAS order by creada_en")
    reent = mlops.leer("select * from REENTRENAMIENTOS order by creado_en")
    estado = mlops.leer("select * from LOTES_ESTADO order by lote_id")
    lotes = mlops.leer("select distinct lote_id, fecha, deriva_inducida from LOTES order by lote_id")
    return mon, alertas, reent, estado, lotes


def metrica(mon: pd.DataFrame, nombre: str, etapa: str | None = None) -> pd.DataFrame:
    m = mon[mon["metrica"] == nombre]
    if etapa:
        m = m[m["etapa"] == etapa]
    return m.sort_values("calculado_en").drop_duplicates(["lote_id", "feature"], keep="last")


def marcas(reent: pd.DataFrame, lotes_alerta: list[int]) -> alt.LayerChart | None:
    """Reglas verticales: reentrenamientos (línea continua) y lotes con deriva inducida."""
    capas = []
    if len(reent):
        r = reent.assign(etiqueta=lambda d: "reentrenamiento v" + d["version_challenger"].astype(str)
                         + " · " + d["decision"])
        capas.append(alt.Chart(r).mark_rule(color=INDIGO, strokeWidth=2).encode(
            x="lote_disparo:Q", tooltip=["etiqueta", "motivo", "recomendacion", "motivo_decision"]))
    if lotes_alerta:
        capas.append(alt.Chart(pd.DataFrame({"lote_id": lotes_alerta})).mark_rule(
            color=AMBAR, strokeDash=[4, 3]).encode(x="lote_id:Q"))
    return alt.layer(*capas) if capas else None


def con_marcas(grafico, capa):
    return grafico + capa if capa is not None else grafico


try:
    mon, alertas, reent, estado, lotes = cargar()
except Exception as e:     # el tablero no sirve sin las tablas: se dice por qué
    st.error(f"No se pudieron leer las tablas de PSET2.MLOPS: {e}")
    st.stop()

st.title("Pronóstico de ventas por hora · monitoreo en producción")
deriva = lotes.dropna(subset=["deriva_inducida"]).drop_duplicates("lote_id")
lotes_deriva = deriva["lote_id"].astype(int).tolist()
capa = marcas(reent, lotes_deriva)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lotes predichos", int(estado["prediccion_registrada_en"].notna().sum()))
c2.metric("Lotes con target", int(estado["target_recibido_en"].notna().sum()))
c3.metric("Alertas", len(alertas))
c4.metric("Reentrenamientos", len(reent))
st.caption("Línea índigo: reentrenamiento (pasa el cursor para ver la decisión). "
           "Línea ámbar punteada: lote con deriva inducida "
           + ", ".join(f"{int(r.lote_id)} ({r.deriva_inducida})" for r in deriva.itertuples()))

# ------------------------------------------------------------------ negocio
st.header("Negocio: costo de la decisión por lote")
costos = mon[(mon["etapa"] == "desempeno") & mon["metrica"].isin(["costo_modelo", "costo_regla", "costo_no_hacer_nada"])]
costos = costos.sort_values("calculado_en").drop_duplicates(["lote_id", "metrica"], keep="last")
if len(costos):
    costos = costos.sort_values("lote_id")
    costos["acumulado"] = costos.groupby("metrica")["valor"].cumsum()
    nombres = {"costo_modelo": "Modelo", "costo_regla": "Regla 4 semanas", "costo_no_hacer_nada": "No hacer nada"}
    costos["estrategia"] = costos["metrica"].map(nombres)
    g = alt.Chart(costos).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30)).encode(
        x=alt.X("lote_id:Q", title="Lote"), y=alt.Y("acumulado:Q", title="Costo acumulado (USD)"),
        color=alt.Color("estrategia:N", scale=alt.Scale(domain=list(nombres.values()), range=[INDIGO, TEAL, GRIS]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["lote_id", "estrategia", alt.Tooltip("valor:Q", title="costo del lote", format="$.2f"),
                 alt.Tooltip("acumulado:Q", format="$.2f")])
    st.altair_chart(con_marcas(g, capa).properties(height=280), use_container_width=True)
else:
    st.info("Todavía no llegó el target de ningún lote.")

# ------------------------------------------------------------------- modelo
st.header("Modelo")
col_a, col_b = st.columns(2)
w = metrica(mon, "wape", "desempeno")
if len(w):
    base = alt.Chart(w).encode(x=alt.X("lote_id:Q", title="Lote"))
    linea = base.mark_line(color=INDIGO, strokeWidth=2).encode(y=alt.Y("valor:Q", title="WAPE"))
    puntos = base.mark_point(size=60, filled=True).encode(
        y="valor:Q", color=alt.condition("datum.alerta", alt.value(ROJO), alt.value(INDIGO)),
        tooltip=["lote_id", alt.Tooltip("valor:Q", format=".3f"), alt.Tooltip("umbral:Q", format=".3f"), "modelo_version"])
    umbral = base.mark_line(color=GRIS, strokeDash=[5, 4]).encode(y="umbral:Q")
    col_a.subheader("WAPE por lote (cuando llega el target)")
    col_a.altair_chart(con_marcas(linea + umbral + puntos, capa).properties(height=260), use_container_width=True)
    col_a.caption("Gris punteado: umbral (media + 1,5 desv. del WAPE diario de la CV). Rojo: lote con alerta.")
tasa = metrica(mon, "tasa_reforzar", "entradas")
pred = metrica(mon, "prediccion_media", "entradas")
if len(tasa):
    col_b.subheader("Decisiones: % de horas con refuerzo")
    g = alt.Chart(tasa).mark_bar(color=TEAL, cornerRadiusEnd=3).encode(
        x=alt.X("lote_id:O", title="Lote"), y=alt.Y("valor:Q", title="% de horas", axis=alt.Axis(format="%")),
        tooltip=["lote_id", alt.Tooltip("valor:Q", format=".0%"), "modelo_version"])
    col_b.altair_chart(g.properties(height=260), use_container_width=True)
if len(pred):
    st.subheader("Distribución de las predicciones (media, p10 y p90 por lote)")
    p = mon[(mon["etapa"] == "entradas") & mon["metrica"].isin(["prediccion_p10", "prediccion_media", "prediccion_p90"])]
    p = p.sort_values("calculado_en").drop_duplicates(["lote_id", "metrica"], keep="last")
    piv = p.pivot_table(index="lote_id", columns="metrica", values="valor").reset_index()
    banda = alt.Chart(piv).mark_area(color=INDIGO, opacity=0.15).encode(
        x=alt.X("lote_id:Q", title="Lote"), y=alt.Y("prediccion_p10:Q", title="Ventas previstas por hora"), y2="prediccion_p90:Q")
    media = alt.Chart(piv).mark_line(color=INDIGO, strokeWidth=2).encode(x="lote_id:Q", y="prediccion_media:Q")
    st.altair_chart(con_marcas(banda + media, capa).properties(height=220), use_container_width=True)

# ------------------------------------------------------------------- deriva
st.header("Datos y deriva")
psi = metrica(mon, "psi", "entradas")
if len(psi):
    psi = psi.assign(sobre_umbral=lambda d: d["valor"] / d["umbral"])
    g = alt.Chart(psi).mark_rect(cornerRadius=2).encode(
        x=alt.X("lote_id:O", title="Lote"), y=alt.Y("feature:N", title=None),
        color=alt.Color("sobre_umbral:Q", title="PSI / umbral",
                        scale=alt.Scale(domain=[0, 1, 3], range=["#eef0fb", "#b9b6f0", INDIGO])),
        stroke=alt.condition("datum.alerta", alt.value(ROJO), alt.value(None)),
        tooltip=["lote_id", "feature", alt.Tooltip("valor:Q", title="PSI", format=".2f"),
                 alt.Tooltip("umbral:Q", format=".2f"), "alerta"])
    st.altair_chart(g.properties(height=200), use_container_width=True)
    st.caption("Color: PSI dividido por su umbral calibrado (≥ 1 = alerta, borde rojo).")
fuera = mon[(mon["etapa"] == "entradas") & mon["metrica"].isin(["fuera_rango", "categorias_no_vistas", "faltantes"])]
if len(fuera):
    resumen = (fuera.sort_values("calculado_en").drop_duplicates(["lote_id", "metrica", "feature"], keep="last")
               .query("alerta").pivot_table(index="lote_id", columns=["metrica", "feature"], values="valor"))
    if len(resumen):
        st.subheader("Calidad de las entradas (solo lotes con alerta)")
        st.dataframe(resumen, use_container_width=True)

# ------------------------------------------------------------------ sistema
st.header("Sistema")
s1, s2 = st.columns(2)
lat = mon[(mon["etapa"] == "entradas") & mon["metrica"].isin(["latencia_p50_ms", "latencia_p95_ms"])]
lat = lat.sort_values("calculado_en").drop_duplicates(["lote_id", "metrica"], keep="last")
if len(lat):
    g = alt.Chart(lat).mark_line(strokeWidth=2, point=True).encode(
        x=alt.X("lote_id:Q", title="Lote"), y=alt.Y("valor:Q", title="ms"),
        color=alt.Color("metrica:N", scale=alt.Scale(range=[TEAL, INDIGO]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["lote_id", "metrica", alt.Tooltip("valor:Q", format=".1f")])
    s1.subheader("Latencia de /predecir")
    s1.altair_chart(g.properties(height=220), use_container_width=True)
err = mon[(mon["etapa"] == "entradas") & mon["metrica"].isin(["tasa_4xx", "tasa_5xx", "volumen_solicitudes"])]
err = err.sort_values("calculado_en").drop_duplicates(["lote_id", "metrica"], keep="last")
if len(err):
    s2.subheader("Volumen y errores por lote")
    s2.dataframe(err.pivot_table(index="lote_id", columns="metrica", values="valor"), use_container_width=True)

# -------------------------------------------------------- alertas y reentrenos
st.header("Alertas")
if len(alertas):
    st.dataframe(alertas[["lote_id", "severidad", "metrica", "feature", "valor", "umbral", "mensaje",
                          "destinatario", "accion"]], use_container_width=True, hide_index=True)
st.header("Reentrenamientos")
if len(reent):
    st.dataframe(reent[["lote_disparo", "motivo", "version_champion", "version_challenger", "lote_evaluacion",
                        "costo_champion", "costo_challenger", "sigma_costo", "wape_champion", "wape_challenger",
                        "recomendacion", "decision", "motivo_decision"]], use_container_width=True, hide_index=True)
else:
    st.info("Sin reentrenamientos todavía.")
