"""Monitoreo de un lote (lo llama kestra/monitoreo.yml).

  python -m monitoreo.ejecutar entradas --lote K
      Después de predecir el lote K: métricas de sistema (latencia, errores, volumen), de
      modelo sin target (distribución de predicciones, tasa de cada decisión) y de datos
      (faltantes, fuera de rango, PSI + KS/chi² contra el entrenamiento).

  python -m monitoreo.ejecutar desempeno --lotes 1,2,3
      Cuando llega el target de esos lotes: WAPE, MAE y costo de negocio del modelo, de la
      regla de 4 semanas y de no hacer nada. Evalúa los disparadores de reentrenamiento.

Todo se guarda en MLOPS.MONITOREO (formato largo) y las alertas en MLOPS.ALERTAS. La
salida para Kestra dice si hay que reentrenar y por qué.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import src.pipeline as P
import src.tracking as T
from monitoreo import alertas as A
from monitoreo.deriva import CATEGORICAS_DERIVA, calibrar_umbrales, deriva_lote, psi
from src import mlops
from src.produccion import referencia, version

NUMERICAS_CONTROL = P.NUMERICAS


def _salida_kestra(**valores) -> None:
    print("::" + json.dumps({"outputs": valores}, default=str) + "::")


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fila(lote, etapa, tipo, metrica, valor, feature=None, umbral=None, alerta=False, version_=None):
    return {"lote_id": lote, "calculado_en": _ahora(), "etapa": etapa, "tipo": tipo,
            "metrica": metrica, "feature": feature,
            "valor": None if valor is None or not np.isfinite(valor) else float(valor),
            "umbral": umbral, "alerta": bool(alerta), "modelo_version": version_}


def _version_de(pred: pd.DataFrame) -> int:
    """Versión del modelo que produjo las predicciones (la del champion si todo fue respaldo)."""
    v = pred["modelo_version"].dropna()
    return int(v.mode().iloc[0]) if len(v) else version("champion").numero


def _publicar(metricas: list[dict], nuevas: list[dict]) -> None:
    mlops.escribir(pd.DataFrame(metricas), "MONITOREO")
    if nuevas:
        mlops.escribir(pd.DataFrame(nuevas), "ALERTAS")
    for a in nuevas:
        print(f"ALERTA [{a['severidad']}] lote {a['lote_id']} · {a['metrica']}"
              f"{' · ' + a['feature'] if a['feature'] else ''}: {a['mensaje']} → {a['destinatario']}: {a['accion']}")


# --------------------------------------------------------------------- entradas
def entradas(lote: int) -> None:
    lote_df = mlops.leer("select * from LOTES where lote_id = %s", (lote,))
    lote_df["hour_date"] = pd.to_datetime(lote_df["hour_date"])
    esperadas = len(lote_df)
    for _ in range(30):                               # la API escribe por tandas
        pred = mlops.leer("select * from PREDICCIONES where lote_id = %s", (lote,))
        sol = mlops.leer("select * from SOLICITUDES where lote_id = %s and ruta = '/predecir'", (lote,))
        if len(sol) >= esperadas:
            break
        time.sleep(2)
    v = version(_version_de(pred)) if len(pred) else version("champion")
    ref = referencia(v)
    train = ref["entrenamiento"]
    umbrales_psi = calibrar_umbrales(train)
    m, nuevas = [], []

    # ---- sistema
    lat = sol["latencia_ms"].astype(float)
    p50, p95 = (float(np.percentile(lat, 50)), float(np.percentile(lat, 95))) if len(lat) else (np.nan, np.nan)
    n = len(sol)
    t4 = float((sol["estado_http"].between(400, 499)).mean()) if n else 0.0
    t5 = float((sol["estado_http"] >= 500).mean()) if n else 0.0
    m += [_fila(lote, "entradas", "sistema", "latencia_p50_ms", p50, version_=v.numero),
          _fila(lote, "entradas", "sistema", "latencia_p95_ms", p95, umbral=A.LATENCIA_P95_MS,
                alerta=p95 > A.LATENCIA_P95_MS, version_=v.numero),
          _fila(lote, "entradas", "sistema", "tasa_4xx", t4, umbral=A.TASA_4XX, alerta=t4 > A.TASA_4XX, version_=v.numero),
          _fila(lote, "entradas", "sistema", "tasa_5xx", t5, umbral=A.TASA_5XX, alerta=t5 > A.TASA_5XX, version_=v.numero),
          _fila(lote, "entradas", "sistema", "volumen_solicitudes", n, version_=v.numero)]
    if p95 > A.LATENCIA_P95_MS:
        nuevas.append(A.nueva("latencia", lote, p95, A.LATENCIA_P95_MS, f"p95 = {p95:.0f} ms"))
    if t5 > A.TASA_5XX:
        nuevas.append(A.nueva("errores_5xx", lote, t5, A.TASA_5XX, f"{t5:.1%} de solicitudes con 5xx"))
    if t4 > A.TASA_4XX:
        nuevas.append(A.nueva("errores_4xx", lote, t4, A.TASA_4XX, f"{t4:.1%} de solicitudes rechazadas (4xx)"))

    # ---- modelo (sin target)
    if len(pred):
        yhat = pred["ventas_previstas"].astype(float)
        oof_dia = ref["oof"][ref["oof"]["iso_day_of_week"] == lote_df["iso_day_of_week"].iloc[0]]["pred"]
        psi_pred = psi(oof_dia, yhat)
        reforzar = float((pred["decision"] == "reforzar").mean())
        respaldo = float(pred["respaldo"].astype(bool).mean())
        m += [_fila(lote, "entradas", "modelo", "prediccion_media", yhat.mean(), version_=v.numero),
              _fila(lote, "entradas", "modelo", "prediccion_p10", yhat.quantile(.1), version_=v.numero),
              _fila(lote, "entradas", "modelo", "prediccion_p90", yhat.quantile(.9), version_=v.numero),
              _fila(lote, "entradas", "modelo", "psi_predicciones", psi_pred, version_=v.numero),
              _fila(lote, "entradas", "modelo", "tasa_reforzar", reforzar, version_=v.numero),
              _fila(lote, "entradas", "modelo", "tasa_no_reforzar", 1 - reforzar, version_=v.numero),
              _fila(lote, "entradas", "modelo", "tasa_respaldo", respaldo, umbral=A.RESPALDO_TASA,
                    alerta=respaldo > A.RESPALDO_TASA, version_=v.numero)]
        if respaldo > A.RESPALDO_TASA:
            motivo = pred["motivo_respaldo"].dropna().iloc[0] if pred["motivo_respaldo"].notna().any() else ""
            nuevas.append(A.nueva("respaldo", lote, respaldo, A.RESPALDO_TASA,
                                  f"{respaldo:.0%} de las predicciones con la regla de negocio ({motivo})"))

    # ---- datos
    for f in NUMERICAS_CONTROL:
        falt_lote = float(lote_df[f].isna().mean())
        falt_ref = float(train[train["iso_day_of_week"] == lote_df["iso_day_of_week"].iloc[0]][f].isna().mean())
        alerta_falt = falt_lote > falt_ref + A.FALTANTES_PP
        m.append(_fila(lote, "entradas", "datos", "faltantes", falt_lote, f, falt_ref + A.FALTANTES_PP,
                       alerta_falt, v.numero))
        if alerta_falt:
            nuevas.append(A.nueva("faltantes", lote, falt_lote, falt_ref + A.FALTANTES_PP,
                                  f"{falt_lote:.0%} faltantes (referencia {falt_ref:.0%})", f))
        x = lote_df[f].dropna()
        fuera = int(((x < train[f].min()) | (x > train[f].max())).sum())
        m.append(_fila(lote, "entradas", "datos", "fuera_rango", fuera, f, 0, fuera > 0, v.numero))
        if fuera:
            nuevas.append(A.nueva("fuera_rango", lote, fuera, 0,
                                  f"{fuera} filas fuera de [{train[f].min():g}, {train[f].max():g}] "
                                  f"(valores {sorted(x[(x < train[f].min()) | (x > train[f].max())].unique())[:5]})", f))
    for f in P.CATEGORICAS + ["staff_present"]:
        no_vistas = int((~lote_df[f].isin(train[f].unique())).sum())
        m.append(_fila(lote, "entradas", "datos", "categorias_no_vistas", no_vistas, f, 0, no_vistas > 0, v.numero))
        if no_vistas:
            nuevas.append(A.nueva("fuera_rango", lote, no_vistas, 0,
                                  f"{no_vistas} filas con categorías no vistas en entrenamiento", f))
    der = deriva_lote(lote_df, train)
    for _, r in der.iterrows():
        u = umbrales_psi[r["feature"]]
        alerta = bool(np.isfinite(r["psi"]) and r["psi"] > u)
        m.append(_fila(lote, "entradas", "deriva", "psi", r["psi"], r["feature"], u, alerta, v.numero))
        m.append(_fila(lote, "entradas", "deriva", f"p_valor_{r['prueba']}", r["p_valor"], r["feature"],
                       0.01, bool(r["p_valor"] < 0.01), v.numero))
        if alerta:
            confirma = r["p_valor"] < 0.01
            nuevas.append(A.nueva(
                "psi", lote, r["psi"], u,
                f"PSI {r['psi']:.2f} > {u:.2f}; {r['prueba'].upper()} p = {r['p_valor']:.2g} "
                f"({'confirma' if confirma else 'no confirma: posible falso positivo por tamaño de muestra'})",
                r["feature"], severidad="Alta" if confirma else "Media"))
    _publicar(m, nuevas)
    print(f"Lote {lote}: {len(m)} métricas, {len(nuevas)} alertas (modelo v{v.numero})")
    _salida_kestra(lote=lote, alertas=len(nuevas),
                   alertas_resumen=[f"{a['severidad']}: {a['metrica']} {a['feature'] or ''} — {a['mensaje']}" for a in nuevas])


# -------------------------------------------------------------------- desempeño
def _costos(ev: pd.DataFrame) -> dict:
    real = ev["reforzar_real"].astype(bool).to_numpy()
    modelo = (ev["decision"] == "reforzar").to_numpy()
    regla = (ev["prediccion_regla"] / ev["staff_present_real"] >= P.UMBRAL_CARGA).to_numpy()

    def costo(dec):
        return float((real & ~dec).sum() * P.COSTO_FN + (~real & dec).sum() * P.COSTO_FP)
    return {"costo_modelo": costo(modelo), "costo_regla": costo(regla),
            "costo_no_hacer_nada": costo(np.zeros_like(real)),
            "horas_sobrecarga": float(real.sum()),
            "recall_sobrecarga": float((real & modelo).sum() / real.sum()) if real.sum() else np.nan,
            "refuerzos_modelo": float(modelo.sum()), "refuerzos_regla": float(regla.sum())}


def _disparadores(champion) -> tuple[bool, str, int | None]:
    """¿Hay que reentrenar? Por desempeño (2 lotes seguidos con alerta de WAPE, posteriores
    al último disparo por desempeño) o por calendario (28 lotes etiquetados desde el fin del
    entrenamiento del champion o desde el último reentrenamiento por calendario)."""
    mon = mlops.leer("select lote_id, alerta from MONITOREO where metrica = 'wape' order by lote_id")
    previos = mlops.leer("select lote_disparo, motivo, creado_en from REENTRENAMIENTOS")
    ultimo_desemp = previos.loc[previos["motivo"].str.contains("desempeño"), "lote_disparo"].max() if len(previos) else None
    mon = mon.drop_duplicates("lote_id", keep="last").sort_values("lote_id")
    recientes = mon.tail(A.LOTES_SEGUIDOS)
    por_desemp = (len(recientes) == A.LOTES_SEGUIDOS and recientes["alerta"].all()
                  and (pd.isna(ultimo_desemp) or recientes["lote_id"].min() > ultimo_desemp)
                  and (recientes["lote_id"].diff().dropna() == 1).all())

    estado = mlops.leer("select lote_id, fecha, target_recibido_en from LOTES_ESTADO")
    etiq = estado[estado["target_recibido_en"].notna()].copy()
    etiq["fecha"] = pd.to_datetime(etiq["fecha"])
    desde = champion["fecha_fin_entrenamiento"]
    cal = previos[previos["motivo"].str.contains("calendario")] if len(previos) else previos
    if len(cal):
        lote_cal = int(cal["lote_disparo"].max())
        desde = max(desde, pd.Timestamp(estado.loc[estado["lote_id"] == lote_cal, "fecha"].iloc[0]))
    n_cal = int((etiq["fecha"] > desde).sum())
    por_calendario = n_cal >= A.LOTES_CALENDARIO

    motivos = []
    if por_desemp:
        motivos.append(f"desempeño: WAPE sobre el umbral en los lotes {recientes['lote_id'].tolist()}")
    if por_calendario:
        motivos.append(f"calendario: {n_cal} lotes etiquetados desde {desde.date()}")
    return bool(motivos), "; ".join(motivos), (int(etiq["lote_id"].max()) if len(etiq) else None)


def desempeno(lotes: list[int]) -> None:
    lista = ", ".join(str(l) for l in lotes)
    ev = mlops.leer(f"select * from EVALUACION where lote_id in ({lista})")
    m, nuevas = [], []
    for lote, g in ev.groupby("lote_id"):
        v = version(int(g["modelo_version"].dropna().mode().iloc[0]) if g["modelo_version"].notna().any() else "champion")
        ref = referencia(v)
        umbral = ref["wape_lote_media"] + A.K_SIGMA_WAPE * ref["wape_lote_std"]
        w = P.wape(g["ventas_reales"], g["ventas_previstas"])
        w_regla = P.wape(g["ventas_reales"], g["prediccion_regla"])
        alerta = w > umbral
        m += [_fila(lote, "desempeno", "modelo", "wape", w, umbral=umbral, alerta=alerta, version_=v.numero),
              _fila(lote, "desempeno", "modelo", "mae", P.mae(g["ventas_reales"], g["ventas_previstas"]), version_=v.numero),
              _fila(lote, "desempeno", "negocio", "wape_regla", w_regla, version_=v.numero)]
        for k, val in _costos(g).items():
            m.append(_fila(lote, "desempeno", "negocio", k, val, version_=v.numero))
        if alerta:
            nuevas.append(A.nueva("wape", lote, w, umbral,
                                  f"WAPE {w:.3f} > {umbral:.3f} (CV diaria {ref['wape_lote_media']:.3f} ± "
                                  f"{ref['wape_lote_std']:.3f}, modelo v{v.numero})"))
    _publicar(m, nuevas)

    # Criterio de negocio: costo acumulado de las últimas 2 semanas etiquetadas vs. la regla.
    todo = mlops.leer("select lote_id, metrica, valor from MONITOREO where etapa = 'desempeno' "
                      "and metrica in ('costo_modelo', 'costo_regla')")
    piv = todo.pivot_table(index="lote_id", columns="metrica", values="valor", aggfunc="last").sort_index().tail(14)
    alerta_neg = []
    if len(piv) == 14 and piv["costo_modelo"].sum() > piv["costo_regla"].sum():
        alerta_neg.append(A.nueva("costo_regla", int(piv.index.max()), piv["costo_modelo"].sum(),
                                  piv["costo_regla"].sum(),
                                  f"Últimos 14 lotes: modelo ${piv['costo_modelo'].sum():.2f} vs regla "
                                  f"${piv['costo_regla'].sum():.2f}"))
        _publicar([], alerta_neg)

    champion = referencia(version("champion"))
    reentrenar, motivo, lote_disparo = _disparadores(champion)
    if reentrenar:
        print(f"DISPARO DE REENTRENAMIENTO ({motivo})")
        if "desempeño" in motivo:
            _publicar([], [A.nueva("wape_seguido", lote_disparo, 1, A.LOTES_SEGUIDOS, motivo)])
    todas = nuevas + alerta_neg
    print(f"Lotes {lotes}: {len(m)} métricas, {len(todas)} alertas")
    _salida_kestra(lotes=lotes, alertas=len(todas), reentrenar=reentrenar, motivo=motivo,
                   lote_disparo=lote_disparo,
                   alertas_resumen=[f"{a['severidad']}: {a['metrica']} lote {a['lote_id']} — {a['mensaje']}" for a in todas])


if __name__ == "__main__":
    T.configurar()
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["entradas", "desempeno"])
    ap.add_argument("--lote", type=int)
    ap.add_argument("--lotes", type=str, help="lista separada por comas")
    a = ap.parse_args()
    if a.etapa == "entradas":
        entradas(a.lote)
    else:
        desempeno([int(x) for x in a.lotes.strip("[] ").split(",") if x.strip()])
