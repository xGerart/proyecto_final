"""Simulación de producción: lotes diarios posteriores al entrenamiento.

Uso (lo llaman las tareas de kestra/produccion_lotes.yml):
  python -m src.simulacion construir           arma LOTES y TARGETS en Snowflake (reinicia MLOPS)
  python -m src.simulacion etiquetar --lote K  llegan los targets disponibles al reloj del lote K
  python -m src.simulacion predecir --lote K   el lote K pasa por la API
  python -m src.simulacion etiquetar --final   llegan los targets que faltan (fin de la simulación)

Reloj. El plan de refuerzo es semanal: las predicciones de una semana se emiten el domingo
anterior a las 23:00 (todas las features miran >= 7 días atrás, así que ya existen). El
target de un día llega ese mismo día a las 21:00, después de la ingesta de las 20:30. Al
emitir el lote K solo se conocen los targets con disponible_en <= emitido_en(K): los de la
semana anterior completa. Así ningún lote se evalúa con información que en la realidad
todavía no existiría.

Deriva inducida (documentada también en docs/cambios.md):
  - Lotes del 12 y 13-11 (DERIVA_CONCEPTUAL): las ventas reales de cada hora se duplican y
    ninguna entrada cambia. Simula un evento de demanda (p. ej. una promoción 2x1) que el
    modelo no conoce: cambia la relación entre las features y el target. La duplicación no
    se arrastra a los rezagos de los lotes siguientes: el experimento aísla su efecto en dos
    lotes y no contamina el resto de la simulación.
  - Lote del 19-11 (DERIVA_DATOS): los rezagos y el promedio de 4 semanas llegan de la
    sucursal 2 en vez de la matriz, como si el join de la OBT perdiera el filtro de
    branch_id. Cambia la distribución de las entradas; el target es el real.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import src.pipeline as P
from src import mlops
from src.datos import RAIZ, SNAPSHOT_PRODUCCION, cargar_obt, conectar

INICIO = P.INICIO_PRUEBA                       # primer día posterior al entrenamiento
FIN = pd.Timestamp("2026-11-25")
DERIVA_CONCEPTUAL = pd.to_datetime(["2026-11-12", "2026-11-13"])
FACTOR_CONCEPTUAL = 2.0
DERIVA_DATOS = pd.to_datetime(["2026-11-19"])
REZAGOS = ["sales_lag_7d", "sales_lag_14d", "sales_lag_21d", "sales_lag_28d",
           "sales_avg_4w_same_hour", "ventas_dia_lag_7d"]
CONTROL = RAIZ / "data" / "obt_control_deriva.parquet"
COLUMNAS_LOTE = ["lote_id", "fecha", "semana", "emitido_en", "deriva_inducida"] + P.ENTRADA


def _salida_kestra(**valores) -> None:
    """Kestra lee las salidas de una tarea de scripts desde esta línea del stdout."""
    print("::" + json.dumps({"outputs": valores}) + "::")


# ------------------------------------------------------------------- construir
def _rezagos_control(fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Rezagos de la sucursal 2 (control) para las fechas con deriva de datos. Se guardan
    en data/ la primera vez para que reconstruir los lotes dé siempre lo mismo."""
    if not CONTROL.exists():
        lista = ", ".join(f"'{f.date()}'" for f in fechas)
        with conectar() as con:
            df = con.cursor().execute(
                f"select hour_date, hour_of_day, {', '.join(REZAGOS[:-1])} "
                f"from PSET2.OBT.OBT_SALES_HOURLY where branch_id = 2 and hour_date in ({lista})"
            ).fetch_pandas_all()
            diaria = con.cursor().execute(
                "select hour_date, sum(sales_count) as ventas_dia from PSET2.OBT.OBT_SALES_HOURLY "
                "where branch_id = 2 group by hour_date").fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        diaria.columns = [c.lower() for c in diaria.columns]
        df["hour_date"] = pd.to_datetime(df["hour_date"])
        diaria["hour_date"] = pd.to_datetime(diaria["hour_date"]) + pd.Timedelta(days=7)
        df = df.merge(diaria.rename(columns={"ventas_dia": "ventas_dia_lag_7d"}), on="hour_date", how="left")
        df.to_parquet(CONTROL, index=False)
    return pd.read_parquet(CONTROL)


def construir_lotes() -> tuple[pd.DataFrame, pd.DataFrame]:
    snap = cargar_obt(ruta=SNAPSHOT_PRODUCCION)
    df = P.agregar_features(snap.datos)      # las features se calculan con toda la historia
    prod = df[(df["hour_date"] >= INICIO) & (df["hour_date"] <= FIN)].copy()
    fechas = sorted(prod["hour_date"].unique())
    prod["lote_id"] = prod["hour_date"].map({f: i + 1 for i, f in enumerate(fechas)})
    prod["fecha"] = prod["hour_date"]
    prod["semana"] = prod["week_start_date"]
    prod["emitido_en"] = prod["semana"] - pd.Timedelta(hours=1)          # domingo 23:00
    prod["deriva_inducida"] = None

    # Deriva de datos: rezagos de la sucursal 2 en las mismas fechas y horas.
    control = _rezagos_control(pd.DatetimeIndex(DERIVA_DATOS)).set_index(["hour_date", "hour_of_day"])
    en_deriva = prod["hour_date"].isin(DERIVA_DATOS)
    for col in REZAGOS:
        claves = list(zip(prod.loc[en_deriva, "hour_date"], prod.loc[en_deriva, "hour_of_day"]))
        prod.loc[en_deriva, col] = [control[col].get(k, np.nan) for k in claves]
    prod.loc[en_deriva, "deriva_inducida"] = "datos: rezagos de la sucursal 2 (join sin filtro de branch_id)"

    targets = prod[["lote_id", "hour_date", "hour_of_day", P.TARGET, "staff_present"]].rename(
        columns={"staff_present": "staff_present_real"})
    targets["disponible_en"] = targets["hour_date"] + pd.Timedelta(hours=21)
    targets["deriva_inducida"] = None
    concepto = targets["hour_date"].isin(DERIVA_CONCEPTUAL)
    targets.loc[concepto, P.TARGET] = np.round(targets.loc[concepto, P.TARGET] * FACTOR_CONCEPTUAL).astype(int)
    targets.loc[concepto, "deriva_inducida"] = f"conceptual: ventas reales x{FACTOR_CONCEPTUAL:g}, entradas sin cambio"
    prod.loc[prod["hour_date"].isin(DERIVA_CONCEPTUAL), "deriva_inducida"] = targets.loc[concepto, "deriva_inducida"].iloc[0]
    return prod[COLUMNAS_LOTE].reset_index(drop=True), targets.reset_index(drop=True)


def construir() -> None:
    lotes, targets = construir_lotes()
    mlops.crear_tablas(reiniciar=True)
    mlops.escribir(lotes, "LOTES")
    mlops.escribir(targets, "TARGETS")
    estado = lotes.groupby("lote_id", as_index=False).agg(fecha=("fecha", "first"),
                                                          emitido_en=("emitido_en", "first"))
    mlops.escribir(estado, "LOTES_ESTADO")
    snap = cargar_obt(ruta=SNAPSHOT_PRODUCCION)
    print(f"{lotes['lote_id'].nunique()} lotes ({lotes['fecha'].min().date()} a {lotes['fecha'].max().date()}), "
          f"{len(lotes)} horas. OBT {snap.tabla_origen} corte {snap.fecha_corte} hash {snap.hash[:16]}")
    _salida_kestra(lotes=sorted(int(x) for x in lotes["lote_id"].unique()), hash_datos=snap.hash)


# ------------------------------------------------------------------- predecir
def _cuerpo(fila: pd.Series) -> dict:
    d = {}
    for c in P.ENTRADA:
        v = fila[c]
        if c == "hour_date":
            d[c] = pd.Timestamp(v).date().isoformat()
        elif pd.isna(v):
            d[c] = None
        elif c in P.CATEGORICAS + P.BINARIAS + ["staff_present"]:
            d[c] = int(v)
        else:
            d[c] = float(v)
    return d


def predecir(lote: int) -> None:
    url = os.environ["API_URL"].rstrip("/") + "/predecir"
    filas = mlops.leer("select * from LOTES where lote_id = %s order by hour_of_day", (lote,))
    estados, versiones = [], set()
    with requests.Session() as s:
        for _, fila in filas.iterrows():
            r = s.post(url, json=_cuerpo(fila), headers={"X-Lote-Id": str(lote)}, timeout=30)
            estados.append(r.status_code)
            if r.ok:
                versiones.add(r.json()["version_modelo"])
    errores = sum(e >= 400 for e in estados)
    mlops.ejecutar(
        "update LOTES_ESTADO set prediccion_registrada_en = %s, n_solicitudes = %s, n_errores = %s "
        "where lote_id = %s",
        (datetime.now(timezone.utc).isoformat(), len(estados), errores, lote))
    print(f"Lote {lote}: {len(estados)} solicitudes, {errores} con error, versiones {sorted(versiones, key=str)}")
    _salida_kestra(lote=lote, solicitudes=len(estados), errores=errores)


# ------------------------------------------------------------------ etiquetar
def _esperar_predicciones(lotes: list[int], timeout_s: int = 60) -> None:
    """La API escribe en Snowflake por tandas (cada ~2 s): se espera a que estén todas."""
    lista = ", ".join(str(l) for l in lotes)
    esperadas = mlops.leer(f"select sum(n_solicitudes - n_errores) as n from LOTES_ESTADO "
                           f"where lote_id in ({lista})")["n"].iloc[0]
    inicio = time.time()
    while True:
        n = mlops.leer(f"select count(*) as n from PREDICCIONES where lote_id in ({lista})")["n"].iloc[0]
        if n >= esperadas or time.time() - inicio > timeout_s:
            if n < esperadas:
                raise RuntimeError(f"Faltan predicciones registradas: {n} de {esperadas}")
            return
        time.sleep(2)


def etiquetar(lote: int | None, final: bool = False) -> list[int]:
    """Une con su target las predicciones de los lotes cuyo target ya llegó al reloj."""
    estado = mlops.leer("select * from LOTES_ESTADO order by lote_id")
    if final:
        reloj = pd.Timestamp(estado["fecha"].max()) + pd.Timedelta(hours=23, minutes=59)
    else:
        reloj = pd.Timestamp(estado.loc[estado["lote_id"] == lote, "emitido_en"].iloc[0])
    targets = mlops.leer("select * from TARGETS")
    targets["disponible_en"] = pd.to_datetime(targets["disponible_en"])
    pendientes = estado[estado["target_recibido_en"].isna() & estado["prediccion_registrada_en"].notna()]
    llegan = sorted(set(targets.loc[targets["disponible_en"] <= reloj, "lote_id"]) & set(pendientes["lote_id"]))
    if not llegan:
        print(f"Reloj {reloj}: no llegó ningún target nuevo")
        _salida_kestra(etiquetados=[], reloj=str(reloj))
        return []
    _esperar_predicciones(llegan)
    lista = ", ".join(str(l) for l in llegan)
    pred = mlops.leer(f"select * from PREDICCIONES where lote_id in ({lista})")
    lotes = mlops.leer(f"select * from LOTES where lote_id in ({lista})")
    tg = targets[targets["lote_id"].isin(llegan)]
    for df in (pred, lotes, tg):
        df["hour_date"] = pd.to_datetime(df["hour_date"])
    from api.servicio import cargar_regla
    lotes["prediccion_regla"] = cargar_regla().predict(lotes)
    ev = (pred.merge(tg[["hour_date", "hour_of_day", P.TARGET, "staff_present_real"]],
                     on=["hour_date", "hour_of_day"], how="inner")
              .merge(lotes[["hour_date", "hour_of_day", "prediccion_regla"]],
                     on=["hour_date", "hour_of_day"], how="left"))
    ev = ev.rename(columns={P.TARGET: "ventas_reales"})
    ev["reforzar_real"] = ev["ventas_reales"] / ev["staff_present_real"] >= P.UMBRAL_CARGA
    ev["evaluado_en"] = reloj
    cols = ["id_prediccion", "lote_id", "hour_date", "hour_of_day", "ventas_previstas",
            "ventas_reales", "staff_present_real", "umbral_carga", "decision", "reforzar_real",
            "prediccion_regla", "modelo_version", "respaldo", "evaluado_en"]
    mlops.escribir(ev[cols], "EVALUACION")
    mlops.ejecutar(f"update LOTES_ESTADO set target_recibido_en = %s where lote_id in ({lista})",
                   (str(reloj),))
    print(f"Reloj {reloj}: llegaron los targets de los lotes {llegan} ({len(ev)} horas evaluadas)")
    _salida_kestra(etiquetados=llegan, reloj=str(reloj))
    return llegan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["construir", "predecir", "etiquetar"])
    ap.add_argument("--lote", type=int)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.accion == "construir":
        construir()
    elif a.accion == "predecir":
        predecir(a.lote)
    else:
        etiquetar(a.lote, final=a.final)
