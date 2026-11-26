"""Tablas de operación del modelo en Snowflake (esquema PSET2.MLOPS) y acceso a ellas.

  LOTES             entradas de cada lote, tal como llegan a producción (con la deriva inducida)
  TARGETS           ventas reales de cada hora y cuándo quedan disponibles
  LOTES_ESTADO      reloj de la simulación: cuándo se emitió cada lote y cuándo llegó su target
  PREDICCIONES      cada predicción de la API: id, timestamp, entradas, salida, versión, latencia
  SOLICITUDES       cada solicitud HTTP a la API (estado y latencia), para métricas de sistema
  EVALUACION        predicciones unidas con su target cuando este llega
  MONITOREO         métricas por lote (formato largo: tipo, métrica, feature, valor, umbral)
  ALERTAS           alertas disparadas
  REENTRENAMIENTOS  cada ciclo de reentrenamiento: disparo, comparación, regla y decisión
  LOTES_EXCLUIDOS   lotes que una persona decidió excluir del entrenamiento (con motivo)
"""
from __future__ import annotations

import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

from src.datos import conectar

ESQUEMA = "MLOPS"

DDL = {
    "LOTES": """
        lote_id int, fecha date, semana date, emitido_en timestamp_ntz,
        deriva_inducida varchar,
        hour_date date, hour_of_day int, iso_day_of_week int, is_holiday int,
        is_last_open_hour int, ventana_pago int, sales_lag_7d float, sales_lag_14d float,
        sales_lag_21d float, sales_lag_28d float, sales_avg_4w_same_hour float,
        ventas_dia_lag_7d float, staff_present int""",
    "TARGETS": """
        lote_id int, hour_date date, hour_of_day int, sales_count int,
        staff_present_real int, disponible_en timestamp_ntz, deriva_inducida varchar""",
    "LOTES_ESTADO": """
        lote_id int, fecha date, emitido_en timestamp_ntz, prediccion_registrada_en timestamp_tz,
        n_solicitudes int, n_errores int, target_recibido_en timestamp_ntz""",
    "PREDICCIONES": """
        id_prediccion varchar, ts timestamp_tz, lote_id int, hour_date date, hour_of_day int,
        entradas varchar, ventas_previstas float, carga_prevista float, umbral_carga float,
        decision varchar, modelo_nombre varchar, modelo_version int, modelo_run_id varchar,
        modelo_alias varchar, respaldo boolean, motivo_respaldo varchar, latencia_ms float""",
    "SOLICITUDES": """
        ts timestamp_tz, ruta varchar, metodo varchar, estado_http int, latencia_ms float,
        lote_id int""",
    "EVALUACION": """
        id_prediccion varchar, lote_id int, hour_date date, hour_of_day int,
        ventas_previstas float, ventas_reales int, staff_present_real int, umbral_carga float,
        decision varchar, reforzar_real boolean, prediccion_regla float, modelo_version int,
        respaldo boolean, evaluado_en timestamp_ntz""",
    "MONITOREO": """
        lote_id int, calculado_en timestamp_tz, etapa varchar, tipo varchar, metrica varchar,
        feature varchar, valor float, umbral float, alerta boolean, modelo_version int""",
    "ALERTAS": """
        alerta_id varchar, creada_en timestamp_tz, lote_id int, metrica varchar, feature varchar,
        valor float, umbral float, ventana varchar, severidad varchar, destinatario varchar,
        accion varchar, mensaje varchar""",
    "REENTRENAMIENTOS": """
        reentrenamiento_id varchar, creado_en timestamp_tz, lote_disparo int, motivo varchar,
        run_id varchar, version_challenger int, version_champion int, lote_evaluacion int,
        wape_champion float, wape_challenger float, costo_champion float,
        costo_challenger float, sigma_costo float, sigma_wape float, recomendacion varchar,
        decision varchar, motivo_decision varchar, decidido_en timestamp_tz,
        lotes_entrenamiento varchar""",
    "LOTES_EXCLUIDOS": "lote_id int, motivo varchar, excluido_en timestamp_tz",
}

# Tablas que reinicia una simulación nueva (las de la API también: cada simulación parte de cero).
TABLAS_SIMULACION = list(DDL)


def crear_tablas(reiniciar: bool = False) -> None:
    with conectar(ESQUEMA) as con:
        cur = con.cursor()
        for tabla, columnas in DDL.items():
            verbo = "create or replace table" if reiniciar else "create table if not exists"
            cur.execute(f"{verbo} {tabla} ({columnas})")


def leer(sql: str, params=None) -> pd.DataFrame:
    with conectar(ESQUEMA) as con:
        df = con.cursor().execute(sql, params).fetch_pandas_all()
    df.columns = [c.lower() for c in df.columns]
    return df


def ejecutar(sql: str, params=None) -> None:
    with conectar(ESQUEMA) as con:
        con.cursor().execute(sql, params)


def _columnas_fecha(tabla: str) -> set[str]:
    """Columnas declaradas como `date` en el DDL (write_pandas no convierte datetime -> DATE)."""
    return {c.split()[0] for c in DDL[tabla].replace("\n", " ").split(",") if c.split()[1:2] == ["date"]}


def escribir(df: pd.DataFrame, tabla: str) -> int:
    """Agrega filas (append). Las columnas se pasan en mayúsculas, como las crea el DDL."""
    if df.empty:
        return 0
    out = df.copy()
    for c in _columnas_fecha(tabla) & set(out.columns):
        out[c] = pd.to_datetime(out[c]).dt.date
    out.columns = [c.upper() for c in out.columns]
    with conectar(ESQUEMA) as con:
        ok, _, filas, _ = write_pandas(con, out, tabla, quote_identifiers=False,
                                       use_logical_type=True)
    if not ok:
        raise RuntimeError(f"write_pandas falló al escribir en {tabla}")
    return filas
