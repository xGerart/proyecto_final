"""Consultas rápidas a Snowflake para revisar el sistema desde la terminal.

    python -m src.consultar obt               # validaciones de la última construcción de la OBT
    python -m src.consultar predicciones      # últimas predicciones de la API, con su trazabilidad
    python -m src.consultar alertas           # alertas de los lotes con deriva inducida
    python -m src.consultar reentrenamientos  # ciclos de reentrenamiento y su decisión
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.mlops import leer

CONSULTAS = {
    "obt": """
        select check_name, expected, actual, passed, checked_at
        from PSET2.OBT.OBT_AUDIT
        where checked_at = (select max(checked_at) from PSET2.OBT.OBT_AUDIT)
        order by check_name""",
    "predicciones": """
        select ts, lote_id, hour_date, hour_of_day, round(ventas_previstas, 2) as ventas_previstas,
               decision, modelo_version, modelo_run_id, respaldo
        from PREDICCIONES order by ts desc limit 5""",
    "alertas": """
        select lote_id, metrica, feature, round(valor, 3) as valor, round(umbral, 3) as umbral,
               severidad, accion
        from ALERTAS where lote_id in (18, 19, 25) order by lote_id, metrica""",
    "reentrenamientos": """
        select lote_disparo, motivo, version_challenger, version_champion,
               round(costo_champion, 2) as costo_champion, round(costo_challenger, 2) as costo_challenger,
               recomendacion, decision, motivo_decision, lotes_entrenamiento
        from REENTRENAMIENTOS order by creado_en""",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("consulta", choices=CONSULTAS)
    args = parser.parse_args()
    df = leer(CONSULTAS[args.consulta]).convert_dtypes()
    df.columns = df.columns.str.lower()
    if args.consulta == "reentrenamientos":   # pocas filas y textos largos: una ficha por ciclo
        for _, fila in df.iterrows():
            print("\n".join(f"{k:>20}: {v}" for k, v in fila.items()), end="\n\n")
        return
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
