"""Consultas rápidas a Snowflake para revisar el sistema desde la terminal.

    python -m src.consultar obt               # validaciones de la última construcción de la OBT
    python -m src.consultar predicciones      # últimas predicciones de la API, con su trazabilidad
    python -m src.consultar alertas           # alertas de los lotes con deriva inducida
    python -m src.consultar reentrenamientos  # ciclos de reentrenamiento y su decisión

Con --exportado lee las copias que `src.exportar` dejó en docs/ en lugar de consultar
Snowflake (sirve sin conexión o con la cuenta suspendida). La OBT no tiene copia exportada.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DOCS = Path(__file__).resolve().parents[1] / "docs"

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


def _desde_csv(consulta: str) -> pd.DataFrame:
    """Lo mismo que CONSULTAS, sobre las tablas de MLOPS exportadas a docs/."""
    if consulta == "predicciones":
        df = pd.read_csv(DOCS / "predicciones.csv").sort_values("ts", ascending=False).head(5)
        df["ventas_previstas"] = df["ventas_previstas"].round(2)
        return df[["ts", "lote_id", "hour_date", "hour_of_day", "ventas_previstas",
                   "decision", "modelo_version", "modelo_run_id", "respaldo"]]
    if consulta == "alertas":
        df = pd.read_csv(DOCS / "alertas.csv")
        df = df[df["lote_id"].isin([18, 19, 25])].sort_values(["lote_id", "metrica"])
        df[["valor", "umbral"]] = df[["valor", "umbral"]].round(3)
        return df[["lote_id", "metrica", "feature", "valor", "umbral", "severidad", "accion"]]
    if consulta == "reentrenamientos":
        df = pd.read_csv(DOCS / "decisiones.csv").sort_values("creado_en")
        df[["costo_champion", "costo_challenger"]] = df[["costo_champion", "costo_challenger"]].round(2)
        return df[["lote_disparo", "motivo", "version_challenger", "version_champion",
                   "costo_champion", "costo_challenger", "recomendacion", "decision",
                   "motivo_decision", "lotes_entrenamiento"]]
    raise SystemExit("La auditoría de la OBT no se exporta a docs/: consúltala en Snowflake.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("consulta", choices=CONSULTAS)
    parser.add_argument("--exportado", action="store_true", help="leer las copias de docs/ en vez de Snowflake")
    args = parser.parse_args()
    if args.exportado:
        df = _desde_csv(args.consulta)
    else:
        from src.mlops import leer
        df = leer(CONSULTAS[args.consulta])
    df = df.convert_dtypes()
    df.columns = df.columns.str.lower()
    if args.consulta == "reentrenamientos":   # pocas filas y textos largos: una ficha por ciclo
        for _, fila in df.iterrows():
            print("\n".join(f"{k:>20}: {v}" for k, v in fila.items()), end="\n\n")
        return
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
