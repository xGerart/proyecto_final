"""Exporta la evidencia a docs/ en CSV (`make exportar`).

  predicciones.csv      MLOPS.PREDICCIONES
  monitoreo.csv         MLOPS.MONITOREO
  alertas.csv           MLOPS.ALERTAS
  evaluacion.csv        MLOPS.EVALUACION (predicción + target, para el resultado de negocio)
  decisiones.csv        MLOPS.REENTRENAMIENTOS (comparación, regla y decisión de cada ciclo)
  corridas.csv          todas las corridas del experimento (mlflow.search_runs)
  reentrenamientos.csv  solo las corridas con tags.fase = 'reentrenamiento'
"""
from __future__ import annotations

import mlflow

import src.tracking as T
from src import mlops
from src.datos import RAIZ

DOCS = RAIZ / "docs"
TABLAS = {
    "predicciones": "select * from PREDICCIONES order by lote_id, hour_of_day, ts",
    "monitoreo": "select * from MONITOREO order by lote_id, calculado_en, etapa, tipo, metrica, feature",
    "alertas": "select * from ALERTAS order by creada_en",
    "evaluacion": "select * from EVALUACION order by lote_id, hour_of_day",
    "decisiones": "select * from REENTRENAMIENTOS order by creado_en",
}

if __name__ == "__main__":
    for nombre, sql in TABLAS.items():
        df = mlops.leer(sql)
        df.to_csv(DOCS / f"{nombre}.csv", index=False)
        print(f"docs/{nombre}.csv: {len(df)} filas")

    T.configurar()
    exp = mlflow.get_experiment_by_name(T.EXPERIMENTO)
    corridas = mlflow.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
    corridas.to_csv(DOCS / "corridas.csv", index=False)
    reent = mlflow.search_runs([exp.experiment_id], filter_string="tags.fase = 'reentrenamiento'",
                               order_by=["attributes.start_time ASC"])
    reent.to_csv(DOCS / "reentrenamientos.csv", index=False)
    print(f"docs/corridas.csv: {len(corridas)} corridas · docs/reentrenamientos.csv: {len(reent)}")
