"""Genera los fixtures de las pruebas desde el servidor de MLflow (se corre a mano, no en CI).

  .venv/bin/python -m tests.fixtures.generar

- modelo/          copia del champion exportada con mlflow.sklearn.save_model
- modelo.json      nombre, versión, alias, run_id y umbral de decisión de esa versión
- muestra.csv      muestra fija: el conjunto de prueba del PSet 3 (139 horas, 26-10 a 07-11)
- metricas.json    métricas que MLflow registró para esa versión sobre esa muestra

CI no tiene acceso al servidor de MLflow: con estos archivos la prueba del modelo
comprueba que la copia reproduce la métrica registrada.
"""
import json
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

import src.pipeline as P
import src.tracking as T
from src.datos import cargar_obt

DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    T.configurar()
    cliente = MlflowClient()
    version = cliente.get_model_version_by_alias(T.MODELO_REGISTRADO, "champion")
    run = cliente.get_run(version.run_id)

    snap = cargar_obt()
    assert snap.hash == run.data.params["datos_hash"], "el snapshot no es el que entrenó al modelo"
    test = P.dividir(P.agregar_features(snap.datos)).test
    test[P.ENTRADA + [P.TARGET]].to_csv(DIR / "muestra.csv", index=False)

    modelo = mlflow.sklearn.load_model(f"models:/{T.MODELO_REGISTRADO}/{version.version}")
    info = mlflow.models.get_model_info(f"models:/{T.MODELO_REGISTRADO}/{version.version}")
    shutil.rmtree(DIR / "modelo", ignore_errors=True)
    mlflow.sklearn.save_model(modelo, DIR / "modelo", signature=info.signature,
                              serialization_format="cloudpickle")
    (DIR / "modelo.json").write_text(json.dumps({
        "nombre": T.MODELO_REGISTRADO, "version": int(version.version), "alias": "champion",
        "run_id": version.run_id, "umbral_decision": float(version.tags["umbral_decision"]),
    }, indent=2))
    (DIR / "metricas.json").write_text(json.dumps({
        k: run.data.metrics[k] for k in ["test_wape", "test_mae", "test_costo_negocio",
                                         "cv_wape_mean", "cv_wape_std"]
    }, indent=2))
    print(f"Fixtures de {T.MODELO_REGISTRADO} v{version.version} ({version.run_id}) en {DIR}")
