"""El modelo cargado reproduce la métrica registrada en MLflow sobre una muestra fija.

Muestra: el conjunto de prueba del PSet 3 (139 horas), guardado en tests/fixtures.
Tolerancia: 1e-9 relativa. Ridge es determinista; la única fuente de diferencia entre
máquinas es el orden de las sumas en punto flotante (BLAS de macOS arm64 frente al de
Linux x86_64 en CI), que mueve el resultado en el orden de 1e-15. Una tolerancia mayor
dejaría pasar un cambio real (otra versión de scikit-learn que cambie el preprocesamiento
mueve el WAPE en 1e-3 o más); una exacta fallaría por redondeo.
"""
import mlflow
import numpy as np
import pytest

import src.pipeline as P
from tests.conftest import FIXTURES, a_json

TOL = 1e-9


@pytest.fixture(scope="module")
def prediccion(muestra):
    modelo = mlflow.pyfunc.load_model(str(FIXTURES / "modelo"))
    return modelo.predict(muestra[P.ENTRADA])


def test_reproduce_wape_registrado(muestra, prediccion, metricas):
    assert P.wape(muestra[P.TARGET], prediccion) == pytest.approx(metricas["test_wape"], rel=TOL)


def test_reproduce_mae_registrado(muestra, prediccion, metricas):
    assert P.mae(muestra[P.TARGET], prediccion) == pytest.approx(metricas["test_mae"], rel=TOL)


def test_reproduce_costo_de_negocio_registrado(muestra, prediccion, metricas, metadatos):
    costo = P.costo_negocio(muestra[P.TARGET], prediccion, muestra["staff_present"],
                            umbral=metadatos["umbral_decision"])["costo"]
    assert costo == pytest.approx(metricas["test_costo_negocio"], rel=TOL)


def test_la_api_sirve_la_misma_prediccion(cliente, muestra, prediccion):
    for i in [0, 45, 100]:
        r = cliente.post("/predecir", json=a_json(muestra.iloc[i]))
        assert r.json()["ventas_previstas"] == pytest.approx(prediccion[i], abs=1e-4)


def test_el_wape_queda_dentro_de_la_variacion_de_la_cv(metricas):
    """Sanidad del registro: la métrica de prueba no se aleja más de 2 desviaciones de la CV."""
    assert abs(metricas["test_wape"] - metricas["cv_wape_mean"]) < 2 * metricas["cv_wape_std"]
