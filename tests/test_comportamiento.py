"""Comportamiento de negocio que el modelo debe cumplir (pruebas de invariancia y de
dirección, sobre la muestra fija)."""
import mlflow
import numpy as np
import pandas as pd
import pytest

import src.pipeline as P
from tests.conftest import FIXTURES, a_json


@pytest.fixture(scope="module")
def modelo():
    return mlflow.pyfunc.load_model(str(FIXTURES / "modelo"))


REZAGOS = ["sales_lag_7d", "sales_lag_14d", "sales_lag_21d", "sales_lag_28d"]


def _historia_sube(muestra: pd.DataFrame, delta: float) -> pd.DataFrame:
    """La misma hora vendió `delta` más en cada una de las 4 semanas anteriores. Se mueven
    juntas todas las columnas que dependen de esa historia, como pasaría en los datos: los 4
    rezagos, su promedio y el total del día de hace 7 días (que incluye esa hora)."""
    X = muestra[P.ENTRADA].copy()
    for c in REZAGOS + ["sales_avg_4w_same_hour", "ventas_dia_lag_7d"]:
        X[c] = X[c] + delta
    return X


@pytest.mark.parametrize("delta", [1.0, 3.0, 8.0])
def test_mas_historia_de_ventas_no_reduce_la_prediccion(modelo, muestra, delta):
    """Si esa hora vendió más en las semanas anteriores, la predicción no puede bajar:
    un plan que quite refuerzo justo donde la demanda creció no tiene sentido.

    Se prueba el aumento conjunto y no columna por columna: por colinealidad, Ridge asigna
    coeficiente negativo a sales_lag_28d y a ventas_dia_lag_7d (ver docs/cambios.md), y subir
    una sola de ellas dejando quietas las demás describe un caso que no existe en los datos."""
    completas = muestra.dropna(subset=REZAGOS + ["sales_avg_4w_same_hour", "ventas_dia_lag_7d"])
    base = modelo.predict(completas[P.ENTRADA])
    subida = modelo.predict(_historia_sube(completas, delta))
    assert np.all(subida >= base)


def test_agregar_un_vendedor_nunca_provoca_un_refuerzo(cliente, muestra):
    """Si con 1 vendedor la hora no necesita refuerzo, con 2 tampoco: el modelo nunca debe
    pedir más personal porque ya se agregó personal. (La carga por vendedor puede subir
    apenas en horas casi vacías, con menos de 1 venta prevista, sin efecto en la decisión.)"""
    for i in range(len(muestra)):
        fila = a_json(muestra.iloc[i])
        fila["staff_present"] = 1
        uno = cliente.post("/predecir", json=fila).json()
        fila["staff_present"] = 2
        dos = cliente.post("/predecir", json=fila).json()
        if uno["decision"] == "no_reforzar":
            assert dos["decision"] == "no_reforzar"
        if uno["carga_prevista"] >= 1:
            assert dos["carga_prevista"] < uno["carga_prevista"]


def test_la_fecha_solo_entra_por_sus_features(modelo, muestra):
    """Invariancia: la misma hora, con las mismas features, en otra semana da la misma
    predicción (el modelo no memoriza fechas, que en producción siempre son nuevas)."""
    X = muestra[P.ENTRADA].copy()
    movida = X.copy()
    movida["hour_date"] = movida["hour_date"] + pd.Timedelta(days=28)
    assert np.allclose(modelo.predict(X), modelo.predict(movida))


def test_la_decision_es_coherente_con_el_umbral(cliente, muestra):
    for i in range(0, len(muestra), 7):
        r = cliente.post("/predecir", json=a_json(muestra.iloc[i])).json()
        assert (r["decision"] == "reforzar") == (r["carga_prevista"] >= r["umbral_carga"] - 1e-4)
