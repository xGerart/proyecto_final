"""Clientes de prueba de la API con la copia del modelo de tests/fixtures (sin MLflow ni Snowflake)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOKEN = "token-de-prueba"


def _cliente(monkeypatch, tmp_path, modelo_uri: str, modo_respaldo: str = "false"):
    monkeypatch.setenv("MODELO_URI", modelo_uri)
    monkeypatch.setenv("MODELO_METADATOS", str(FIXTURES / "modelo.json"))
    monkeypatch.setenv("REGISTRO_PREDICCIONES", "archivo")
    monkeypatch.setenv("REGISTRO_RUTA", str(tmp_path / "registro"))
    monkeypatch.setenv("MODO_RESPALDO", modo_respaldo)
    monkeypatch.setenv("TOKEN_ADMIN", TOKEN)
    monkeypatch.setenv("INTERVALO_RECARGA_S", "0")
    from api.main import app
    return TestClient(app)


@pytest.fixture
def cliente(monkeypatch, tmp_path):
    with _cliente(monkeypatch, tmp_path, str(FIXTURES / "modelo")) as c:
        yield c


@pytest.fixture
def cliente_sin_modelo(monkeypatch, tmp_path):
    """El modelo no carga (ruta inexistente): el servicio debe seguir arriba con el respaldo."""
    with _cliente(monkeypatch, tmp_path, str(tmp_path / "no-existe")) as c:
        yield c


@pytest.fixture(scope="session")
def muestra() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "muestra.csv", parse_dates=["hour_date"])


@pytest.fixture(scope="session")
def metricas() -> dict:
    return json.loads((FIXTURES / "metricas.json").read_text())


@pytest.fixture(scope="session")
def metadatos() -> dict:
    return json.loads((FIXTURES / "modelo.json").read_text())


def a_json(fila: pd.Series) -> dict:
    """Fila de la muestra -> cuerpo de /predecir (NaN -> null, fecha ISO, enteros)."""
    d = {}
    for k, v in fila.items():
        if k == "sales_count":
            continue
        if k == "hour_date":
            d[k] = pd.Timestamp(v).date().isoformat()
        elif pd.isna(v):
            d[k] = None
        elif k in {"hour_of_day", "iso_day_of_week", "is_holiday", "is_last_open_hour",
                   "ventana_pago", "staff_present"}:
            d[k] = int(v)
        else:
            d[k] = float(v)
    return d


@pytest.fixture
def valida(muestra) -> dict:
    return a_json(muestra.iloc[30])
