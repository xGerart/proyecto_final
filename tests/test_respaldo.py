"""Respaldo: si el modelo no carga, o si se activa el interruptor, responde la línea base
de negocio del PSet 1 y la respuesta lo indica."""
import pytest

from tests.conftest import TOKEN


def test_sin_modelo_el_servicio_sigue_arriba_y_lo_reporta(cliente_sin_modelo):
    salud = cliente_sin_modelo.get("/salud").json()
    assert salud["estado"] == "degradado"
    assert salud["modelo_cargado"] is False
    assert salud["detalle"]


def test_sin_modelo_responde_la_regla_de_negocio(cliente_sin_modelo, valida):
    r = cliente_sin_modelo.post("/predecir", json=valida)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["respaldo"] is True
    assert "no cargó" in cuerpo["motivo_respaldo"]
    assert cuerpo["version_modelo"] is None
    # La regla: promedio de la misma hora en las 4 semanas anteriores, con umbral 8.
    assert cuerpo["ventas_previstas"] == pytest.approx(valida["sales_avg_4w_same_hour"])
    assert cuerpo["umbral_carga"] == 8.0


def test_regla_sin_historia_usa_la_media_de_la_hora(cliente_sin_modelo, valida):
    valida["sales_avg_4w_same_hour"] = None
    cuerpo = cliente_sin_modelo.post("/predecir", json=valida).json()
    assert cuerpo["respaldo"] is True
    assert cuerpo["ventas_previstas"] > 0


def test_el_interruptor_exige_token(cliente):
    r = cliente.post("/respaldo", json={"activo": True, "motivo": "prueba"})
    assert r.status_code == 401
    r = cliente.post("/respaldo", json={"activo": True, "motivo": "prueba"},
                     headers={"X-Token-Admin": "otro"})
    assert r.status_code == 401


def test_el_interruptor_activa_y_desactiva_el_respaldo(cliente, valida):
    h = {"X-Token-Admin": TOKEN}
    assert cliente.post("/predecir", json=valida).json()["respaldo"] is False

    r = cliente.post("/respaldo", json={"activo": True, "motivo": "costo sobre la línea base"}, headers=h)
    assert r.json()["modo_respaldo"] is True
    cuerpo = cliente.post("/predecir", json=valida).json()
    assert cuerpo["respaldo"] is True
    assert "interruptor" in cuerpo["motivo_respaldo"]
    assert cliente.get("/modelo").json()["respaldo_activo"] is True

    cliente.post("/respaldo", json={"activo": False, "motivo": "revisado"}, headers=h)
    assert cliente.post("/predecir", json=valida).json()["respaldo"] is False


def test_modo_respaldo_por_variable_de_entorno(monkeypatch, tmp_path, valida):
    from tests.conftest import FIXTURES, _cliente
    with _cliente(monkeypatch, tmp_path, str(FIXTURES / "modelo"), modo_respaldo="true") as c:
        assert c.get("/salud").json()["modo_respaldo"] is True
        assert c.post("/predecir", json=valida).json()["respaldo"] is True
