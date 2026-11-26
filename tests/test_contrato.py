"""Contrato de la API: una entrada válida devuelve 200 con el esquema esperado; una
inválida devuelve 422 y el mensaje dice qué campo falló."""
import json

import pytest

CAMPOS_RESPUESTA = {"id_prediccion", "ventas_previstas", "carga_prevista", "umbral_carga",
                    "decision", "accion", "version_modelo", "alias", "run_id", "respaldo",
                    "motivo_respaldo"}


def _campos_con_error(respuesta) -> set[str]:
    return {str(e["loc"][-1]) for e in respuesta.json()["detail"]}


def test_entrada_valida_devuelve_200_con_el_esquema(cliente, valida, metadatos):
    r = cliente.post("/predecir", json=valida)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == CAMPOS_RESPUESTA
    assert cuerpo["decision"] in {"reforzar", "no_reforzar"}
    assert cuerpo["version_modelo"] == metadatos["version"]
    assert cuerpo["run_id"] == metadatos["run_id"]
    assert cuerpo["respaldo"] is False
    assert cuerpo["umbral_carga"] == metadatos["umbral_decision"]
    assert cuerpo["accion"]


def test_la_prediccion_queda_registrada_con_su_version(cliente, valida, tmp_path, metadatos):
    r = cliente.post("/predecir", json=valida, headers={"X-Lote-Id": "7"})
    filas = [json.loads(l) for l in (tmp_path / "registro" / "predicciones.jsonl").read_text().splitlines()]
    fila = next(f for f in filas if f["id_prediccion"] == r.json()["id_prediccion"])
    assert fila["modelo_version"] == metadatos["version"]
    assert fila["modelo_run_id"] == metadatos["run_id"]
    assert fila["lote_id"] == 7
    assert fila["latencia_ms"] > 0
    assert json.loads(fila["entradas"])["hour_of_day"] == valida["hour_of_day"]


def test_campo_faltante_devuelve_422(cliente, valida):
    del valida["staff_present"]
    r = cliente.post("/predecir", json=valida)
    assert r.status_code == 422
    assert "staff_present" in _campos_con_error(r)


@pytest.mark.parametrize("campo, valor", [
    ("staff_present", "dos"),
    ("hour_of_day", 10.5),
    ("sales_lag_7d", "nueve"),
    ("hour_date", "ayer"),
    ("is_holiday", True),
])
def test_tipo_incorrecto_devuelve_422(cliente, valida, campo, valor):
    valida[campo] = valor
    r = cliente.post("/predecir", json=valida)
    assert r.status_code == 422
    assert campo in _campos_con_error(r)


@pytest.mark.parametrize("campo, valor", [
    ("hour_of_day", 22),      # la tienda no abre a esa hora
    ("hour_of_day", 7),
    ("iso_day_of_week", 9),
])
def test_categoria_desconocida_devuelve_422(cliente, valida, campo, valor):
    valida[campo] = valor
    r = cliente.post("/predecir", json=valida)
    assert r.status_code == 422
    assert campo in _campos_con_error(r)


@pytest.mark.parametrize("campo, valor", [
    ("staff_present", 0), ("staff_present", 9), ("sales_lag_7d", -1.0), ("is_last_open_hour", 2),
])
def test_valor_fuera_de_rango_devuelve_422(cliente, valida, campo, valor):
    valida[campo] = valor
    r = cliente.post("/predecir", json=valida)
    assert r.status_code == 422
    assert campo in _campos_con_error(r)


def test_campo_desconocido_devuelve_422(cliente, valida):
    valida["sales_count"] = 12   # el target nunca es una entrada
    r = cliente.post("/predecir", json=valida)
    assert r.status_code == 422
    assert "sales_count" in _campos_con_error(r)


def test_dia_incoherente_con_la_fecha_devuelve_422(cliente, valida):
    valida["iso_day_of_week"] = valida["iso_day_of_week"] % 7 + 1
    r = cliente.post("/predecir", json=valida)
    assert r.status_code == 422
    assert "iso_day_of_week" in r.text


def test_rezagos_nulos_se_aceptan(cliente, valida):
    for c in ["sales_lag_21d", "sales_lag_28d"]:
        valida[c] = None
    assert cliente.post("/predecir", json=valida).status_code == 200


def test_salud_y_modelo(cliente, metadatos):
    salud = cliente.get("/salud").json()
    assert salud == {"estado": "ok", "modelo_cargado": True, "modo_respaldo": False, "detalle": None}
    info = cliente.get("/modelo").json()
    assert (info["nombre"], info["version"], info["alias"], info["run_id"]) == (
        metadatos["nombre"], metadatos["version"], metadatos["alias"], metadatos["run_id"])
