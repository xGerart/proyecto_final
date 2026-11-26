"""Servicio de predicción de ventas por hora de la matriz.

Endpoints:
  GET  /salud            estado del servicio y si el modelo cargó
  GET  /modelo           nombre, versión, alias y run_id del modelo en uso
  POST /predecir         predicción, decisión de refuerzo, acción y versión del modelo
  POST /modelo/recargar  (protegido) vuelve a resolver el alias sin esperar al hilo
  POST /respaldo         (protegido) interruptor: regla de negocio en vez del modelo

Configuración (variables de entorno):
  MODELO_URI            models:/ventas-hora-matriz-modelo@champion, o una ruta local
  MODELO_METADATOS      JSON con versión/run_id/umbral (solo si MODELO_URI es una ruta)
  REGISTRO_PREDICCIONES snowflake | archivo  (+ REGISTRO_RUTA si es archivo)
  MODO_RESPALDO         true para arrancar con el interruptor activado
  TOKEN_ADMIN           token de los endpoints protegidos (cabecera X-Token-Admin)
  INTERVALO_RECARGA_S   cada cuántos segundos se revisa el alias (0 = no revisar)
"""
from __future__ import annotations

import logging
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request

from api.esquemas import (EntradaPrediccion, EstadoSalud, InfoModelo, RespuestaPrediccion,
                          SolicitudRespaldo)
from api.registro import crear_registro
from api.servicio import ServicioModelo
from src.pipeline import ENTRADA, decision_refuerzo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


def _bool(nombre: str) -> bool:
    valor = os.environ[nombre].strip().lower()
    if valor not in {"true", "false"}:
        raise ValueError(f"{nombre} debe ser true o false, no {valor!r}")
    return valor == "true"


def _revisar_alias_periodicamente(servicio: ServicioModelo, intervalo: float, parar: threading.Event):
    while not parar.wait(intervalo):
        servicio.revisar_alias()


@asynccontextmanager
async def lifespan(app: FastAPI):
    servicio = ServicioModelo(
        uri=os.environ["MODELO_URI"],
        ruta_metadatos=os.environ.get("MODELO_METADATOS"),
        modo_respaldo=_bool("MODO_RESPALDO"),
    )
    servicio.cargar()
    app.state.servicio = servicio
    app.state.registro = crear_registro(os.environ["REGISTRO_PREDICCIONES"],
                                        os.environ.get("REGISTRO_RUTA"))
    app.state.token_admin = os.environ["TOKEN_ADMIN"]
    parar = threading.Event()
    intervalo = float(os.environ["INTERVALO_RECARGA_S"])
    if intervalo > 0:
        threading.Thread(target=_revisar_alias_periodicamente, args=(servicio, intervalo, parar),
                         name="recarga-alias", daemon=True).start()
    yield
    parar.set()
    app.state.registro.cerrar()


app = FastAPI(title="Ventas por hora · matriz", version="1.0", lifespan=lifespan,
              description="Pronóstico de ventas por hora y decisión de refuerzo de personal.")


def _ahora() -> datetime:
    return datetime.now(timezone.utc)


def _lote(request: Request) -> int | None:
    """La simulación de producción marca cada solicitud con su lote (cabecera X-Lote-Id)."""
    valor = request.headers.get("x-lote-id")
    return int(valor) if valor and valor.isdigit() else None


@app.middleware("http")
async def medir_solicitud(request: Request, call_next):
    """Métricas de sistema: toda solicitud (también las 4xx/5xx) queda con estado y latencia."""
    inicio = time.perf_counter()
    estado = 500
    try:
        respuesta = await call_next(request)
        estado = respuesta.status_code
        return respuesta
    finally:
        registro = getattr(request.app.state, "registro", None)
        if registro is not None and request.url.path not in {"/docs", "/openapi.json"}:
            registro.solicitud({
                "ts": _ahora().isoformat(), "ruta": request.url.path, "metodo": request.method,
                "estado_http": estado, "latencia_ms": (time.perf_counter() - inicio) * 1000,
                "lote_id": _lote(request),
            })


def exigir_admin(request: Request, x_token_admin: str = Header(default="")):
    if not secrets.compare_digest(x_token_admin, request.app.state.token_admin):
        raise HTTPException(status_code=401, detail="Token de administración inválido")


# ------------------------------------------------------------------- endpoints
@app.get("/salud", response_model=EstadoSalud)
def salud(request: Request):
    s: ServicioModelo = request.app.state.servicio
    cargado = s.actual is not None
    detalle = s.error_carga if not cargado else (
        f"respaldo activo: {s.motivo_interruptor}" if s.respaldo_forzado else None)
    return EstadoSalud(estado="ok" if cargado and not s.respaldo_forzado else "degradado",
                       modelo_cargado=cargado, modo_respaldo=s.respaldo_forzado, detalle=detalle)


@app.get("/modelo", response_model=InfoModelo)
def modelo(request: Request):
    s: ServicioModelo = request.app.state.servicio
    m = s.actual
    if m is None:
        return InfoModelo(nombre=s.uri, version=None, alias=None, run_id=None, umbral_decision=None,
                          cargado_en=None, origen=s.uri, respaldo_activo=True)
    return InfoModelo(nombre=m.nombre, version=m.version, alias=m.alias, run_id=m.run_id,
                      umbral_decision=m.umbral, cargado_en=m.cargado_en, origen=m.origen,
                      respaldo_activo=s.respaldo_forzado)


def _accion(entrada: EntradaPrediccion, decision: str) -> str:
    hora = f"{entrada.hour_of_day:02d}:00-{entrada.hour_of_day + 1:02d}:00"
    if decision == "reforzar":
        return f"Programar al vendedor de refuerzo el {entrada.hour_date.isoformat()} de {hora}."
    return f"Sin refuerzo el {entrada.hour_date.isoformat()} de {hora}: alcanza el personal programado."


@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: EntradaPrediccion, request: Request):
    inicio = time.perf_counter()
    s: ServicioModelo = request.app.state.servicio
    fila = entrada.model_dump()
    X = pd.DataFrame([fila])
    # Tipos exactos de la firma registrada en MLflow (datetime y enteros long).
    X["hour_date"] = pd.to_datetime(X["hour_date"])
    for c in ["hour_of_day", "iso_day_of_week", "is_holiday", "is_last_open_hour",
              "ventana_pago", "staff_present"]:
        X[c] = X[c].astype("int64")
    for c in ENTRADA:
        if X[c].dtype == object:
            X[c] = X[c].astype(float)

    salida = s.predecir(X)
    yhat = float(salida["yhat"][0])
    reforzar = bool(decision_refuerzo([yhat], [entrada.staff_present], salida["umbral"])[0])
    decision = "reforzar" if reforzar else "no_reforzar"
    m = salida["modelo"]
    respuesta = RespuestaPrediccion(
        id_prediccion=str(uuid.uuid4()),
        ventas_previstas=round(yhat, 4),
        carga_prevista=round(yhat / entrada.staff_present, 4),
        umbral_carga=salida["umbral"],
        decision=decision,
        accion=_accion(entrada, decision),
        version_modelo=m.version if m else None,
        alias=m.alias if m else None,
        run_id=m.run_id if m else None,
        respaldo=salida["respaldo"],
        motivo_respaldo=salida["motivo_respaldo"],
    )
    request.app.state.registro.prediccion({
        "id_prediccion": respuesta.id_prediccion, "ts": _ahora().isoformat(),
        "lote_id": _lote(request), "hour_date": entrada.hour_date.isoformat(),
        "hour_of_day": entrada.hour_of_day, "entradas": entrada.model_dump_json(),
        "ventas_previstas": yhat, "carga_prevista": respuesta.carga_prevista,
        "umbral_carga": respuesta.umbral_carga, "decision": decision,
        "modelo_nombre": m.nombre if m else "regla_promedio_4_semanas",
        "modelo_version": respuesta.version_modelo, "modelo_run_id": respuesta.run_id,
        "modelo_alias": respuesta.alias, "respaldo": respuesta.respaldo,
        "motivo_respaldo": respuesta.motivo_respaldo,
        "latencia_ms": (time.perf_counter() - inicio) * 1000,
    })
    return respuesta


@app.post("/modelo/recargar", response_model=InfoModelo, dependencies=[Depends(exigir_admin)])
def recargar(request: Request):
    s: ServicioModelo = request.app.state.servicio
    s.cargar()
    if s.actual is None:
        raise HTTPException(status_code=503, detail=f"No se pudo cargar el modelo: {s.error_carga}")
    return modelo(request)


@app.post("/respaldo", response_model=EstadoSalud, dependencies=[Depends(exigir_admin)])
def respaldo(cuerpo: SolicitudRespaldo, request: Request):
    request.app.state.servicio.fijar_respaldo(cuerpo.activo, cuerpo.motivo)
    return salud(request)
