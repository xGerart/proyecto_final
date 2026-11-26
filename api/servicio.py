"""Modelo en producción: carga por alias, recarga sin reconstruir la imagen y respaldo.

- El modelo se resuelve desde el Model Registry (`models:/<nombre>@champion`): el alias
  se traduce a una versión concreta y se carga ESA versión, así la respuesta siempre dice
  qué versión la produjo aunque el alias se mueva a mitad de una solicitud.
- Un hilo revisa el alias cada INTERVALO_RECARGA_S segundos; si cambió (promoción o
  rollback), carga la nueva versión y la cambia de forma atómica. Si la carga falla, se
  queda con la versión anterior.
- Respaldo: si no hay modelo (no cargó) o si se activa el interruptor, responde la línea
  base de negocio del PSet 1 y la respuesta lo declara (`respaldo=True` + motivo).
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import ENTRADA, UMBRAL_CARGA, ReglaPromedio4Semanas

log = logging.getLogger("api.servicio")
RUTA_REGLA = Path(__file__).with_name("regla_respaldo.json")


@dataclass(frozen=True)
class ModeloCargado:
    modelo: object            # pyfunc de MLflow
    nombre: str
    version: int | None
    alias: str | None
    run_id: str | None
    umbral: float             # umbral de carga PREDICHA elegido en el PSet 3
    origen: str               # URI desde la que se cargó
    cargado_en: str


def cargar_regla() -> ReglaPromedio4Semanas:
    """La regla de negocio ya ajustada (medias por hora del entrenamiento del PSet 3)."""
    params = json.loads(RUTA_REGLA.read_text())
    regla = ReglaPromedio4Semanas()
    regla.media_hora_ = pd.Series({int(h): v for h, v in params["media_hora"].items()})
    regla.media_global_ = params["media_global"]
    return regla


def _cargar_desde_registro(uri: str) -> ModeloCargado:
    import mlflow
    from mlflow import MlflowClient

    nombre, alias = uri.removeprefix("models:/").split("@")
    cliente = MlflowClient()
    version = cliente.get_model_version_by_alias(nombre, alias)
    if "umbral_decision" not in version.tags:
        raise ValueError(f"{nombre} v{version.version} no tiene el tag umbral_decision")
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version.version}")
    return ModeloCargado(modelo, nombre, int(version.version), alias, version.run_id,
                         float(version.tags["umbral_decision"]), uri, _ahora())


def _cargar_desde_archivo(ruta: str, ruta_metadatos: str) -> ModeloCargado:
    """Copia exportada con mlflow.sklearn.save_model (CI no tiene servidor de MLflow)."""
    import mlflow

    meta = json.loads(Path(ruta_metadatos).read_text())
    modelo = mlflow.pyfunc.load_model(ruta)
    return ModeloCargado(modelo, meta["nombre"], int(meta["version"]), meta["alias"],
                         meta["run_id"], float(meta["umbral_decision"]), ruta, _ahora())


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ServicioModelo:
    def __init__(self, uri: str, ruta_metadatos: str | None, modo_respaldo: bool):
        self.uri = uri
        self.ruta_metadatos = ruta_metadatos
        self.regla = cargar_regla()
        self._actual: ModeloCargado | None = None
        self.error_carga: str | None = None
        self._respaldo_forzado = modo_respaldo
        self.motivo_interruptor: str | None = "MODO_RESPALDO=true al iniciar" if modo_respaldo else None
        self._candado = threading.Lock()

    # ------------------------------------------------------------------ carga
    def cargar(self) -> bool:
        """Carga (o recarga) el modelo. Devuelve True si cambió la versión en uso."""
        try:
            if self.uri.startswith("models:/"):
                nuevo = _cargar_desde_registro(self.uri)
            else:
                if not self.ruta_metadatos:
                    raise ValueError("MODELO_METADATOS es obligatorio cuando MODELO_URI es una ruta")
                nuevo = _cargar_desde_archivo(self.uri, self.ruta_metadatos)
        except Exception as e:  # el servicio sigue arriba con el respaldo
            self.error_carga = f"{type(e).__name__}: {e}"
            log.error("No se pudo cargar el modelo desde %s: %s", self.uri, self.error_carga)
            return False
        with self._candado:
            anterior = self._actual
            self._actual, self.error_carga = nuevo, None
        cambio = anterior is None or anterior.version != nuevo.version
        if cambio:
            log.info("Modelo en uso: %s v%s (run %s)", nuevo.nombre, nuevo.version, nuevo.run_id)
        return cambio

    def version_en_registro(self) -> int | None:
        """Versión a la que apunta hoy el alias (sin cargarla)."""
        from mlflow import MlflowClient

        nombre, alias = self.uri.removeprefix("models:/").split("@")
        return int(MlflowClient().get_model_version_by_alias(nombre, alias).version)

    def revisar_alias(self) -> None:
        if not self.uri.startswith("models:/"):
            return
        try:
            en_registro = self.version_en_registro()
        except Exception as e:
            log.warning("No se pudo consultar el alias: %s", e)
            return
        if self._actual is None or self._actual.version != en_registro:
            self.cargar()

    # ---------------------------------------------------------------- estado
    @property
    def actual(self) -> ModeloCargado | None:
        return self._actual

    @property
    def respaldo_forzado(self) -> bool:
        return self._respaldo_forzado

    def fijar_respaldo(self, activo: bool, motivo: str) -> None:
        self._respaldo_forzado = activo
        self.motivo_interruptor = motivo if activo else None
        log.warning("Interruptor de respaldo %s: %s", "ACTIVADO" if activo else "desactivado", motivo)

    # ------------------------------------------------------------- predicción
    def predecir(self, X: pd.DataFrame) -> dict:
        """Predicción + decisión. Con el respaldo, la regla de negocio y su umbral (8)."""
        actual = self._actual
        if self._respaldo_forzado or actual is None:
            motivo = (f"interruptor de respaldo activo ({self.motivo_interruptor})"
                      if self._respaldo_forzado else f"el modelo no cargó ({self.error_carga})")
            yhat = self.regla.predict(X)
            return {"yhat": yhat, "umbral": UMBRAL_CARGA, "modelo": None,
                    "respaldo": True, "motivo_respaldo": motivo}
        # Sin post-proceso: la predicción registrada es la del modelo, la misma con la que se
        # calculó su métrica en MLflow (así el monitoreo compara cifras comparables).
        yhat = np.asarray(actual.modelo.predict(X[ENTRADA]), float)
        return {"yhat": yhat, "umbral": actual.umbral, "modelo": actual,
                "respaldo": False, "motivo_respaldo": None}
