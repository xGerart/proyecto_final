"""Registro de predicciones y solicitudes.

Cada predicción se guarda con su id, timestamp, entradas, salida, decisión, versión y
run_id del modelo y latencia; cada solicitud HTTP (incluidas las rechazadas) con su
código de estado y latencia, para las métricas de sistema.

La escritura no bloquea la respuesta: las filas van a una cola y un hilo las inserta en
Snowflake por tandas. La latencia registrada es la de atender la solicitud, sin la
escritura. Destino según REGISTRO_PREDICCIONES:
  snowflake -> tablas MLOPS.PREDICCIONES y MLOPS.SOLICITUDES (producción)
  archivo   -> JSON Lines en REGISTRO_RUTA (pruebas y CI, sin Snowflake)
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

log = logging.getLogger("api.registro")

COLUMNAS_PREDICCION = [
    "id_prediccion", "ts", "lote_id", "hour_date", "hour_of_day", "entradas",
    "ventas_previstas", "carga_prevista", "umbral_carga", "decision", "modelo_nombre",
    "modelo_version", "modelo_run_id", "modelo_alias", "respaldo", "motivo_respaldo",
    "latencia_ms",
]
COLUMNAS_SOLICITUD = ["ts", "ruta", "metodo", "estado_http", "latencia_ms", "lote_id"]


class RegistroArchivo:
    def __init__(self, ruta: str):
        self.ruta = Path(ruta)
        self.ruta.mkdir(parents=True, exist_ok=True)
        self._candado = threading.Lock()

    def _escribir(self, tabla: str, fila: dict) -> None:
        with self._candado, open(self.ruta / f"{tabla}.jsonl", "a") as f:
            f.write(json.dumps(fila, default=str) + "\n")

    def prediccion(self, fila: dict) -> None:
        self._escribir("predicciones", fila)

    def solicitud(self, fila: dict) -> None:
        self._escribir("solicitudes", fila)

    def cerrar(self) -> None:
        pass


class RegistroSnowflake:
    def __init__(self, intervalo_s: float = 2.0, tanda: int = 200):
        from src.datos import conectar

        self._con = conectar("MLOPS")
        self._cola: queue.Queue = queue.Queue()
        self._intervalo, self._tanda = intervalo_s, tanda
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, name="registro-snowflake", daemon=True)
        self._hilo.start()

    def prediccion(self, fila: dict) -> None:
        self._cola.put(("PREDICCIONES", fila))

    def solicitud(self, fila: dict) -> None:
        self._cola.put(("SOLICITUDES", fila))

    def _bucle(self) -> None:
        while not self._parar.wait(self._intervalo):
            self._vaciar()
        self._vaciar()

    def _vaciar(self) -> None:
        por_tabla: dict[str, list[dict]] = {"PREDICCIONES": [], "SOLICITUDES": []}
        while not self._cola.empty():
            tabla, fila = self._cola.get_nowait()
            por_tabla[tabla].append(fila)
        for tabla, filas in por_tabla.items():
            cols = COLUMNAS_PREDICCION if tabla == "PREDICCIONES" else COLUMNAS_SOLICITUD
            for i in range(0, len(filas), self._tanda):
                tanda = filas[i:i + self._tanda]
                try:
                    self._insertar(tabla, cols, tanda)
                except Exception:
                    # No se descarta en silencio: queda en el log del contenedor con el detalle.
                    log.exception("No se pudieron insertar %d filas en %s", len(tanda), tabla)

    def _insertar(self, tabla: str, cols: list[str], filas: list[dict]) -> None:
        marcadores = ", ".join(["%s"] * len(cols))
        sql = f"insert into {tabla} ({', '.join(cols)}) values ({marcadores})"
        valores = [tuple(f.get(c) for c in cols) for f in filas]
        self._con.cursor().executemany(sql, valores)

    def cerrar(self) -> None:
        self._parar.set()
        self._hilo.join(timeout=30)
        self._con.close()


def crear_registro(destino: str, ruta: str | None):
    if destino == "snowflake":
        return RegistroSnowflake()
    if destino == "archivo":
        if not ruta:
            raise ValueError("REGISTRO_RUTA es obligatorio con REGISTRO_PREDICCIONES=archivo")
        return RegistroArchivo(ruta)
    raise ValueError(f"REGISTRO_PREDICCIONES debe ser 'snowflake' o 'archivo', no {destino!r}")
