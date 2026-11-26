"""Catálogo de alertas (sección 6.3) y su evaluación.

Cada alerta tiene métrica, umbral, ventana, severidad, quién la recibe y la acción. Las
alertas se escriben en MLOPS.ALERTAS y en el log de Kestra (nivel WARN/ERROR).

Umbrales:
- PSI: calibrado por feature con los días de entrenamiento (ver monitoreo/deriva.py).
- WAPE del lote: media + 1,5 desviaciones del WAPE diario fuera de pliegue del champion.
  Con 1,5 desviaciones, un día normal la supera ~7 % de las veces: la alerta de un lote es
  una advertencia. El reentrenamiento exige 2 lotes seguidos (probabilidad ~0,5 % por azar).
- Latencia y errores: el servicio responde en ~5-20 ms en local; 500 ms es una degradación
  clara, no ruido.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

K_SIGMA_WAPE = 1.5
LOTES_SEGUIDOS = 2              # disparador por desempeño
LOTES_CALENDARIO = 28           # disparador por calendario: 4 semanas de lotes etiquetados
LATENCIA_P95_MS = 500.0
TASA_5XX = 0.01
TASA_4XX = 0.05
FALTANTES_PP = 0.20             # +20 puntos de faltantes sobre la referencia
RESPALDO_TASA = 0.0             # cualquier predicción con respaldo


@dataclass(frozen=True)
class DefinicionAlerta:
    metrica: str
    umbral: str
    ventana: str
    severidad: str
    destinatario: str
    accion: str


CATALOGO = {
    "psi": DefinicionAlerta(
        "PSI de cada feature (lote vs. mismo día de la semana del entrenamiento)",
        "> percentil 99 de los días de entrenamiento (mín. 0,25)", "1 lote",
        "Alta si KS/chi² p < 0,001; si no, Media", "Equipo de datos",
        "Revisar la fuente y el pipeline de la OBT; no reentrenar si el desempeño no cae"),
    "fuera_rango": DefinicionAlerta(
        "Valores fuera del rango de entrenamiento o categorías no vistas", "> 0 filas", "1 lote",
        "Media", "Equipo de datos", "Revisar el origen del valor; el modelo extrapola"),
    "faltantes": DefinicionAlerta(
        "% de faltantes por feature", "> referencia + 20 pp", "1 lote", "Media",
        "Equipo de datos", "Revisar la ingesta y los joins de la OBT"),
    "wape": DefinicionAlerta(
        "WAPE del lote (cuando llega el target)", "> media + 1,5 desv. del WAPE diario de la CV",
        "1 lote", "Media", "Científico de datos",
        "Diagnosticar el tipo de deriva; si se repite en 2 lotes seguidos, se reentrena"),
    "wape_seguido": DefinicionAlerta(
        "WAPE del lote", "> media + 1,5 desv. de la CV", "2 lotes seguidos", "Alta",
        "Científico de datos y administración",
        "Dispara reentrenamiento.yml (challenger + aprobación manual)"),
    "costo_regla": DefinicionAlerta(
        "Costo de negocio del modelo vs. la regla de 4 semanas", "> costo de la regla",
        "Acumulado de las últimas 2 semanas", "Alta", "Administración y científico de datos",
        "Evaluar pasar al respaldo (criterio de retirada 1)"),
    "latencia": DefinicionAlerta(
        "Latencia p95 de /predecir", "> 500 ms", "1 lote", "Media", "Equipo de plataforma",
        "Revisar recursos del contenedor y la carga del modelo"),
    "errores_5xx": DefinicionAlerta(
        "Tasa de errores 5xx", "> 1 %", "1 lote", "Alta", "Equipo de plataforma",
        "Revisar logs de la API; si persiste, activar el respaldo"),
    "errores_4xx": DefinicionAlerta(
        "Tasa de errores 4xx", "> 5 %", "1 lote", "Media", "Equipo de datos",
        "Entradas inválidas: revisar el contrato con el productor de los lotes"),
    "respaldo": DefinicionAlerta(
        "Predicciones servidas por el respaldo", "> 0", "1 lote", "Alta", "Equipo de plataforma",
        "El modelo no cargó o el interruptor está activo: confirmar que sea intencional"),
}


def nueva(clave: str, lote: int, valor: float, umbral: float, mensaje: str,
          feature: str | None = None, severidad: str | None = None) -> dict:
    d = CATALOGO[clave]
    return {"alerta_id": str(uuid.uuid4()), "creada_en": datetime.now(timezone.utc).isoformat(),
            "lote_id": lote, "metrica": clave, "feature": feature, "valor": float(valor),
            "umbral": float(umbral), "ventana": d.ventana, "severidad": severidad or d.severidad,
            "destinatario": d.destinatario, "accion": d.accion, "mensaje": mensaje}


def tabla_catalogo() -> pd.DataFrame:
    return pd.DataFrame([{"clave": k, **vars(v)} for k, v in CATALOGO.items()])
