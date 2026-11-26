"""Contrato de la API.

`EntradaPrediccion` reproduce la firma que MLflow registró con el modelo (las 13 columnas
de `src.pipeline.ENTRADA`, con sus tipos y cuáles admiten nulos) y le agrega lo que la
firma no expresa: rangos válidos, categorías permitidas y coherencia con la fecha.

Rangos: el esquema rechaza valores imposibles, no valores raros. Un valor posible pero
nunca visto en entrenamiento (p. ej. 3 vendedores presentes) se acepta y lo señala el
monitoreo como «fuera de rango»: rechazarlo dejaría a la tienda sin plan esa semana.
"""
from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

# Categorías permitidas (las que vio el entrenamiento): horas de inicio en que la matriz
# abre (8 = 08:00-09:00 … 18 = 18:00-19:00) y días ISO (1 = lunes … 7 = domingo).
Hora = Literal[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
Dia = Literal[1, 2, 3, 4, 5, 6, 7]
Binaria = Annotated[StrictInt, Field(ge=0, le=1)]
# 60 ventas en una hora es ~2,5 veces el máximo observado: por encima es un error de datos.
VentasHora = Annotated[float, Field(ge=0, le=60)]
VentasDia = Annotated[float, Field(ge=0, le=600)]


class EntradaPrediccion(BaseModel):
    """Una hora abierta de la matriz, con las features disponibles al planificar la semana."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "hour_date": "2026-11-10", "hour_of_day": 11, "iso_day_of_week": 2,
        "is_holiday": 0, "is_last_open_hour": 0, "ventana_pago": 0,
        "sales_lag_7d": 9, "sales_lag_14d": 7, "sales_lag_21d": 11, "sales_lag_28d": 8,
        "sales_avg_4w_same_hour": 8.75, "ventas_dia_lag_7d": 71, "staff_present": 2,
    }]})

    hour_date: date
    hour_of_day: Hora
    iso_day_of_week: Dia
    is_holiday: Binaria
    is_last_open_hour: Binaria
    ventana_pago: Binaria = Field(description="1 si el día es >= 28 o <= 3 (pago de sueldos).")
    # Los rezagos pueden faltar (semanas sin historia): el modelo los imputa.
    sales_lag_7d: VentasHora | None
    sales_lag_14d: VentasHora | None
    sales_lag_21d: VentasHora | None
    sales_lag_28d: VentasHora | None
    sales_avg_4w_same_hour: VentasHora | None
    ventas_dia_lag_7d: VentasDia | None
    staff_present: Annotated[StrictInt, Field(ge=1, le=4, description="Vendedores presentes en la hora.")]

    @model_validator(mode="after")
    def _coherencia(self):
        if self.iso_day_of_week != self.hour_date.isoweekday():
            raise ValueError(f"iso_day_of_week={self.iso_day_of_week} no coincide con hour_date "
                             f"{self.hour_date} (día {self.hour_date.isoweekday()})")
        pago = int(self.hour_date.day >= 28 or self.hour_date.day <= 3)
        if self.ventana_pago != pago:
            raise ValueError(f"ventana_pago={self.ventana_pago} no coincide con hour_date {self.hour_date}")
        return self


Decision = Literal["reforzar", "no_reforzar"]


class RespuestaPrediccion(BaseModel):
    id_prediccion: str
    ventas_previstas: float = Field(description="Ventas esperadas en la hora.")
    carga_prevista: float = Field(description="Ventas previstas por vendedor presente.")
    umbral_carga: float = Field(description="Carga prevista a partir de la cual se refuerza.")
    decision: Decision
    accion: str = Field(description="Qué hace la administración con esta predicción (PSet 1).")
    version_modelo: int | None = Field(description="Versión del Model Registry; None con el respaldo.")
    alias: str | None
    run_id: str | None
    respaldo: bool = Field(description="True si respondió la regla de negocio y no el modelo.")
    motivo_respaldo: str | None


class EstadoSalud(BaseModel):
    estado: Literal["ok", "degradado"]
    modelo_cargado: bool
    modo_respaldo: bool
    detalle: str | None


class InfoModelo(BaseModel):
    nombre: str
    version: int | None
    alias: str | None
    run_id: str | None
    umbral_decision: float | None
    cargado_en: str | None
    origen: str
    respaldo_activo: bool


class SolicitudRespaldo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activo: bool
    motivo: str = Field(min_length=3)
