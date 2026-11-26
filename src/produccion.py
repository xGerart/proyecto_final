"""El modelo en producción visto desde el Model Registry: versiones, alias y datos de
entrenamiento de cada versión (la referencia contra la que se mide la deriva)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.base import clone

import src.pipeline as P
import src.tracking as T
from src.datos import RAIZ, cargar_obt

DIR_REFERENCIAS = RAIZ / "data" / "referencias"
ARTEFACTO_DATOS = "datos/entrenamiento.parquet"


@dataclass(frozen=True)
class Version:
    numero: int
    run_id: str
    umbral: float
    alias: tuple[str, ...]


def version(numero_o_alias: int | str) -> Version:
    cliente = MlflowClient()
    if isinstance(numero_o_alias, str) and not numero_o_alias.isdigit():
        v = cliente.get_model_version_by_alias(T.MODELO_REGISTRADO, numero_o_alias)
    else:
        v = cliente.get_model_version(T.MODELO_REGISTRADO, str(numero_o_alias))
    if "umbral_decision" not in v.tags:
        raise ValueError(f"v{v.version} no tiene el tag umbral_decision")
    return Version(int(v.version), v.run_id, float(v.tags["umbral_decision"]), tuple(v.aliases))


def cargar_modelo(numero: int):
    return mlflow.sklearn.load_model(f"models:/{T.MODELO_REGISTRADO}/{numero}")


def datos_entrenamiento(v: Version) -> pd.DataFrame:
    """Filas con las que se entrenó la versión. Las versiones reentrenadas guardan su
    conjunto como artefacto; las del PSet 3 se reconstruyen del snapshot, verificando que
    el hash coincida con el registrado en la corrida."""
    cliente = MlflowClient()
    artefactos = {a.path for a in cliente.list_artifacts(v.run_id, "datos")}
    if ARTEFACTO_DATOS in artefactos:
        ruta = mlflow.artifacts.download_artifacts(run_id=v.run_id, artifact_path=ARTEFACTO_DATOS)
        return pd.read_parquet(ruta)
    params = cliente.get_run(v.run_id).data.params
    snap = cargar_obt()
    if snap.hash != params["datos_hash"]:
        raise ValueError(f"El snapshot local no es el que entrenó v{v.numero} "
                         f"({snap.hash[:12]} != {params['datos_hash'][:12]})")
    return P.dividir(P.agregar_features(snap.datos)).train


def metricas_por_dia(df: pd.DataFrame, pred, umbral: float, col_staff: str = "staff_present") -> pd.DataFrame:
    """WAPE, MAE y costo de negocio por día (el grano de un lote)."""
    tmp = df.assign(_pred=np.asarray(pred, float))
    filas = []
    for fecha, g in tmp.groupby("hour_date"):
        c = P.costo_negocio(g[P.TARGET], g["_pred"], g[col_staff], umbral)
        filas.append({"hour_date": fecha, "wape": P.wape(g[P.TARGET], g["_pred"]),
                      "mae": P.mae(g[P.TARGET], g["_pred"]), "costo": c["costo"],
                      "fn": c["fn"], "fp": c["fp"]})
    return pd.DataFrame(filas)


def referencia(v: Version) -> dict:
    """Referencia de monitoreo de una versión (se calcula una vez y se guarda en data/):
    - entrenamiento: sus filas de entrenamiento (distribuciones por feature)
    - por_dia: WAPE y costo por día de sus predicciones fuera de pliegue (misma CV del PSet 3)
    - oof: predicciones fuera de pliegue (distribución esperada de la salida)"""
    ruta = DIR_REFERENCIAS / f"v{v.numero}"
    if (ruta / "estadisticas.json").exists():
        ref = json.loads((ruta / "estadisticas.json").read_text())
        ref["fecha_fin_entrenamiento"] = pd.Timestamp(ref["fecha_fin_entrenamiento"])
        for nombre in ["entrenamiento", "por_dia", "oof"]:
            ref[nombre] = pd.read_parquet(ruta / f"{nombre}.parquet")
        return ref
    train = datos_entrenamiento(v)
    modelo = cargar_modelo(v.numero)
    res = P.evaluar(clone(modelo), train)
    en_cv = ~np.isnan(res.oof)
    por_dia = metricas_por_dia(train[en_cv], res.oof[en_cv], v.umbral)
    estadisticas = {"version": v.numero,
                     "wape_lote_media": float(por_dia["wape"].mean()),
                     "wape_lote_std": float(por_dia["wape"].std(ddof=1)),
                     "costo_lote_media": float(por_dia["costo"].mean()),
                     "costo_lote_std": float(por_dia["costo"].std(ddof=1)),
                     "fecha_fin_entrenamiento": str(train["hour_date"].max().date())}
    tablas = {"entrenamiento": train, "por_dia": por_dia,
              "oof": train[en_cv].assign(pred=res.oof[en_cv])}
    ruta.mkdir(parents=True, exist_ok=True)
    for nombre, df in tablas.items():
        df.to_parquet(ruta / f"{nombre}.parquet", index=False)
    (ruta / "estadisticas.json").write_text(json.dumps(estadisticas, indent=2))
    return referencia(v)
