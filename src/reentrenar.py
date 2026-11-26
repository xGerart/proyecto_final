"""Reentrenamiento del champion con datos nuevos (lo llama kestra/reentrenamiento.yml).

  python -m src.reentrenar entrenar --motivo "..." --lote-disparo K
      1. Ventana: todo el histórico etiquetado. El entrenamiento del PSet 3 (snapshot
         verificado por hash) + los lotes de producción cuyo target ya llegó, salvo
         (a) el lote etiquetado más reciente, que se reserva para comparar champion y
         challenger, (b) los lotes con alertas de datos de severidad Alta (entradas que se
         sabe que están mal) y (c) los que una persona excluyó en una decisión anterior.
      2. Misma preparación (src/pipeline.py) e hiperparámetros del champion: se clona su
         Pipeline, sin tuning. El umbral de decisión se vuelve a elegir con las
         predicciones fuera de pliegue, igual que en el PSet 3.
      3. Corrida en MLflow con fase=reentrenamiento y el motivo; nueva versión en el
         Model Registry con alias 'challenger'.
      4. Champion y challenger se evalúan sobre el lote reservado y se aplica la regla de
         promoción. El resultado queda en MLOPS.REENTRENAMIENTOS como recomendación.

  python -m src.reentrenar decidir --id ID --decision aprobar|rechazar --motivo "..." [--excluir 18,19]
      La decisión de una persona. Si aprueba, el alias 'champion' pasa al challenger (y la
      versión anterior queda con el alias 'anterior', para el rollback). Si rechaza, el
      champion no cambia y queda el motivo.

  python -m src.reentrenar alias --version N
      Rollback (o promoción manual): mueve 'champion' a la versión N sin reconstruir nada.

Regla de promoción (sección 4): el challenger reemplaza al champion si, en el mismo lote
reservado, su costo de negocio es menor que el del champion en más de una desviación
estándar del costo diario de la CV del champion, y su WAPE no es peor en más de una
desviación estándar del WAPE diario de la CV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import clone

import src.pipeline as P
import src.tracking as T
from src import mlops
from src.datos import RAIZ, cargar_obt
from src.produccion import ARTEFACTO_DATOS, cargar_modelo, referencia, version


def _salida_kestra(**valores) -> None:
    print("::" + json.dumps({"outputs": valores}, default=str) + "::")


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _lotes_etiquetados() -> pd.DataFrame:
    """Entradas tal como llegaron + target real, solo de lotes cuyo target ya llegó."""
    df = mlops.leer("""
        select l.*, t.sales_count
        from LOTES l
        join TARGETS t on t.lote_id = l.lote_id and t.hour_date = l.hour_date
                      and t.hour_of_day = l.hour_of_day
        join LOTES_ESTADO e on e.lote_id = l.lote_id
        where e.target_recibido_en is not null""")
    df["hour_date"] = pd.to_datetime(df["hour_date"])
    df["week_start_date"] = pd.to_datetime(df["semana"])
    return df


def _excluidos() -> tuple[set[int], set[int]]:
    """(lotes con alertas de datos Alta, lotes excluidos por una persona)."""
    alertas = mlops.leer("select distinct lote_id from ALERTAS where severidad = 'Alta' "
                         "and metrica in ('psi', 'fuera_rango', 'faltantes')")
    personas = mlops.leer("select distinct lote_id from LOTES_EXCLUIDOS")
    return set(alertas["lote_id"].astype(int)), set(personas["lote_id"].astype(int))


def ventana_entrenamiento() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    snap = cargar_obt()
    base = P.dividir(P.agregar_features(snap.datos)).train
    etiq = _lotes_etiquetados()
    lote_eval = int(etiq["lote_id"].max())
    por_datos, por_persona = _excluidos()
    usados = sorted(set(etiq["lote_id"]) - {lote_eval} - por_datos - por_persona)
    nuevos = etiq[etiq["lote_id"].isin(usados)]
    cols = P.ENTRADA + [P.TARGET, "week_start_date"]
    train = pd.concat([base[cols], nuevos[cols]], ignore_index=True).sort_values(
        ["hour_date", "hour_of_day"]).reset_index(drop=True)
    evaluacion = etiq[etiq["lote_id"] == lote_eval].copy()
    info = {"lote_evaluacion": lote_eval, "lotes_usados": usados,
            "excluidos_por_alerta_de_datos": sorted(por_datos & set(etiq["lote_id"])),
            "excluidos_por_decision": sorted(por_persona), "snapshot_hash": snap.hash,
            "ventana_inicio": str(train["hour_date"].min().date()),
            "ventana_fin": str(train["hour_date"].max().date())}
    return train, evaluacion, info


def _evaluar_en_lote(modelo, umbral: float, lote: pd.DataFrame) -> dict:
    pred = modelo.predict(lote[P.ENTRADA])
    c = P.costo_negocio(lote[P.TARGET], pred, lote["staff_present"], umbral)
    return {"wape": P.wape(lote[P.TARGET], pred), "costo": c["costo"], "fn": c["fn"], "fp": c["fp"]}


def entrenar(motivo: str, lote_disparo: int) -> None:
    T.configurar()
    champion = version("champion")
    ref = referencia(champion)
    modelo_champion = cargar_modelo(champion.numero)
    train, evaluacion, info = ventana_entrenamiento()

    candidato = clone(modelo_champion)              # misma preparación e hiperparámetros
    res = P.evaluar(candidato, train)
    en_cv = ~np.isnan(res.oof)
    umbral = P.umbral_optimo(train[P.TARGET].to_numpy()[en_cv], res.oof[en_cv],
                             train["staff_present"].to_numpy()[en_cv])
    hash_train = hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()

    with mlflow.start_run(run_name=f"reentrenamiento-lote-{lote_disparo}") as run:
        mlflow.log_params({**{k: v for k, v in candidato.named_steps["modelo"].get_params(deep=False).items()
                              if isinstance(v, (int, float, str, bool, type(None)))},
                           "semilla": P.SEMILLA, "version_champion": champion.numero,
                           "run_id_champion": champion.run_id, "datos_hash": hash_train,
                           "datos_snapshot_hash": info["snapshot_hash"],
                           "datos_tabla_origen": "PSET2.OBT.OBT_SALES_HOURLY + PSET2.MLOPS.LOTES/TARGETS",
                           "datos_filas_train": len(train), "ventana": "todo el histórico etiquetado",
                           "ventana_inicio": info["ventana_inicio"], "ventana_fin": info["ventana_fin"],
                           "lote_evaluacion": info["lote_evaluacion"],
                           "lotes_produccion_usados": ",".join(map(str, info["lotes_usados"])),
                           "lotes_excluidos_alerta_datos": ",".join(map(str, info["excluidos_por_alerta_de_datos"])) or "-",
                           "lotes_excluidos_decision": ",".join(map(str, info["excluidos_por_decision"])) or "-",
                           "umbral_decision": umbral})
        mlflow.log_metrics({k: v for k, v in res.metricas.items() if np.isfinite(v)})
        for i, w in enumerate(res.wape_pliegues):
            mlflow.log_metric("cv_wape_pliegue", w, step=i)
        mlflow.set_tags({"fase": "reentrenamiento", "motivo": motivo, "lote_disparo": str(lote_disparo),
                         "familia": "lineal", "git_commit": T.commit_git(),
                         "modelo_clase": type(candidato.named_steps["modelo"]).__name__})
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "entrenamiento.parquet"
            train.to_parquet(ruta, index=False)
            mlflow.log_artifact(str(ruta), artifact_path=str(Path(ARTEFACTO_DATOS).parent))
        ejemplo = train[P.ENTRADA].head(5)
        firma = infer_signature(ejemplo, res.oof[en_cv][:5].astype(float))
        info_modelo = mlflow.sklearn.log_model(
            res.modelo, name="modelo", signature=firma, input_example=ejemplo,
            serialization_format="cloudpickle", code_paths=[str(RAIZ / "src")],
            registered_model_name=T.MODELO_REGISTRADO)
        run_id = run.info.run_id

    cliente = MlflowClient()
    nueva = int(info_modelo.registered_model_version)
    cliente.set_model_version_tag(T.MODELO_REGISTRADO, str(nueva), "umbral_decision", f"{umbral:.2f}")
    cliente.set_model_version_tag(T.MODELO_REGISTRADO, str(nueva), "fase", "reentrenamiento")
    cliente.set_registered_model_alias(T.MODELO_REGISTRADO, "challenger", str(nueva))

    m_cp = _evaluar_en_lote(modelo_champion, champion.umbral, evaluacion)
    m_ch = _evaluar_en_lote(res.modelo, umbral, evaluacion)
    sigma_costo, sigma_wape = ref["costo_lote_std"], ref["wape_lote_std"]
    gana_negocio = m_ch["costo"] < m_cp["costo"] - sigma_costo
    no_empeora = m_ch["wape"] <= m_cp["wape"] + sigma_wape
    recomendacion = "promover" if (gana_negocio and no_empeora) else "mantener champion"
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({"eval_lote_wape_champion": m_cp["wape"], "eval_lote_wape_challenger": m_ch["wape"],
                            "eval_lote_costo_champion": m_cp["costo"], "eval_lote_costo_challenger": m_ch["costo"]})
        mlflow.set_tag("recomendacion", recomendacion)

    rid = str(uuid.uuid4())
    mlops.escribir(pd.DataFrame([{
        "reentrenamiento_id": rid, "creado_en": _ahora(), "lote_disparo": lote_disparo, "motivo": motivo,
        "run_id": run_id, "version_challenger": nueva, "version_champion": champion.numero,
        "lote_evaluacion": info["lote_evaluacion"], "wape_champion": m_cp["wape"],
        "wape_challenger": m_ch["wape"], "costo_champion": m_cp["costo"],
        "costo_challenger": m_ch["costo"], "sigma_costo": sigma_costo, "sigma_wape": sigma_wape,
        "recomendacion": recomendacion, "decision": "pendiente", "motivo_decision": None,
        "decidido_en": None, "lotes_entrenamiento": json.dumps(info)}]), "REENTRENAMIENTOS")

    resumen = (f"Lote de evaluación {info['lote_evaluacion']} (no lo vio ninguno de los dos). "
               f"Champion v{champion.numero}: costo ${m_cp['costo']:.2f}, WAPE {m_cp['wape']:.3f}. "
               f"Challenger v{nueva}: costo ${m_ch['costo']:.2f}, WAPE {m_ch['wape']:.3f}. "
               f"Regla: costo del challenger < costo del champion - {sigma_costo:.2f} "
               f"({'sí' if gana_negocio else 'no'}) y WAPE no peor en más de {sigma_wape:.3f} "
               f"({'sí' if no_empeora else 'no'}). Recomendación: {recomendacion.upper()}.")
    print(resumen)
    _salida_kestra(reentrenamiento_id=rid, run_id=run_id, version_challenger=nueva,
                   version_champion=champion.numero, recomendacion=recomendacion, resumen=resumen)


def decidir(rid: str, decision: str, motivo: str, excluir: list[int]) -> None:
    T.configurar()
    fila = mlops.leer("select * from REENTRENAMIENTOS where reentrenamiento_id = %s", (rid,)).iloc[0]
    cliente = MlflowClient()
    if decision == "aprobar":
        anterior = version("champion").numero
        cliente.set_registered_model_alias(T.MODELO_REGISTRADO, "anterior", str(anterior))
        cliente.set_registered_model_alias(T.MODELO_REGISTRADO, "champion", str(int(fila["version_challenger"])))
        cliente.set_model_version_tag(T.MODELO_REGISTRADO, str(int(fila["version_challenger"])),
                                      "aprobado", motivo)
        texto = f"champion: v{anterior} -> v{int(fila['version_challenger'])}"
    else:
        cliente.set_model_version_tag(T.MODELO_REGISTRADO, str(int(fila["version_challenger"])),
                                      "rechazado", motivo)
        texto = f"champion sin cambios (v{int(fila['version_champion'])})"
    if excluir:
        mlops.escribir(pd.DataFrame({"lote_id": excluir, "motivo": motivo, "excluido_en": _ahora()}),
                       "LOTES_EXCLUIDOS")
    contraria = (decision == "aprobar") != (fila["recomendacion"] == "promover")
    mlops.ejecutar("update REENTRENAMIENTOS set decision = %s, motivo_decision = %s, decidido_en = %s "
                   "where reentrenamiento_id = %s",
                   (("aprobado" if decision == "aprobar" else "rechazado")
                    + (" (contra la recomendación)" if contraria else ""), motivo, _ahora(), rid))
    with mlflow.start_run(run_id=fila["run_id"]):
        mlflow.set_tags({"decision": decision, "motivo_decision": motivo})
    print(f"Decisión: {decision} — {motivo}. {texto}. Lotes excluidos: {excluir or 'ninguno'}")
    _salida_kestra(decision=decision, texto=texto)


def mover_alias(numero: int) -> None:
    """Rollback o promoción manual: 'champion' pasa a la versión indicada."""
    T.configurar()
    cliente = MlflowClient()
    actual = version("champion").numero
    cliente.set_registered_model_alias(T.MODELO_REGISTRADO, "champion", str(numero))
    print(f"champion: v{actual} -> v{numero} (la API lo toma sin reconstruir la imagen)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="accion", required=True)
    e = sub.add_parser("entrenar")
    e.add_argument("--motivo", required=True)
    e.add_argument("--lote-disparo", type=int, required=True)
    d = sub.add_parser("decidir")
    d.add_argument("--id", required=True)
    d.add_argument("--decision", choices=["aprobar", "rechazar"], required=True)
    d.add_argument("--motivo", required=True)
    d.add_argument("--excluir", default="")
    al = sub.add_parser("alias")
    al.add_argument("--version", type=int, required=True)
    a = ap.parse_args()
    if a.accion == "entrenar":
        entrenar(a.motivo, a.lote_disparo)
    elif a.accion == "decidir":
        decidir(a.id, a.decision, a.motivo,
                [int(x) for x in a.excluir.replace(" ", "").split(",") if x])
    else:
        mover_alias(a.version)
