# Cambios respecto de los PSets 1 a 3

Registro de las observaciones recibidas, las decisiones que cambiaron al integrar el proyecto y los cambios de estructura. Cada entrada dice qué se cambió y en qué archivo.

## 1. Observaciones recibidas

| PSet | Observación | Qué se cambió | Dónde |
|---|---|---|---|
| 1 | Viabilidad: una comparación antes/después del refuerzo confunde su efecto con la estacionalidad (fin de mes, feriados, temporada). | El diseño de evaluación del piloto pasó a *switchback* aleatorizado en la matriz (horas o días asignados al azar con y sin refuerzo) con diferencias en diferencias contra la sucursal 2 como control. Por eso la ingesta trae las tres sucursales y la sucursal 2 se conserva en la OBT con su rol experimental. En el proyecto final la sucursal 2 se usa también para inducir la deriva de datos del lote 25. | `kestra/ingesta.yml`, `dbt/models/gold/dim_branch.sql` (`experiment_role`), `docs/historial/PSet2_memo_Torres_Chicaiza.pdf` §1; memo del PSet 4 §10 |

No se recibieron otras observaciones escritas de los PSets 2 y 3 al momento de la entrega.

## 2. Decisiones actualizadas

| Decisión | Antes | Ahora | Por qué | Dónde |
|---|---|---|---|---|
| Fin de la ventana de análisis | 2026-11-07 (desde el 08-11 trabaja una cajera) | 2026-11-25 | La simulación de producción necesita datos posteriores al entrenamiento. La cajera solo cobra: no cambia la dotación de vendedores. | `dbt/dbt_project.yml`, `.env.example`, `spark/build_obt.py` |
| Dotación de vendedores desde el 08-11 | Sin regla (el seed terminaba el 07-11) | Segundo período del seed con la misma dotación, vigente hasta el 31-12 | Confirmado con la administración: la cajera no vende. El efecto posible sobre la venta por vendedor lo vigila el monitoreo. | `dbt/seeds/staffing_rules.csv`, `dbt/seeds/properties.yml` |
| Modelo en producción | Ridge v1 (PSet 3), umbral de carga prevista 6,75 | Ridge v3, mismos hiperparámetros, reentrenado con 3 semanas de producción; umbral 5,25 | Reentrenamiento disparado por desempeño (lotes 18-19) y aprobado según la regla de promoción. El umbral se volvió a elegir con predicciones fuera de pliegue, como en el PSet 3. | Model Registry `ventas-hora-matriz-modelo@champion`; `docs/decisiones.csv` |
| Uso de la métrica de negocio | Solo para elegir el umbral y comparar modelos | También como alerta (costo del modelo frente a la regla en 14 lotes) y como criterio de promoción y de retirada | El PSet 3 mostró que un mejor WAPE no garantiza una mejor decisión. | `monitoreo/alertas.py`, `src/reentrenar.py` |
| Cifras del pitch del PSet 1 | Simulación retrospectiva: WAPE de la regla 38,4 %, piloto de 80 h ≈ $212 | Resultado de la simulación de producción (31 días): costo del modelo $350, regla $372, no hacer nada $489 | Coherencia con la sección 9 del memo del PSet 4. | `docs/PSet4_presentacion_Torres_Chicaiza.pdf` |

Sin cambios: target (ventas por hora de la matriz), unidad (hora abierta), horizonte (plan semanal, features con rezago ≥ 7 días), métrica técnica (WAPE), métrica de negocio (costo de FN × $7,19 + FP × $2,65), línea base (promedio de la misma hora en las 4 semanas anteriores), grano de la OBT (sucursal × fecha × hora abierta) y fuente de datos.

## 3. Cambios de estructura y de código

| Cambio | Motivo | Archivos |
|---|---|---|
| Los repositorios `pset_2` y `pset_3` se unieron en `proyecto_final/` con un solo `docker-compose.yml` (Kestra, Spark, MLflow, API, tablero). El namespace de Kestra pasó de `pset2` a `proyecto_final`; las imágenes, de `pset2-*`/`pset3-*` a `proyecto-final-*`. | Requisito del proyecto final. | `docker-compose.yml`, `Makefile`, `kestra/*.yml` |
| Los flujos del PSet 2 se renombraron: `pset2.ingest_erp.yml` → `ingesta.yml` y `pset2.transform.yml` → `transformacion.yml`. | Estructura pedida (`kestra/ingesta.yml`). | `kestra/` |
| El historial de MLflow (104 corridas y el Model Registry) se migró del volumen del PSet 3 al del proyecto. | Mantener la trazabilidad de los modelos del PSet 3. | volúmenes `proyecto_final_mlflow-*` |
| `notebooks/06_reproduccion.ipynb` se retiró. La prueba de que el modelo reproduce la métrica registrada pasó a `tests/test_modelo.py`, que corre en CI. El número 06 lo ocupa ahora la simulación. | Estructura pedida (`06_simulacion_produccion`, `07_analisis_monitoreo`). | `tests/test_modelo.py`, `notebooks/06_*`, `notebooks/07_*` |
| `src/datos.py`: la conexión a Snowflake es pública (`conectar`), acepta la llave como contenido (secreto de Kestra) y hay un segundo snapshot para producción. | Las tareas de Kestra y la API usan la misma conexión. | `src/datos.py` |
| Nuevos módulos: `src/mlops.py` (tablas MLOPS), `src/produccion.py` (versiones y referencias de monitoreo), `src/simulacion.py`, `src/reentrenar.py`, `src/exportar.py`, `monitoreo/`, `api/`, `tablero/`. | PSet 4. | — |
| Hallazgo de las pruebas de comportamiento: por colinealidad, Ridge asigna coeficiente negativo a `sales_lag_28d` y a `ventas_dia_lag_7d`. La prueba verifica el aumento conjunto de la historia, que es el caso que ocurre en los datos. | Documentar el comportamiento real del modelo. | `tests/test_comportamiento.py`; memo §4 |
| `docs/` reorganizado: memos anteriores en `docs/historial/`, documentación de datos del PSet 2 en `docs/datos/`, figuras y capturas de ambos PSets en `docs/figuras/` y `docs/capturas/`. | Estructura pedida. | `docs/` |
