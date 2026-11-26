"""Deriva de datos: PSI por feature, con KS (numéricas) y chi-cuadrado (categóricas) de respaldo.

Por qué estos umbrales y no los clásicos 0,10 / 0,25: esos valores suponen muestras de
miles de filas. Un lote de este proyecto es un día de tienda (9 a 11 horas), y con tan
pocas filas el PSI de un día normal ya es alto solo por azar. Por eso el umbral de cada
feature se calibra con los propios datos de entrenamiento: se calcula el PSI de cada día de
entrenamiento contra el resto (mismo día de la semana) y el umbral es el percentil 99 de esa
distribución, nunca menor que 0,25. Un lote supera el umbral cuando su PSI es más alto que el
del 99 % de los días normales.

Referencia: siempre el mismo día de la semana. El personal y la venta cambian por diseño
entre días (domingo y lunes hay un solo vendedor); comparar un lunes contra toda la semana
marcaría deriva todos los lunes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

N_CORTES = 5          # quintiles de la referencia: con ~11 filas, más cortes son puro ruido
EPS = 1e-4
PSI_MINIMO = 0.25     # piso del umbral calibrado (el valor clásico de "cambio significativo")

NUMERICAS_DERIVA = ["sales_lag_7d", "sales_lag_14d", "sales_lag_21d", "sales_lag_28d",
                    "sales_avg_4w_same_hour"]
CATEGORICAS_DERIVA = ["staff_present"]
# hour_of_day, iso_day_of_week, ventana_pago, is_holiday e is_last_open_hour son calendario:
# su distribución en un lote está fijada por la fecha, no por los datos. Se vigilan como
# «categorías no vistas», no con PSI. ventas_dia_lag_7d tiene un solo valor por día (es el
# total del día de hace una semana): en un lote diario su PSI no mide nada, así que se vigila
# con el control de valores fuera del rango de entrenamiento.


def _cortes(referencia: np.ndarray) -> np.ndarray:
    q = np.unique(np.quantile(referencia, np.linspace(0, 1, N_CORTES + 1)[1:-1]))
    return np.concatenate([[-np.inf], q, [np.inf]])


def _proporciones(x: np.ndarray, cortes: np.ndarray) -> np.ndarray:
    conteo = np.histogram(x, bins=cortes)[0].astype(float)
    return np.clip(conteo / max(conteo.sum(), 1), EPS, None)


def psi(referencia, actual) -> float:
    """PSI con cortes por cuantiles de la referencia. Ignora nulos (se vigilan aparte)."""
    ref = pd.Series(referencia, dtype=float).dropna().to_numpy()
    act = pd.Series(actual, dtype=float).dropna().to_numpy()
    if len(ref) == 0 or len(act) == 0:
        return float("nan")
    cortes = _cortes(ref)
    p, q = _proporciones(ref, cortes), _proporciones(act, cortes)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorico(referencia, actual) -> float:
    ref = pd.Series(referencia).dropna()
    act = pd.Series(actual).dropna()
    cats = sorted(set(ref) | set(act))
    p = np.clip(ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(), EPS, None)
    q = np.clip(act.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy(), EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks(referencia, actual) -> float:
    """p-valor de Kolmogorov-Smirnov de dos muestras."""
    ref = pd.Series(referencia, dtype=float).dropna()
    act = pd.Series(actual, dtype=float).dropna()
    if len(ref) == 0 or len(act) == 0:
        return float("nan")
    return float(stats.ks_2samp(ref, act).pvalue)


def chi2(referencia, actual) -> float:
    """p-valor de chi-cuadrado: conteos del lote contra las proporciones de la referencia."""
    ref = pd.Series(referencia).dropna()
    act = pd.Series(actual).dropna()
    cats = sorted(set(ref) | set(act))
    esperado = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() * len(act)
    observado = act.value_counts().reindex(cats, fill_value=0).to_numpy()
    if (esperado == 0).any():           # categoría nueva: imposible bajo la referencia
        return 0.0
    if len(cats) < 2:
        return 1.0
    return float(stats.chisquare(observado, esperado).pvalue)


def calibrar_umbrales(entrenamiento: pd.DataFrame, percentil: float = 99) -> dict[str, float]:
    """Umbral de PSI por feature: percentil 99 del PSI de cada día de entrenamiento contra
    los demás días del mismo día de la semana (piso PSI_MINIMO)."""
    umbrales = {}
    for f in NUMERICAS_DERIVA + CATEGORICAS_DERIVA:
        valores = []
        for fecha, dia in entrenamiento.groupby("hour_date"):
            resto = entrenamiento[(entrenamiento["iso_day_of_week"] == dia["iso_day_of_week"].iloc[0])
                                  & (entrenamiento["hour_date"] != fecha)]
            fn = psi_categorico if f in CATEGORICAS_DERIVA else psi
            v = fn(resto[f], dia[f])
            if np.isfinite(v):
                valores.append(v)
        umbrales[f] = max(PSI_MINIMO, float(np.percentile(valores, percentil)))
    return umbrales


def deriva_lote(lote: pd.DataFrame, entrenamiento: pd.DataFrame) -> pd.DataFrame:
    """PSI y prueba de respaldo de cada feature del lote contra el mismo día de la semana."""
    dia = lote["iso_day_of_week"].iloc[0]
    ref = entrenamiento[entrenamiento["iso_day_of_week"] == dia]
    filas = []
    for f in NUMERICAS_DERIVA:
        filas.append({"feature": f, "psi": psi(ref[f], lote[f]), "prueba": "ks",
                      "p_valor": ks(ref[f], lote[f])})
    for f in CATEGORICAS_DERIVA:
        filas.append({"feature": f, "psi": psi_categorico(ref[f], lote[f]), "prueba": "chi2",
                      "p_valor": chi2(ref[f], lote[f])})
    return pd.DataFrame(filas)
