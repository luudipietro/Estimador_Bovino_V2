"""Métricas de peso. Siempre se reporta MAPE junto con la mejora sobre predecir la mediana."""
import numpy as np
import pandas as pd

from config import BANDAS_PESO


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true) / y_true) * 100)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def baseline_mediana(y_train, y_test) -> np.ndarray:
    """Predice siempre la mediana del train: piso contra el cual se mide toda mejora."""
    return np.full(len(y_test), np.median(y_train), dtype=float)


def mape_por_banda(y_true, y_pred, bandas=BANDAS_PESO) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    filas = []
    for lo, hi in zip(bandas[:-1], bandas[1:]):
        m = (y_true >= lo) & (y_true < hi)
        if m.any():
            filas.append({"banda_kg": f"{lo}-{hi if hi < 10_000 else '+'}", "n": int(m.sum()),
                          "mape": mape(y_true[m], y_pred[m])})
    return pd.DataFrame(filas)


def reporte(y_train, y_true, y_pred, nombre="modelo") -> dict:
    """MAPE del modelo, MAPE de la mediana y mejora relativa (positivo = mejor que la mediana)."""
    m_modelo = mape(y_true, y_pred)
    m_base = mape(y_true, baseline_mediana(y_train, y_true))
    return {"modelo": nombre, "n": len(y_true), "mape": m_modelo, "mape_mediana": m_base,
            "mejora_pct": (m_base - m_modelo) / m_base * 100}
