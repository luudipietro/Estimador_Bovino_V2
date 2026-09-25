"""Features geométricas 2D a partir de keypoints (en píxeles).

Tres familias, todas derivadas de las 36 distancias entre los 9 keypoints:
- d_<a>_<b>   : distancia en píxeles (depende de la distancia a la cámara; solo como insumo)
- r_<a>_<b>   : distancia / largo corporal (wither-pinbone). Adimensional: NO requiere marcador.
- s_<a>_<b>   : distancia / diámetro del sticker. Escala real relativa: SÍ requiere marcador.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from common.acmeai import KP_CANONICOS

PARES = list(combinations(KP_CANONICOS, 2))
REFERENCIA = ("wither", "pinbone")  # largo corporal, normalizador de los ratios


def distancias_px(kp: pd.DataFrame) -> pd.DataFrame:
    """kp con columnas <kp>_x / <kp>_y -> DataFrame de 36 distancias d_<a>_<b> en píxeles."""
    out = {}
    for a, b in PARES:
        out[f"d_{a}_{b}"] = np.hypot(kp[f"{a}_x"] - kp[f"{b}_x"], kp[f"{a}_y"] - kp[f"{b}_y"])
    return pd.DataFrame(out, index=kp.index)


def ratios(d: pd.DataFrame) -> pd.DataFrame:
    """35 razones respecto del largo corporal (se omite la propia referencia)."""
    ref = d[f"d_{REFERENCIA[0]}_{REFERENCIA[1]}"]
    cols = {f"r_{a}_{b}": d[f"d_{a}_{b}"] / ref for a, b in PARES if (a, b) != REFERENCIA}
    return pd.DataFrame(cols, index=d.index)


def normalizadas_por_sticker(d: pd.DataFrame, diam_sticker_px: pd.Series) -> pd.DataFrame:
    """36 distancias expresadas en diámetros de sticker (calibración por marcador físico)."""
    cols = {f"s_{a}_{b}": d[f"d_{a}_{b}"] / diam_sticker_px for a, b in PARES}
    return pd.DataFrame(cols, index=d.index)
