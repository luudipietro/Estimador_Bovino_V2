import numpy as np
import pandas as pd
import pytest

from common import features
from common.acmeai import KP_CANONICOS


def _kp(escala=1.0):
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (5, len(KP_CANONICOS), 2)) * escala
    cols = {}
    for j, k in enumerate(KP_CANONICOS):
        cols[f"{k}_x"], cols[f"{k}_y"] = xy[:, j, 0], xy[:, j, 1]
    return pd.DataFrame(cols)


def test_cantidad_de_features():
    d = features.distancias_px(_kp())
    assert d.shape[1] == 36
    assert features.ratios(d).shape[1] == 35


def test_ratios_invariantes_a_la_escala():
    # duplicar el tamaño en píxeles (otra distancia a la cámara) no cambia los ratios: por eso solos no dan escala real
    r1 = features.ratios(features.distancias_px(_kp(1.0)))
    r2 = features.ratios(features.distancias_px(_kp(2.5)))
    assert np.allclose(r1, r2)


def test_normalizacion_por_sticker_corrige_la_escala():
    k1, k2 = _kp(1.0), _kp(2.0)
    s1 = features.normalizadas_por_sticker(features.distancias_px(k1), pd.Series(10.0, index=k1.index))
    s2 = features.normalizadas_por_sticker(features.distancias_px(k2), pd.Series(20.0, index=k2.index))
    assert np.allclose(s1, s2)


def test_distancia_conocida():
    kp = _kp()
    kp.loc[0, ["wither_x", "wither_y", "pinbone_x", "pinbone_y"]] = [0, 0, 3, 4]
    assert features.distancias_px(kp).loc[0, "d_wither_pinbone"] == pytest.approx(5.0)
