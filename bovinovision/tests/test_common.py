import numpy as np
import pandas as pd
import pytest

from common import geometry, metrics, splits


def test_pinhole_ida_y_vuelta():
    fx = fy = 1500.0
    cx, cy = 960.0, 540.0
    p = np.array([[0.5, -0.2, 3.0], [1.2, 0.4, 5.5]])
    u, v, z = geometry.proyectar(p, fx, fy, cx, cy)
    assert np.allclose(geometry.deproyectar(u, v, z, fx, fy, cx, cy), p)


def test_distancia_metrica_exacta():
    # dos puntos a 1.5 m entre sí, a 4 m de la cámara: la distancia 3D debe recuperarse
    fx = fy = 1200.0
    cx, cy = 640.0, 360.0
    a, b = np.array([-0.75, 0.0, 4.0]), np.array([0.75, 0.0, 4.0])
    ua, va, za = geometry.proyectar(a, fx, fy, cx, cy)
    ub, vb, zb = geometry.proyectar(b, fx, fy, cx, cy)
    pa = geometry.deproyectar(ua, va, za, fx, fy, cx, cy)
    pb = geometry.deproyectar(ub, vb, zb, fx, fy, cx, cy)
    assert geometry.distancia_3d(pa, pb) == pytest.approx(1.5)


def test_error_de_escala_se_propaga_al_cubo():
    # justifica el análisis de sensibilidad: 10% de error en escala -> ~33% en volumen/peso
    assert (1.10 ** 3 - 1) == pytest.approx(0.331, abs=1e-3)


def test_focal_desde_35mm():
    assert geometry.focal_px_desde_35mm(26, 4000, 3000) == pytest.approx(26 / 36 * 4000)
    assert geometry.focal_px_desde_35mm(None, 4000, 3000) is None


def test_mape_y_mediana():
    assert metrics.mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert list(metrics.baseline_mediana([100, 200, 300], [1, 2, 3, 4])) == [200.0] * 4


def test_mejora_sobre_mediana():
    r = metrics.reporte([100, 200, 300], [100, 300], [100, 300], "perfecto")
    assert r["mape"] == 0 and r["mejora_pct"] == pytest.approx(100.0)


def test_split_por_animal_sin_fuga_y_determinista():
    ids = [f"a{i}" for i in range(100)] * 3  # 3 fotos por animal
    s = splits.asignar_split(ids)
    assert set(s.unique()) == {"train", "val", "test"}
    assert (s == "train").sum() == 70
    df = pd.DataFrame({"animal_id": ids})
    df["split"] = df["animal_id"].map(s)
    splits.verificar_sin_fuga(df)
    assert splits.asignar_split(ids).equals(s)


def test_verificar_sin_fuga_detecta_fuga():
    df = pd.DataFrame({"animal_id": ["a", "a"], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        splits.verificar_sin_fuga(df)
