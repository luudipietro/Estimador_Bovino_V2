"""Modelo de cámara pinhole: convierte píxeles + profundidad en coordenadas métricas 3D (sec. 8 del plan)."""
import numpy as np


def deproyectar(u, v, z, fx, fy, cx, cy) -> np.ndarray:
    """(u, v) en píxeles y z (profundidad, m) -> puntos (X, Y, Z) en la cámara. Devuelve array (..., 3)."""
    u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(z, float))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def proyectar(puntos, fx, fy, cx, cy):
    """Inversa de deproyectar: (X, Y, Z) -> (u, v, z)."""
    p = np.asarray(puntos, float)
    z = p[..., 2]
    return p[..., 0] * fx / z + cx, p[..., 1] * fy / z + cy, z


def distancia_3d(p1, p2) -> float:
    return float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p2, float)))


def focal_px_desde_35mm(focal_35mm: float | None, ancho_px: int, alto_px: int) -> float | None:
    """Focal en píxeles a partir de la focal equivalente 35 mm (f_px = f35 / 36 * lado mayor).
    Devuelve None si no hay dato (sin sensor conocido no se puede convertir)."""
    if not focal_35mm:
        return None
    return focal_35mm / 36.0 * max(ancho_px, alto_px)
