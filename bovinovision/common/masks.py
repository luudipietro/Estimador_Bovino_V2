"""Máscaras semánticas de AcmeAI (PNG RGBA de 3 colores) -> vaca y sticker de referencia.

Colores verificados en muestras de B3 y B4: fondo (0,255,193), sticker (0,117,255), vaca (255,30,249).
"""
import cv2
import numpy as np

COLOR_FONDO = (0, 255, 193)
COLOR_STICKER = (0, 117, 255)
COLOR_VACA = (255, 30, 249)
COLOR_SIN_ETIQUETA = (0, 0, 0)  # píxeles no anotados (aparece en algunas imágenes); se trata como fondo


def separar(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGBA (H, W, 4) -> (mascara_vaca, mascara_sticker) booleanas. Falla si aparecen colores desconocidos."""
    rgb = rgba[..., :3]
    vaca = np.all(rgb == COLOR_VACA, axis=-1)
    sticker = np.all(rgb == COLOR_STICKER, axis=-1)
    fondo = np.all(rgb == COLOR_FONDO, axis=-1) | np.all(rgb == COLOR_SIN_ETIQUETA, axis=-1)
    resto = ~(vaca | sticker | fondo)
    assert resto.mean() < 0.01, f"colores de máscara desconocidos en {resto.mean():.1%} de los píxeles"
    return vaca, sticker


def estadisticas(vaca: np.ndarray, sticker: np.ndarray) -> dict:
    """Área de la vaca, bbox de la vaca y geometría del sticker (mayor componente conexa)."""
    out = {"area_vaca_px": int(vaca.sum()), "area_sticker_px": int(sticker.sum())}
    ys, xs = np.nonzero(vaca)
    if len(xs):
        out.update(vaca_x0=int(xs.min()), vaca_x1=int(xs.max()), vaca_y0=int(ys.min()), vaca_y1=int(ys.max()))
    n, _, stats, cent = cv2.connectedComponentsWithStats(sticker.astype(np.uint8), connectivity=8)
    out["n_comp_sticker"] = n - 1
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        w, h, area = (int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT]),
                      int(stats[k, cv2.CC_STAT_AREA]))
        out.update(
            sticker_area_px=area, sticker_cx=float(cent[k, 0]), sticker_cy=float(cent[k, 1]),
            sticker_w_px=w, sticker_h_px=h,
            sticker_diam_px=float(2 * np.sqrt(area / np.pi)),  # diámetro del círculo de igual área
            sticker_redondez=float(min(w, h) / max(w, h)),  # ~1 si es un círculo completo y no ocluido
        )
    return out
