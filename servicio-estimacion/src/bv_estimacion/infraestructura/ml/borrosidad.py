"""Medición de la nitidez de la foto para R-24. Solo mide: decidir si es "borrosa" es del dominio (`es_borrosa`)."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

MIN_LADO_PX = 3  # el Laplaciano de 3x3 necesita al menos un píxel con todos sus vecinos


def varianza_laplaciano(gris: NDArray[np.floating]) -> float | None:
    """Varianza del Laplaciano (núcleo de 4 vecinos) sobre los píxeles interiores. Más alta = más nítida.

    `None` si la imagen es demasiado chica para medirla.
    """
    alto, ancho = gris.shape
    if alto < MIN_LADO_PX or ancho < MIN_LADO_PX:
        return None
    g = gris.astype(np.float64, copy=False)
    laplaciano = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(laplaciano.var())


def medir_borrosidad(imagen: Image.Image, caja: Sequence[float] | None, lado_max_px: int) -> float | None:
    """Varianza del Laplaciano del recorte del animal.

    `estimador` reduce la foto a `lado_max_px` (lado mayor) **antes** de calcular la caja que devuelve en `recorte`,
    así que acá se reduce igual antes de cortar: la caja está en las coordenadas de la imagen reducida. Sin caja,
    o con una caja vacía, se mide la imagen entera.
    """
    imagen = imagen.convert("RGB")
    factor = lado_max_px / max(imagen.size)
    if factor < 1:
        imagen = imagen.resize((round(imagen.width * factor), round(imagen.height * factor)), Image.Resampling.LANCZOS)

    if caja is not None:
        x0, y0 = max(0, int(caja[0])), max(0, int(caja[1]))
        x1, y1 = min(imagen.width, int(caja[2])), min(imagen.height, int(caja[3]))
        if x1 > x0 and y1 > y0:
            imagen = imagen.crop((x0, y0, x1, y1))

    return varianza_laplaciano(np.asarray(imagen.convert("L")))
