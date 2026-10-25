"""Validación y normalización de la foto que llega a la API."""
import io

from PIL import Image, ImageOps

from .config import FORMATOS_ACEPTADOS, FORMATOS_PUBLICOS, LADO_MAX_PX, LADO_MIN_PX, MAX_BYTES


class ErrorImagen(Exception):
    """La foto no se puede procesar. Lleva el código HTTP que corresponde."""

    def __init__(self, status: int, detalle: str):
        super().__init__(detalle)
        self.status = status
        self.detalle = detalle


def preparar(datos: bytes) -> tuple[Image.Image, str | None]:
    """Valida los bytes recibidos y devuelve (imagen RGB con la orientación aplicada, aviso o None)."""
    if not datos:
        raise ErrorImagen(400, "el archivo llegó vacío")
    if len(datos) > MAX_BYTES:
        raise ErrorImagen(413, f"la foto supera los {MAX_BYTES // (1024 * 1024)} MB; "
                               f"achicala a {LADO_MAX_PX} px de lado mayor antes de enviarla")
    try:
        im = Image.open(io.BytesIO(datos))
        formato = im.format
        im.load()
    except Exception:
        raise ErrorImagen(415, "el archivo no es una imagen válida") from None
    if formato not in FORMATOS_ACEPTADOS:
        raise ErrorImagen(415, f"formato {formato} no soportado; usar {' o '.join(FORMATOS_PUBLICOS)}")

    # El cliente puede haber conservado el tag EXIF de orientación; el estimador no lo aplica por sí solo.
    im = ImageOps.exif_transpose(im).convert("RGB")

    lado = max(im.size)
    if lado < LADO_MIN_PX:
        raise ErrorImagen(422, f"imagen demasiado chica ({im.width}x{im.height}); el lado mayor debe ser "
                               f"de al menos {LADO_MIN_PX} px y lo ideal es {LADO_MAX_PX}")
    aviso = None
    if lado > LADO_MAX_PX:
        aviso = (f"la foto llegó a {im.width}x{im.height}; el servidor la reduce a {LADO_MAX_PX} px de lado mayor. "
                 f"Achicala en el cliente para ahorrar red")
    return im, aviso
