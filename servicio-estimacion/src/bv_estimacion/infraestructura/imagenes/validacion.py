"""Validación de una imagen recibida (RF-15): formato real, tamaño, decodificación y resolución.

Es síncrona y gasta CPU (decodifica la imagen entera): quien la llame desde un endpoint tiene que correrla en un
hilo (`anyio.to_thread.run_sync`) para no bloquear el event loop.
"""

import io
from dataclasses import dataclass

import pillow_heif
from PIL import Image

from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio

pillow_heif.register_heif_opener()  # habilita HEIC en Pillow (R-08)

FORMATO_JPEG = "jpeg"
FORMATO_PNG = "png"
FORMATO_HEIC = "heic"

_MARCAS_HEIC = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}


@dataclass(frozen=True, slots=True)
class ImagenValidada:
    formato: str
    ancho: int
    alto: int
    advertencias: tuple[str, ...]  # sin el número de imagen: lo agrega el caso de uso


def detectar_formato(datos: bytes) -> str | None:
    """Mira los *magic bytes*, no la extensión ni el `Content-Type`, que el cliente puede falsear."""
    if datos.startswith(b"\xff\xd8\xff"):
        return FORMATO_JPEG
    if datos.startswith(b"\x89PNG\r\n\x1a\n"):
        return FORMATO_PNG
    if datos[4:8] == b"ftyp" and datos[8:12] in _MARCAS_HEIC:
        return FORMATO_HEIC
    return None


def validar_imagen(
    datos: bytes,
    *,
    max_bytes: int,
    max_pixeles: int,
    resolucion_minima_px: int,
) -> ImagenValidada:
    """Lanza `ErrorDeDominio` con `imagen_muy_grande` (413), `formato_no_admitido` (415), `imagen_corrupta` (422) o
    `resolucion_insuficiente` (422). Los mensajes no incluyen detalles internos."""
    if len(datos) > max_bytes:
        raise ErrorDeDominio(CodigoError.IMAGEN_MUY_GRANDE, "la imagen supera el tamaño máximo permitido")
    if not datos:
        raise ErrorDeDominio(CodigoError.IMAGEN_CORRUPTA, "el archivo llegó vacío")

    formato = detectar_formato(datos)
    if formato is None:
        raise ErrorDeDominio(CodigoError.FORMATO_NO_ADMITIDO, "el archivo no es JPEG, PNG ni HEIC")

    try:
        with Image.open(io.BytesIO(datos)) as imagen:
            ancho, alto = imagen.size  # sale del encabezado: todavía no se decodificó nada
            if ancho * alto > max_pixeles:
                raise ErrorDeDominio(CodigoError.IMAGEN_MUY_GRANDE, "la imagen tiene demasiados píxeles")
            imagen.load()  # decodifica: detecta archivos truncados o dañados
            tiene_exif = bool(imagen.getexif())  # solo se mira si hay metadata; no se leen ni guardan sus valores
    except ErrorDeDominio:
        raise
    except Image.DecompressionBombError as error:
        raise ErrorDeDominio(CodigoError.IMAGEN_MUY_GRANDE, "la imagen tiene demasiados píxeles") from error
    except Exception as error:  # los decodificadores de imágenes no confiables lanzan casi cualquier cosa
        raise ErrorDeDominio(CodigoError.IMAGEN_CORRUPTA, "no se pudo decodificar la imagen") from error

    if ancho * alto < resolucion_minima_px:
        raise ErrorDeDominio(CodigoError.RESOLUCION_INSUFICIENTE, "la resolución de la imagen es insuficiente")

    advertencias = () if tiene_exif else ("no trae metadata EXIF",)  # R-07: sin metadata hay advertencia
    return ImagenValidada(formato=formato, ancho=ancho, alto=alto, advertencias=advertencias)
