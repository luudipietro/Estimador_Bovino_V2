"""Imágenes generadas con Pillow al correr los tests: no hay fotos de campo (ni su GPS) en el repo."""

import io
import os

from PIL import Image


def _pixeles(ancho: int, alto: int, *, ruido: bool) -> Image.Image:
    if ruido:  # incompresible: el archivo es grande y un truncado se nota
        return Image.frombytes("RGB", (ancho, alto), os.urandom(ancho * alto * 3))
    return Image.new("RGB", (ancho, alto), (120, 90, 60))


def jpeg(ancho: int = 100, alto: int = 100, *, con_exif: bool = False, ruido: bool = False) -> bytes:
    buffer = io.BytesIO()
    imagen = _pixeles(ancho, alto, ruido=ruido)
    if con_exif:
        exif = Image.Exif()
        exif[0x010F] = "Prueba"  # Make
        imagen.save(buffer, "JPEG", exif=exif)
    else:
        imagen.save(buffer, "JPEG")
    return buffer.getvalue()


def png(ancho: int = 100, alto: int = 100) -> bytes:
    buffer = io.BytesIO()
    _pixeles(ancho, alto, ruido=False).save(buffer, "PNG")
    return buffer.getvalue()


def heic(ancho: int = 100, alto: int = 100) -> bytes:
    import pillow_heif

    pillow_heif.register_heif_opener()
    buffer = io.BytesIO()
    _pixeles(ancho, alto, ruido=False).save(buffer, "HEIF")
    return buffer.getvalue()


def gif() -> bytes:
    buffer = io.BytesIO()
    Image.new("P", (10, 10)).save(buffer, "GIF")
    return buffer.getvalue()


def bmp() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, "BMP")
    return buffer.getvalue()
