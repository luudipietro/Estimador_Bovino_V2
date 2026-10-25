"""Pruebas de la validación de fotos (no necesitan modelos ni contenedor).

    python -m unittest discover -s API_estimacion/tests -t API_estimacion
"""
import io
import unittest

from PIL import Image

from app.config import LADO_MAX_PX, MAX_BYTES
from app.imagenes import ErrorImagen, preparar


def jpeg(ancho, alto, orientacion=None, formato="JPEG") -> bytes:
    im = Image.new("RGB", (ancho, alto), (120, 90, 60))
    buf = io.BytesIO()
    kw = {}
    if orientacion:
        exif = Image.Exif()
        exif[0x0112] = orientacion
        kw["exif"] = exif
    im.save(buf, formato, **kw)
    return buf.getvalue()


class TestPreparar(unittest.TestCase):
    def codigo(self, datos):
        with self.assertRaises(ErrorImagen) as c:
            preparar(datos)
        return c.exception.status

    def test_vacio(self):
        self.assertEqual(self.codigo(b""), 400)

    def test_demasiado_grande_en_bytes(self):
        self.assertEqual(self.codigo(b"0" * (MAX_BYTES + 1)), 413)

    def test_no_es_imagen(self):
        self.assertEqual(self.codigo(b"hola, esto es texto renombrado a .jpg"), 415)

    def test_formato_no_soportado(self):
        self.assertEqual(self.codigo(jpeg(800, 600, formato="BMP")), 415)

    def test_imagen_chica(self):
        self.assertEqual(self.codigo(jpeg(400, 300)), 422)

    def test_tamano_correcto_sin_aviso(self):
        im, aviso = preparar(jpeg(LADO_MAX_PX, 768))
        self.assertEqual(im.size, (LADO_MAX_PX, 768))
        self.assertIsNone(aviso)

    def test_mpo_de_celular_aceptado(self):
        # Pillow rotula como MPO a muchos JPEG de celular; no deben rechazarse
        im, _ = preparar(jpeg(1024, 768, formato="MPO"))
        self.assertEqual(im.size, (1024, 768))

    def test_png_aceptado(self):
        im, _ = preparar(jpeg(900, 700, formato="PNG"))
        self.assertEqual(im.mode, "RGB")

    def test_foto_grande_se_acepta_con_aviso(self):
        im, aviso = preparar(jpeg(2000, 1500))
        self.assertEqual(im.size, (2000, 1500))  # el achicado lo hace el módulo, no la API
        self.assertIn("2000x1500", aviso)

    def test_orientacion_exif_se_aplica(self):
        # orientación 6 = rotada 90°: el archivo guarda 1000x800 pero hay que mostrarla 800x1000
        im, _ = preparar(jpeg(1000, 800, orientacion=6))
        self.assertEqual(im.size, (800, 1000))


if __name__ == "__main__":
    unittest.main()
