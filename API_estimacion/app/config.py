"""Constantes y parámetros de la API. Lo que depende del entorno se lee de variables de entorno."""
import os
from pathlib import Path

# Contrato de imagen: el estimador reduce todo a 1024 px de lado mayor antes de procesar (estimador.py, LADO_MAX).
# Achicar MÁS en el cliente degrada el resultado: el detector y la CNN trabajan sobre recortes del animal.
LADO_MAX_PX = 1024
LADO_MIN_PX = 640
# MPO: así rotula Pillow a muchos JPEG de celular (multi-imagen); son JPEG válidos
FORMATOS_ACEPTADOS = ("JPEG", "MPO", "PNG")
FORMATOS_PUBLICOS = ("JPEG", "PNG")  # lo que se le informa al cliente

# Una foto a 1024 px pesa menos de 1 MB; este tope detecta clientes que mandan la foto original sin achicar.
MAX_BYTES = int(os.environ.get("MAX_MB", "5")) * 1024 * 1024

MODELOS_DIR = Path(os.environ.get("MODELOS_DIR", "/modelos"))

# El modelo actual no extrapola: satura cerca de ~270 kg. Un valor cercano al techo puede ser un animal mucho más pesado.
PESO_SATURACION_KG = float(os.environ.get("PESO_SATURACION_KG", "240"))
