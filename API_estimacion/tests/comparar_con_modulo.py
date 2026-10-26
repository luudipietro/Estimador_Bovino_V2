"""Compara la API (foto achicada por el cliente) contra el módulo usado directamente (foto original).

    python API_estimacion/tests/comparar_con_modulo.py foto1.jpg foto2.jpg ... [--url http://localhost:8000]
                                                    [--modelos estimador/modelos]

Necesita la API corriendo y torch/ultralytics/timm instalados (el entorno de estimador/). Reporta, por foto, el peso
del módulo con la foto original, el peso de la API con la foto achicada, la diferencia y el ahorro de red.
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "estimador"))
sys.path.insert(0, str(RAIZ / "API_estimacion" / "cliente"))

from PIL import Image, ImageOps  # noqa: E402

from enviar_foto import enviar, preparar_foto  # noqa: E402
from estimador import EstimadorPeso  # noqa: E402

# El achicado del cliente es idéntico al interno del módulo; lo único que cambia es la recompresión JPEG. Medido sobre
# ~30 fotos reales: desvío medio ≈2.5%, con casos de hasta ≈15%, sin mejora al subir la calidad JPEG.
# Con --formato png (sin pérdida) la API debe coincidir exactamente con el módulo.
# Con JPEG la tolerancia no exige igualdad: detecta divergencias groseras (orientación, formato, resolución mal aplicados).
TOLERANCIA_REL = {"png": 0.005, "jpeg": 0.20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fotos", nargs="+")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--modelos", default=str(RAIZ / "estimador" / "modelos"))
    ap.add_argument("--formato", choices=["jpeg", "png"], default="jpeg")
    a = ap.parse_args()

    modulo = EstimadorPeso(a.modelos)
    print(f"{'foto':<34}{'módulo (kg)':>12}{'API (kg)':>10}{'dif':>8}{'KB orig':>9}{'KB env':>8}")
    peor = 0.0
    for f in a.fotos:
        # El módulo no aplica la orientación EXIF por sí solo (una foto vertical de celular no se detecta); la API sí.
        directo = modulo.estimar(ImageOps.exif_transpose(Image.open(f)))
        jpeg = preparar_foto(f, a.formato)
        api = enviar(jpeg, a.url, Path(f).name)
        p0, p1 = directo["peso_kg"], api["peso_kg"]
        if p0 is None or p1 is None:
            print(f"{Path(f).name:<34}{str(p0):>12}{str(p1):>10}   (sin peso: {api.get('motivo')})")
            continue
        dif = abs(p0 - p1)
        peor = max(peor, dif / p0)
        print(f"{Path(f).name:<34}{p0:>12.1f}{p1:>10.1f}{dif:>8.1f}"
              f"{Path(f).stat().st_size / 1024:>9.0f}{len(jpeg) / 1024:>8.0f}")
    tol = TOLERANCIA_REL[a.formato]
    print(f"\ndiferencia relativa máxima: {peor:.1%} (tolerancia {tol:.1%} para {a.formato})")
    sys.exit(0 if peor <= tol else 1)


if __name__ == "__main__":
    main()
