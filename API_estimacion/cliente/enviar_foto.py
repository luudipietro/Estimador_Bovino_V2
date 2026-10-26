"""Cliente de ejemplo — y especificación de qué tiene que hacer la app antes de enviar la foto.

    python enviar_foto.py vaca.jpg [--url http://localhost:8000]

Requiere solo Pillow. Pasos que la app móvil debe reproducir:
  1. Aplicar la orientación EXIF a los píxeles (al reducir se pierde el tag y el servidor ya no puede corregirla).
  2. Reducir a 1024 px de lado mayor, conservando la relación de aspecto (nunca agrandar).
  3. Codificar JPEG, calidad ~90 (la foto pasa de varios MB a ~150-300 KB).
  4. POST multipart/form-data con el archivo en el campo `foto`.

No achicar por debajo de 1024: el servidor recorta al animal y reescala; con menos píxeles de partida pierde precisión.
"""
import argparse
import io
import json
import uuid
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

LADO_MAX_PX = 1024
CALIDAD_JPEG = 90


def preparar_foto(ruta: str | Path, formato: str = "jpeg") -> bytes:
    """Foto original -> bytes listos para enviar. `formato`: "jpeg" (~270 KB) o "png" (~1,3 MB, sin pérdida).

    La compresión JPEG agrega ruido al peso estimado: medido sobre ~30 fotos reales, ~2,5% en promedio y hasta ~15%
    en algunas, sin mejorar al subir la calidad. PNG da exactamente el mismo resultado que el módulo."""
    im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    f = LADO_MAX_PX / max(im.size)
    if f < 1:
        im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    buf = io.BytesIO()
    if formato == "png":
        im.save(buf, "PNG")
    else:
        im.save(buf, "JPEG", quality=CALIDAD_JPEG)
    return buf.getvalue()


def enviar(imagen: bytes, url: str, nombre: str = "foto.jpg", timeout: float = 60) -> dict:
    """POST multipart al endpoint /estimar usando solo la librería estándar."""
    limite = uuid.uuid4().hex
    tipo = "image/png" if imagen[:4] == b"\x89PNG" else "image/jpeg"
    cuerpo = (f"--{limite}\r\nContent-Disposition: form-data; name=\"foto\"; filename=\"{nombre}\"\r\n"
              f"Content-Type: {tipo}\r\n\r\n").encode() + imagen + f"\r\n--{limite}--\r\n".encode()
    req = urllib.request.Request(url.rstrip("/") + "/estimar", data=cuerpo, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("foto")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--formato", choices=["jpeg", "png"], default="jpeg")
    a = ap.parse_args()

    jpeg = preparar_foto(a.foto, a.formato)
    original = Path(a.foto).stat().st_size
    print(f"foto: {original / 1024:.0f} KB  ->  enviada: {len(jpeg) / 1024:.0f} KB "
          f"({100 * len(jpeg) / original:.0f}% del original)")
    r = enviar(jpeg, a.url, Path(a.foto).name)
    r.pop("keypoints", None)
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
