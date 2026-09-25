"""Verifica que las fotos del campo traigan la metadata que necesitamos, ANTES de sacar cientos.

Uso rápido (en el campo, con 2 o 3 fotos de prueba):
    python verificar_fotos.py C:\\ruta\\a\\las\\fotos

Qué revisa por foto: resolución, marca/modelo, fecha, focal equivalente 35 mm, y —si es iPhone y está
instalado exiftool— el vector de aceleración, que es la inclinación del teléfono al momento del disparo.

El error más común es transferir las fotos de una forma que borra la metadata (WhatsApp, "optimizar para
web", convertir a PNG). En el dataset de Mendeley que analizamos, las fotos eran de un iPhone 13 y llegaron
en PNG: perdieron absolutamente todo el EXIF. Este script detecta eso enseguida.

No necesita instalar nada: usa Pillow, que ya viene con el proyecto. Para HEIC hace falta `pillow-heif`;
para la inclinación, el programa `exiftool` (opcional, https://exiftool.org).
"""
import subprocess
import sys
from pathlib import Path

from PIL import ExifTags, Image

EXIF_FOCAL_35MM = 0xA405
EXTENSIONES = {".jpg", ".jpeg", ".heic", ".png", ".tif", ".tiff"}


def leer_exiftool(ruta):
    """Devuelve {tag: valor} con los campos de Apple, si exiftool está instalado. Si no, {}."""
    try:
        s = subprocess.run(["exiftool", "-s", "-AccelerationVector", "-Make", "-Model",
                            "-FocalLengthIn35mmFormat", "-GPSPosition", str(ruta)],
                           capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    out = {}
    for linea in s.stdout.splitlines():
        if ":" in linea:
            k, v = linea.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def revisar(ruta):
    info = {"archivo": ruta.name, "formato": ruta.suffix.lower()}
    try:
        with Image.open(ruta) as im:
            info["resolucion"] = f"{im.width}x{im.height}"
            info["megapixeles"] = round(im.width * im.height / 1e6, 1)
            ex = im.getexif()
            tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()} if ex else {}
            info["marca"] = str(tags.get("Make", "")).strip()
            info["modelo"] = str(tags.get("Model", "")).strip()
            info["fecha"] = str(tags.get("DateTime", "")).strip()
            try:
                info["focal_35mm"] = ex.get_ifd(0x8769).get(EXIF_FOCAL_35MM)
            except Exception:
                info["focal_35mm"] = None
    except Exception as e:
        info["error"] = f"no se pudo abrir ({e.__class__.__name__}). Si es HEIC: pip install pillow-heif"
        return info

    extra = leer_exiftool(ruta)
    info["inclinacion"] = extra.get("AccelerationVector")
    if not info["marca"]:
        info["marca"] = extra.get("Make", "")
    return info


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    carpeta = Path(sys.argv[1])
    fotos = sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not fotos:
        print(f"No encontré fotos en {carpeta}")
        return

    print(f"Revisando {len(fotos)} fotos en {carpeta}\n")
    sin_exif, sin_fecha, chicas, con_inclinacion = [], [], [], 0
    for p in fotos[:200]:
        i = revisar(p)
        if "error" in i:
            print(f"  {i['archivo']}: {i['error']}")
            sin_exif.append(p.name)
            continue
        if i.get("inclinacion"):
            con_inclinacion += 1
        if not i["marca"]:
            sin_exif.append(i["archivo"])
        if not i["fecha"]:
            sin_fecha.append(i["archivo"])
        if i["megapixeles"] < 3:
            chicas.append(f"{i['archivo']} ({i['resolucion']})")
        if len(fotos) <= 10:  # con pocas fotos, mostrar el detalle de cada una
            print(f"  {i['archivo']:28s} {i['resolucion']:>11s} | {i['marca']} {i['modelo']} | "
                  f"{i['fecha']} | focal35={i['focal_35mm']} | inclinación={i['inclinacion'] or '-'}")

    n = min(len(fotos), 200)
    print(f"\n{'=' * 60}\nRESULTADO sobre {n} fotos:")
    ok = True
    if sin_exif:
        ok = False
        print(f"  [PROBLEMA] {len(sin_exif)} sin metadata de cámara: la transferencia la borró.")
        print(f"             No usar WhatsApp ni 'optimizar'. Pasar por cable, AirDrop (Datos originales)")
        print(f"             o iCloud descargando el original. Ejemplos: {sin_exif[:3]}")
    if sin_fecha:
        ok = False
        print(f"  [PROBLEMA] {len(sin_fecha)} sin fecha/hora. Hace falta para cruzar foto y pesaje.")
    if chicas:
        ok = False
        print(f"  [PROBLEMA] {len(chicas)} de baja resolución (menos de 3 MP): {chicas[:3]}")
    if con_inclinacion:
        print(f"  [OK] {con_inclinacion}/{n} traen la inclinación del teléfono (iPhone + exiftool).")
    elif not leer_exiftool(fotos[0]):
        print(f"  [aviso] exiftool no está instalado, así que no puedo revisar la inclinación.")
        print(f"          No es bloqueante: si las fotos son de iPhone y conservan el EXIF, el dato está ahí.")
    if ok:
        print("  [OK] Las fotos conservan la metadata necesaria. Pueden seguir así.")
    print("\nRecordar además, por sesión: altura a la que se toma la foto, identificación de cada animal")
    print("y el peso de balanza del mismo día.")


if __name__ == "__main__":
    main()
