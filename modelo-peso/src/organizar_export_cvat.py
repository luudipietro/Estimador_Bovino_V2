"""Toma el .zip de labels exportado de CVAT (formato YOLO Pose 1.0, SIN
imagenes) y lo deja en data/pose/ con la estructura que espera entrenar_pose.py.

Las imagenes no hace falta pedirselas a CVAT (bajarlas con imagenes esta
pago en el plan gratuito) porque ya las tenemos en etiquetado/imagenes/ --
son las mismas que se subieron para etiquetar. El script las empareja con
los .txt del zip por nombre de archivo.

Uso:  python src/organizar_export_cvat.py --zip C:/ruta/al/export.zip
"""
import argparse
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
import config

DESTINO = config.DATOS / "pose"
IMAGENES = AQUI.parent / "etiquetado" / "imagenes"
EXTS_IMG = {".jpg", ".jpeg", ".png"}


def encontrar_pares(carpeta_labels: Path):
    """Empareja cada .txt del export con su imagen en etiquetado/imagenes/."""
    imgs = {p.stem: p for p in IMAGENES.glob("*") if p.suffix.lower() in EXTS_IMG}
    labels = {p.stem: p for p in carpeta_labels.rglob("*.txt")
              if p.name not in ("obj.names", "train.txt", "val.txt")}

    stems = sorted(set(imgs) & set(labels))
    sin_label = sorted(set(imgs) - set(labels))
    sin_imagen = sorted(set(labels) - set(imgs))

    if sin_label:
        print(f"  Aviso: {len(sin_label)} imagen(es) todavia sin etiquetar: {sin_label[:5]}")
    if sin_imagen:
        print(f"  Aviso: {len(sin_imagen)} .txt sin imagen en etiquetado/imagenes/: {sin_imagen[:5]}")

    return [(imgs[s], labels[s]) for s in stems]


def main(args):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(tmp)

        pares = encontrar_pares(tmp)
        if not pares:
            print("No se encontro ningun par imagen+label dentro del zip.")
            return

        random.Random(args.seed).shuffle(pares)
        n_val = max(1, round(len(pares) * args.val))
        splits = {"val": pares[:n_val], "train": pares[n_val:]}

        for split, items in splits.items():
            (DESTINO / "images" / split).mkdir(parents=True, exist_ok=True)
            (DESTINO / "labels" / split).mkdir(parents=True, exist_ok=True)
            for img, lbl in items:
                shutil.copy2(img, DESTINO / "images" / split / img.name)
                shutil.copy2(lbl, DESTINO / "labels" / split / lbl.name)
            print(f"  {split}: {len(items)} imagenes")

    print(f"\n{len(pares)} pares copiados a {DESTINO}")
    print("Listo para: python src/entrenar_pose.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="ruta al .zip exportado de CVAT")
    ap.add_argument("--val", type=float, default=0.2, help="fraccion para validacion")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
