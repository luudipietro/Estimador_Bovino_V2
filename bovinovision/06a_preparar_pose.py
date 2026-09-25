"""Etapa 5 (prep): arma el dataset YOLO-pose en data/pose/{images,labels}/{train,val,test}.

- Imágenes reducidas a 1024 px de lado mayor (los keypoints se guardan normalizados, no cambian).
- Un objeto por imagen (clase 0 'bovino'), bbox = unión de la máscara de la vaca y de los 9 keypoints.
- Los 9 keypoints son visibles (v=2); ninguno tiene par izquierda/derecha (flip_idx identidad, ver 06_train_pose.py).
Se ejecuta en Windows (usa manifest, keypoints_gt y mascaras) y luego se copia data/ a Fedora.
Uso:  uv run python 06a_preparar_pose.py
"""
import shutil

import numpy as np
import pandas as pd
from PIL import Image

import config
from common.acmeai import KP_CANONICOS

LADO_MAX = 1024
DIR = config.DATA / "pose"


def nombre_seguro(image_id: str) -> str:
    return image_id.replace("/", "__")  # 'B3/100_s_109_M.jpg' -> 'B3__100_s_109_M.jpg'


def linea_label(r, ancho, alto) -> str:
    xs = np.array([r[f"{k}_x"] for k in KP_CANONICOS], float)
    ys = np.array([r[f"{k}_y"] for k in KP_CANONICOS], float)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if not np.isnan(r.get("vaca_x0", np.nan)):
        x0, x1 = min(x0, r["vaca_x0"]), max(x1, r["vaca_x1"])
        y0, y1 = min(y0, r["vaca_y0"]), max(y1, r["vaca_y1"])
    x0, x1, y0, y1 = np.clip([x0, x1], 0, ancho).tolist() + np.clip([y0, y1], 0, alto).tolist()
    cx, cy, w, h = (x0 + x1) / 2 / ancho, (y0 + y1) / 2 / alto, (x1 - x0) / ancho, (y1 - y0) / alto
    kps = " ".join(f"{np.clip(x / ancho, 0, 1):.6f} {np.clip(y / alto, 0, 1):.6f} 2" for x, y in zip(xs, ys))
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kps}\n"


def main():
    man = pd.read_parquet(config.TABLAS / "manifest.parquet").set_index("image_id")
    kp = pd.read_parquet(config.TABLAS / "keypoints_gt.parquet").set_index("image_id")
    msk = pd.read_parquet(config.TABLAS / "mascaras.parquet").set_index("image_id")[["vaca_x0", "vaca_x1", "vaca_y0", "vaca_y1"]]
    df = kp.join(msk).join(man[["split", "ancho", "alto", "ruta_imagen"]])

    if DIR.exists():
        shutil.rmtree(DIR)
    for s in ("train", "val", "test"):
        (DIR / "images" / s).mkdir(parents=True)
        (DIR / "labels" / s).mkdir(parents=True)

    for i, (image_id, r) in enumerate(df.iterrows(), 1):
        nombre = nombre_seguro(image_id)
        im = Image.open(config.ACMEAI / r.ruta_imagen).convert("RGB")
        escala = LADO_MAX / max(im.size)
        if escala < 1:
            im = im.resize((round(im.width * escala), round(im.height * escala)), Image.LANCZOS)
        im.save(DIR / "images" / r.split / nombre, quality=93)
        (DIR / "labels" / r.split / (nombre[:-4] + ".txt")).write_text(linea_label(r, r.ancho, r.alto))
        if i % 1000 == 0:
            print(f"  {i}/{len(df)}")
    print(df.groupby("split").size().to_dict(), f"-> {DIR}")


if __name__ == "__main__":
    main()
