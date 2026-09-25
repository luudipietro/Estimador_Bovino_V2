"""Etapa 3: keypoints de referencia (anotados a mano) -> data/tablas/keypoints_gt.parquet + overlays de control.

Una fila por imagen, columnas <kp>_x, <kp>_y, <kp>_v en PÍXELES DE LA IMAGEN ENTREGADA (no del original):
en B3 el JSON guarda las coordenadas en la resolución original de la foto, por eso se reescala con
ancho_real / ancho_json (en B4 el factor es 1).
Uso:  uv run python 04_keypoints_gt.py
"""
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config
from common import acmeai


def main():
    manifest = pd.read_parquet(config.TABLAS / "manifest.parquet").set_index("image_id")
    filas, sin_anot = [], 0
    for lote, rutas in acmeai.LOTES_SIDE.items():
        coco = json.load(open(config.ACMEAI / rutas["coco"], encoding="utf-8"))
        imagenes = {i["id"]: i for i in coco["images"]}
        nombres = coco["categories"][0]["keypoints"]
        vistos = set()
        for a in coco["annotations"]:
            im = imagenes[a["image_id"]]
            image_id = f"{lote}/{im['file_name']}"
            if image_id not in manifest.index or image_id in vistos or a.get("num_keypoints", 0) != 9:
                continue
            vistos.add(image_id)
            r = manifest.loc[image_id]
            fx, fy = r.ancho / im["width"], r.alto / im["height"]
            fila = {"image_id": image_id, "factor_escala_x": fx, "factor_escala_y": fy}
            for kp, (x, y, v) in acmeai.kp_por_nombre(a["keypoints"], nombres).items():
                fila.update({f"{kp}_x": x * fx, f"{kp}_y": y * fy, f"{kp}_v": v})
            filas.append(fila)
        sin_anot += (manifest["lote"] == lote).sum() - len(vistos)

    df = pd.DataFrame(filas)
    df.to_parquet(config.TABLAS / "keypoints_gt.parquet", index=False)
    print(f"{len(df)} imágenes con 9 keypoints; sin anotación: {sin_anot}")
    print("factor de escala por lote:\n", df.assign(lote=df.image_id.str[:2]).groupby("lote")[
        ["factor_escala_x", "factor_escala_y"]].agg(["min", "median", "max"]).round(3))

    # Chequeo de consistencia: tras reescalar, ningún punto debe caer fuera de la imagen
    m = manifest.loc[df.image_id]
    xs = df[[f"{k}_x" for k in acmeai.KP_CANONICOS]].to_numpy()
    ys = df[[f"{k}_y" for k in acmeai.KP_CANONICOS]].to_numpy()
    fuera = ((xs > m.ancho.to_numpy()[:, None] + 1) | (ys > m.alto.to_numpy()[:, None] + 1) | (xs < -1) | (ys < -1)).any(1)
    print(f"imágenes con algún keypoint fuera de la imagen tras reescalar: {fuera.sum()}")

    # Overlays de control visual (12 al azar, 6 por lote)
    out = config.RUNS / "overlays_kp_gt"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    for lote in acmeai.LOTES_SIDE:
        sub = df[df.image_id.str.startswith(lote)]
        for _, f in sub.iloc[rng.choice(len(sub), 6, replace=False)].iterrows():
            im = Image.open(config.ACMEAI / manifest.loc[f.image_id].ruta_imagen).convert("RGB")
            d = ImageDraw.Draw(im)
            for k in acmeai.KP_CANONICOS:
                x, y = f[f"{k}_x"], f[f"{k}_y"]
                d.ellipse([x - 8, y - 8, x + 8, y + 8], outline="red", width=3)
                d.text((x + 10, y - 6), k, fill="yellow")
            im.save(out / f"{f.image_id.replace('/', '_')}.png")
    print(f"overlays en {out}")


if __name__ == "__main__":
    main()
