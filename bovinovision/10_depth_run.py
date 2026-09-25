"""Etapa 8: profundidad monocular métrica sobre un subconjunto de imágenes -> tabla + caché de mapas.

Por imagen guarda: profundidad (m) mediana sobre el sticker, sobre la vaca y en cada uno de los 9 keypoints
(parche 5x5), y la focal en px que estima Depth Pro (Depth Anything V2 no la estima: NaN).
Los mapas completos se guardan reducidos 4x en data/cache/depth/<modelo>/ (float16) para no recalcular.
Con esta tabla, 11_depth_scale_check.py mide qué tan bien la profundidad recupera la escala real.

Modelos:  pro (apple/DepthPro-hf, no comercial) | dav2s|dav2b|dav2l (Metric-Outdoor) | dav2s-in|dav2l-in (Metric-Indoor)
Uso (Fedora):  HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run python 10_depth_run.py --modelo pro --n 300
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import config
from common import masks
from common.acmeai import KP_CANONICOS

MODELOS = {
    "pro": "apple/DepthPro-hf",
    "dav2s": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "dav2b": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "dav2l": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    # variantes Indoor (entrenadas en Hypersim): probar también, el rango de distancias es más corto y quizá más afín
    "dav2s-in": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "dav2l-in": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


def cargar(nombre, device, fp16):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dtype = torch.float16 if fp16 else torch.float32
    if nombre == "pro":
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
        proc = DepthProImageProcessorFast.from_pretrained(MODELOS[nombre])
        modelo = DepthProForDepthEstimation.from_pretrained(MODELOS[nombre], dtype=dtype)
    else:
        proc = AutoImageProcessor.from_pretrained(MODELOS[nombre])
        modelo = AutoModelForDepthEstimation.from_pretrained(MODELOS[nombre], dtype=dtype)
    return proc, modelo.to(device).eval(), dtype


def inferir(nombre, proc, modelo, dtype, device, im):
    """Devuelve (mapa de profundidad en metros con el tamaño de la imagen, focal_px o NaN)."""
    import torch
    import torch.nn.functional as F
    inp = proc(images=im, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    with torch.no_grad():
        out = modelo(**inp)
    alto, ancho = im.height, im.width
    if nombre == "pro":
        r = proc.post_process_depth_estimation(out, target_sizes=[(alto, ancho)])[0]
        return r["predicted_depth"].float().cpu().numpy(), float(r["focal_length"])
    d = F.interpolate(out.predicted_depth.float().unsqueeze(1), size=(alto, ancho), mode="bicubic",
                      align_corners=False)[0, 0]
    return d.cpu().numpy(), float("nan")


def parche(mapa, x, y, r=2):
    h, w = mapa.shape
    xi, yi = int(round(min(max(x, 0), w - 1))), int(round(min(max(y, 0), h - 1)))
    return float(np.median(mapa[max(yi - r, 0): yi + r + 1, max(xi - r, 0): xi + r + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=list(MODELOS), default="dav2s")
    ap.add_argument("--n", type=int, default=300, help="imágenes a procesar (0 = todas)")
    ap.add_argument("--split", default="test", help="split del que se toma el subconjunto (val/test/train/all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()

    import torch
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"modelo={a.modelo} device={device} torch={torch.__version__}")

    man = pd.read_parquet(config.TABLAS / "manifest.parquet")
    kp = pd.read_parquet(config.TABLAS / "keypoints_gt.parquet").set_index("image_id")
    sub = man[man.image_id.isin(kp.index)]
    if a.split != "all":
        sub = sub[sub.split == a.split]
    if a.n:  # muestreo estratificado por banda de peso para cubrir terneros y adultos
        sub = (sub.assign(b=pd.cut(sub.peso_kg, config.BANDAS_PESO, right=False))
               .groupby("b", observed=True, group_keys=False)
               .apply(lambda g: g.sample(min(len(g), max(1, round(a.n * len(g) / len(sub)))), random_state=config.SEED)))

    proc, modelo, dtype = cargar(a.modelo, device, a.fp16)
    cache = config.CACHE / "depth" / a.modelo
    cache.mkdir(parents=True, exist_ok=True)

    filas = []
    for i, r in enumerate(sub.itertuples(), 1):
        im = Image.open(config.ACMEAI / r.ruta_imagen).convert("RGB")
        mapa, focal = inferir(a.modelo, proc, modelo, dtype, device, im)
        rgba_mascara = np.array(Image.open(config.ACMEAI / r.ruta_mascara).convert("RGBA"))
        if rgba_mascara.shape[:2] != (im.height, im.width):
            # ~1% de las máscaras (sobre todo B4) vienen rotadas 90° respecto a la foto/keypoints;
            # confirmado empíricamente (keypoints GT caen dentro de la vaca tras rotar con k=1, no con k=3
            # ni con transpose). Si tampoco calza tras rotar, se descarta la imagen (mejor que indexar mal).
            rgba_mascara = np.rot90(rgba_mascara, k=1, axes=(0, 1))
            if rgba_mascara.shape[:2] != (im.height, im.width):
                print(f"  aviso: máscara de {r.image_id} no calza ni rotada, se omite")
                continue
        vaca, sticker = masks.separar(rgba_mascara)
        fila = {"image_id": r.image_id, "focal_px_est": focal,
                "z_sticker_m": float(np.median(mapa[sticker])) if sticker.any() else np.nan,
                "z_vaca_m": float(np.median(mapa[vaca])) if vaca.any() else np.nan}
        for k in KP_CANONICOS:
            fila[f"z_{k}_m"] = parche(mapa, kp.loc[r.image_id, f"{k}_x"], kp.loc[r.image_id, f"{k}_y"])
        filas.append(fila)
        np.save(cache / (r.image_id.replace("/", "__") + ".npy"), mapa[::4, ::4].astype(np.float16))
        if i % 25 == 0 or i == len(sub):
            print(f"  {i}/{len(sub)}")
            pd.DataFrame(filas).to_parquet(config.TABLAS / f"depth_{a.modelo}.parquet", index=False)

    d = pd.DataFrame(filas)
    print(f"-> {config.TABLAS / f'depth_{a.modelo}.parquet'}")
    print(d[["focal_px_est", "z_sticker_m", "z_vaca_m"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
