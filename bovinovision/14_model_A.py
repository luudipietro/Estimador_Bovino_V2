"""Etapa 11: Modelo A = "una CNN mira la foto y adivina el peso", sin keypoints ni sticker (sec. 14 del plan).

Backbone preentrenado en ImageNet, CONGELADO (no se reentrena: en CPU no da el tiempo y con ~3.200 imágenes de
train se sobreajustaría fácil) -> embedding de 512 features -> Ridge/HGB sobre log(peso). Sirve como referencia
de cuánto se puede estimar sin ninguna geometría explícita; se compara contra B (keypoints, sin escala) y C
(keypoints + sticker + forma). La combinación de A con C es la etapa 12 (fusión).

Los embeddings se cachean en data/cache/embeddings_<backbone>.parquet (tardan ~1 vez; 4.540 imágenes en CPU).
Uso:  uv run python 14_model_A.py [--backbone resnet18] [--recalcular]
"""
import argparse
import importlib

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from common import metrics

bl = importlib.import_module("09_baselines")


def extraer_embeddings(backbone: str, cache_path):
    import timm
    from torchvision import transforms

    man = pd.read_parquet(config.TABLAS / "manifest.parquet")
    modelo = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg").eval()
    cfg = timm.data.resolve_data_config({}, model=modelo)
    tfm = transforms.Compose([
        transforms.Resize((cfg["input_size"][1], cfg["input_size"][2])),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    filas, lote_x, lote_ids = [], [], []
    with torch.no_grad():
        for i, r in enumerate(man.itertuples(), 1):
            im = Image.open(config.ACMEAI / r.ruta_imagen).convert("RGB")
            lote_x.append(tfm(im))
            lote_ids.append(r.image_id)
            if len(lote_x) == 32 or i == len(man):
                emb = modelo(torch.stack(lote_x)).numpy()
                for image_id, e in zip(lote_ids, emb):
                    filas.append({"image_id": image_id, **{f"e{j}": float(v) for j, v in enumerate(e)}})
                lote_x, lote_ids = [], []
            if i % 500 == 0:
                print(f"  {i}/{len(man)}")
    df = pd.DataFrame(filas)
    df.to_parquet(cache_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--recalcular", action="store_true")
    a = ap.parse_args()

    cache = config.CACHE / f"embeddings_{a.backbone}.parquet"
    if cache.exists() and not a.recalcular:
        emb = pd.read_parquet(cache)
        print(f"embeddings desde caché: {cache}")
    else:
        emb = extraer_embeddings(a.backbone, cache)
        print(f"embeddings calculados -> {cache}")

    man = pd.read_parquet(config.TABLAS / "manifest.parquet")[["image_id", "split", "peso_kg"]]
    df = emb.merge(man, on="image_id")
    c = [c for c in df.columns if c.startswith("e")]
    print(f"{len(df)} imágenes, {len(c)} dimensiones de embedding")

    tr = df[df.split == "train"]
    filas = []
    for split in ("val", "test"):
        te = df[df.split == split]
        for nom_m in ("ridge", "hgb"):
            p = bl.ajustar_predecir(nom_m, tr[c], tr.peso_kg, te[c])
            filas.append({"modelo": nom_m, "split": split, **metrics.reporte(tr.peso_kg, te.peso_kg, p, nom_m)})
    res = pd.DataFrame(filas)[["split", "modelo", "n", "mape", "mape_mediana", "mejora_pct"]]
    res.to_csv(config.RUNS / "model_a.csv", index=False)
    print(res.round(2).to_string(index=False))

    modelo_final = bl.modelo("ridge").fit(tr[c], np.log(tr.peso_kg))  # ridge: más estable con 512 features correlacionadas
    (config.RUNS / "models").mkdir(exist_ok=True)
    joblib.dump({"modelo": modelo_final, "columnas": c, "backbone": a.backbone}, config.RUNS / "models" / "model_a.joblib")

    te = df[df.split == "test"]
    p = np.exp(modelo_final.predict(te[c]))
    print(f"\nMAPE por banda (test, Modelo A, ridge):")
    print(metrics.mape_por_banda(te.peso_kg, p).round(2).to_string(index=False))
    df.assign(pred_modelo_a=np.exp(modelo_final.predict(df[c])))[["image_id", "split", "peso_kg", "pred_modelo_a"]].to_parquet(
        config.TABLAS / "pred_modelo_a.parquet", index=False)


if __name__ == "__main__":
    main()
