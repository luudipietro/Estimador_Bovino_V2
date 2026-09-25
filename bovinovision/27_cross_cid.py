"""Fase 4c: separar el efecto RANGO DE PESO del efecto DOMINIO VISUAL.

Con Mendeley no se podía distinguir: su rango (341-644 kg) casi no se solapa con el de entrenamiento, así que
fallar ahí podía deberse a extrapolación, a cambio de dominio, o a ambos.

CID sí permite separarlo: va de 150 a 816 kg y **se solapa con AcmeAI entre 150 y 300 kg**, donde AcmeAI tiene
miles de imágenes. Entonces:
  - si la CNN anda bien en CID DENTRO del rango común  -> el problema es el rango (no extrapola)
  - si falla también dentro del rango común            -> el problema es el dominio visual

Aviso sobre el dominio de CID: son fotos de una estación fija (plataforma, fondo constante, vara graduada
amarilla al lado del animal) y de menor resolución (1200x675 / 800x450) que AcmeAI. O sea, el cambio de dominio
es grande; hay que tenerlo en cuenta al interpretar.

Vista lateral: cada animal tiene 4 fotos (lateral, frente, trasera, lateral). Se elige automáticamente la más
"alargada" según los keypoints predichos (la lateral es la de mayor relación ancho/alto).

Uso:  uv run python 27_cross_cid.py [--corrida convnext_tiny_crop_384_mse]
"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV

import config
from common import metrics
from common.acmeai import KP_CANONICOS

CID = config.ROOT.parent / "datasets_externos"
LADO = 1024
RANGO_COMUN = (150, 300)  # kg: donde CID y AcmeAI tienen datos los dos


def manifest_cid() -> pd.DataFrame:
    meta = pd.read_csv(CID / "cid_dataset.csv")
    filas = []
    for r in meta.itertuples():
        carpeta = CID / "images" / str(r.sku)
        if not carpeta.exists():
            continue
        for p in sorted(carpeta.glob("*.jpg")):
            filas.append({"image_id": f"CID/{r.sku}/{p.name}", "animal_id": str(r.sku), "ruta_imagen": str(p),
                          "peso_kg": float(r.weight_in_kg), "raza": r.breed, "sexo": r.sex,
                          "edad": r.age_in_year, "altura_in": r.height_in_inch})
    return pd.DataFrame(filas)


def keypoints_cid(man, imgsz=640):
    ruta = config.TABLAS / "keypoints_cid.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta).set_index("image_id")
    from ultralytics import YOLO
    modelo = YOLO(config.RUNS / "pose" / "pose9" / "weights" / "best.pt")
    filas = []
    for i, r in enumerate(man.itertuples(), 1):
        res = modelo.predict(source=r.ruta_imagen, imgsz=imgsz, conf=0.05, max_det=3, verbose=False)[0]
        fila = {"image_id": r.image_id, "n_det": len(res.boxes)}
        if len(res.boxes):
            k = int(res.boxes.conf.argmax())
            xy = res.keypoints.xy[k].cpu().numpy()
            fila["det_conf"] = float(res.boxes.conf[k])
            for j, n in enumerate(KP_CANONICOS):
                fila[f"{n}_x"], fila[f"{n}_y"] = float(xy[j, 0]), float(xy[j, 1])
        filas.append(fila)
        if i % 250 == 0:
            print(f"  pose {i}/{len(man)}")
    df = pd.DataFrame(filas).set_index("image_id")
    df.reset_index().to_parquet(ruta, index=False)
    return df


def elegir_lateral(man, kp):
    """Una foto por animal: la de mayor relación ancho/alto de los keypoints (la vista lateral)."""
    d = man.set_index("image_id").join(kp, how="inner").dropna(subset=["det_conf"])
    xs = d[[f"{n}_x" for n in KP_CANONICOS]].to_numpy()
    ys = d[[f"{n}_y" for n in KP_CANONICOS]].to_numpy()
    d["aspecto"] = (xs.max(1) - xs.min(1)) / np.maximum(ys.max(1) - ys.min(1), 1)
    return d.loc[d.groupby("animal_id").aspecto.idxmax()]


def predecir(corrida, d):
    """Predicción directa de la CNN entrenada en AcmeAI, y también su embedding (para adaptación)."""
    import json
    import timm
    from torchvision.transforms import v2
    cfg = json.loads((config.RUNS / "cnn" / corrida / "historial.json").read_text())["config"]
    modelo = timm.create_model(cfg["backbone"], pretrained=False, num_classes=1)
    modelo.load_state_dict(torch.load(config.RUNS / "cnn" / corrida / "mejor.pt", map_location="cpu"))
    modelo.eval()
    tfm = v2.Compose([v2.Resize((cfg["tam"], cfg["tam"])), v2.ToDtype(torch.float32, scale=True),
                      v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    preds, embs = [], []
    with torch.no_grad():
        for i, r in enumerate(d.itertuples(), 1):
            im = Image.open(r.ruta_imagen).convert("RGB")
            f = LADO / max(im.size)
            if f < 1:
                im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
            else:
                f = 1.0
            xs = np.array([getattr(r, f"{n}_x") for n in KP_CANONICOS]) * f
            ys = np.array([getattr(r, f"{n}_y") for n in KP_CANONICOS]) * f
            mx, my = 0.10 * (xs.max() - xs.min()), 0.15 * (ys.max() - ys.min())
            caja = (max(0, int(xs.min() - mx)), max(0, int(ys.min() - my)),
                    min(im.width, int(xs.max() + mx)), min(im.height, int(ys.max() + my)))
            if caja[2] - caja[0] > 10 and caja[3] - caja[1] > 10:
                im = im.crop(caja)
            x = tfm(v2.functional.pil_to_tensor(im)).unsqueeze(0)
            e = modelo.forward_features(x)
            e = modelo.forward_head(e, pre_logits=True)
            preds.append(float(modelo.get_classifier()(e).squeeze()))
            embs.append(e.squeeze(0).numpy())
            if i % 100 == 0:
                print(f"  cnn {i}/{len(d)}")
    return np.exp(np.array(preds)), np.array(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrida", default="convnext_tiny_crop_384_mse")
    a = ap.parse_args()

    man = manifest_cid()
    print(f"CID: {len(man)} imágenes de {man.animal_id.nunique()} animales, "
          f"{man.peso_kg.min():.0f}-{man.peso_kg.max():.0f} kg")
    kp = keypoints_cid(man)
    print(f"detección: {kp.det_conf.notna().mean():.1%} de las imágenes")
    d = elegir_lateral(man, kp)
    print(f"laterales elegidas: {len(d)} (una por animal)")

    pred, E = predecir(a.corrida, d)
    y = d.peso_kg.to_numpy()
    acme = pd.read_parquet(config.TABLAS / "manifest.parquet")
    y_acme_tr = acme[acme.split == "train"].peso_kg

    print(f"\n=== CNN entrenada en AcmeAI, aplicada a CID (sin adaptar) ===")
    print(f"  predice {pred.min():.0f}-{pred.max():.0f} kg | real {y.min():.0f}-{y.max():.0f} kg")
    filas = []
    for nombre, m in (("CID completo", np.ones(len(y), bool)),
                      (f"CID en el rango común {RANGO_COMUN[0]}-{RANGO_COMUN[1]} kg",
                       (y >= RANGO_COMUN[0]) & (y <= RANGO_COMUN[1])),
                      ("CID fuera del rango común", (y < RANGO_COMUN[0]) | (y > RANGO_COMUN[1]))):
        if m.sum() < 5:
            continue
        filas.append({"subconjunto": nombre, "n": int(m.sum()),
                      "mape_cnn": metrics.mape(y[m], pred[m]),
                      "mape_mediana_acme": metrics.mape(y[m], np.full(m.sum(), np.median(y_acme_tr))),
                      "mape_mediana_local": metrics.mape(y[m], np.full(m.sum(), np.median(y[m]))),
                      "spearman": pd.Series(pred[m]).corr(pd.Series(y[m]), method="spearman")})
    res = pd.DataFrame(filas)
    print(res.round(2).to_string(index=False))
    res.to_csv(config.RUNS / "cross_cid.csv", index=False)

    print(f"\n=== ¿el embedding discrimina peso dentro de CID? (adaptación con N animales locales) ===")
    rng = np.random.default_rng(config.SEED)
    for n in (10, 25, 50, 100, 200):
        r_med, r_rid = [], []
        for _ in range(20):
            idx = rng.permutation(len(y))
            cal, resto = idx[:n], idx[n:]
            r_med.append(metrics.mape(y[resto], np.full(len(resto), np.median(y[cal]))))
            mo = RidgeCV(alphas=np.logspace(0, 5, 30)).fit(E[cal], np.log(y[cal]))
            r_rid.append(metrics.mape(y[resto], np.exp(mo.predict(E[resto]))))
        print(f"  N={n:3d} | mediana local {np.mean(r_med):5.2f}% | CNN adaptada {np.mean(r_rid):5.2f}%")

    d.assign(pred_cnn=pred).reset_index()[["image_id", "animal_id", "peso_kg", "raza", "sexo", "edad",
                                           "altura_in", "pred_cnn"]].to_parquet(
        config.TABLAS / "pred_cid.parquet", index=False)
    np.save(config.CACHE / "emb_cid.npy", E)
    print(f"-> {config.TABLAS / 'pred_cid.parquet'}")


if __name__ == "__main__":
    main()
