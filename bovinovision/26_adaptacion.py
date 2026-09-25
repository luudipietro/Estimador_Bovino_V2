"""Fase 4b: ¿CUÁNTOS ANIMALES HAY QUE PESAR EN BALANZA para que el sistema funcione en un establecimiento nuevo?

Es la pregunta que el proyecto necesita responder para planificar la recolección del dataset argentino.

Idea: no se reentrena la red (el equipo no va a hacer eso en cada campo). Se usa la CNN ya entrenada en Bangladesh
como EXTRACTOR de características del bovino, y se ajusta encima una regresión simple con N animales locales
pesados en balanza. Es la estrategia realista de puesta en marcha: "pesá N animales una vez y el sistema queda
calibrado para tu rodeo".

Dominio nuevo = Mendeley (72 bovinos de Mongolia Interior, 341-644 kg, otra raza, otra cámara, y un rango de peso
que AcmeAI casi no tiene: solo 13 imágenes por encima de 341 kg).

Se comparan, para cada N:
  - CNN sin adaptar (predicción directa)             -> lo medido en 24_cross_mendeley.py: se rompe
  - mediana de los N animales locales                -> baseline honesto de "solo pesé N animales"
  - Ridge sobre los embeddings de la CNN con N animales locales
Con 30 repeticiones por N (distintos animales sorteados) para no depender de la suerte del muestreo.

Uso:  uv run python 26_adaptacion.py [--corrida convnext_tiny_crop_384_mse]
"""
import argparse
import importlib

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV

import config
from common import metrics
from common.acmeai import KP_CANONICOS

LADO = 1024
NS = [5, 10, 15, 25, 40]
REPETICIONES = 30


def keypoints_mendeley():
    ruta = config.TABLAS / "keypoints_mendeley.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta).set_index("image_id")
    c24 = importlib.import_module("24_cross_mendeley")
    kp, _ = c24.keypoints_mendeley(config.RUNS / "pose" / "pose9" / "weights" / "best.pt")
    kp.reset_index().to_parquet(ruta, index=False)
    return kp


def cargar_cnn(corrida):
    import timm
    cfg = __import__("json").loads((config.RUNS / "cnn" / corrida / "historial.json").read_text())["config"]
    modelo = timm.create_model(cfg["backbone"], pretrained=False, num_classes=1,
                               in_chans=4 if cfg["encuadre"] == "crop_mascara" else 3)
    modelo.load_state_dict(torch.load(config.RUNS / "cnn" / corrida / "mejor.pt", map_location="cpu"))
    modelo.reset_classifier(0)  # deja el embedding, sin la cabeza de regresión
    return modelo.eval(), cfg


def embeddings(modelo, cfg, man, kp):
    from torchvision.transforms import v2
    tfm = v2.Compose([v2.Resize((cfg["tam"], cfg["tam"])), v2.ToDtype(torch.float32, scale=True),
                      v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    salida = []
    with torch.no_grad():
        for r in man.itertuples():
            im = Image.open(r.ruta_imagen).convert("RGB")
            f = LADO / max(im.size)
            im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
            if cfg["encuadre"] in ("crop", "crop_mascara"):
                k = kp.loc[r.image_id]
                xs = np.array([k[f"{n}_x"] for n in KP_CANONICOS]) * f
                ys = np.array([k[f"{n}_y"] for n in KP_CANONICOS]) * f
                mx, my = 0.10 * (xs.max() - xs.min()), 0.15 * (ys.max() - ys.min())
                caja = (max(0, int(xs.min() - mx)), max(0, int(ys.min() - my)),
                        min(im.width, int(xs.max() + mx)), min(im.height, int(ys.max() + my)))
                if caja[2] - caja[0] > 10 and caja[3] - caja[1] > 10:
                    im = im.crop(caja)
            x = tfm(v2.functional.pil_to_tensor(im)).unsqueeze(0)
            salida.append(modelo(x).squeeze(0).numpy())
    return np.array(salida)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrida", default="convnext_tiny_crop_384_mse")
    a = ap.parse_args()

    man = pd.read_parquet(config.TABLAS / "manifest_mendeley.parquet")
    kp = keypoints_mendeley()
    man = man[man.image_id.isin(kp.index)].reset_index(drop=True)
    modelo, cfg = cargar_cnn(a.corrida)
    print(f"CNN: {a.corrida} ({cfg['backbone']}, {cfg['encuadre']}, {cfg['tam']}px)")
    E = embeddings(modelo, cfg, man, kp)
    y = man.peso_kg.to_numpy()
    print(f"Mendeley: {len(y)} animales, {y.min():.0f}-{y.max():.0f} kg | embedding de {E.shape[1]} dimensiones")

    filas = []
    rng = np.random.default_rng(config.SEED)
    for n in NS:
        r_mediana, r_ridge = [], []
        for _ in range(REPETICIONES):
            idx = rng.permutation(len(y))
            cal, resto = idx[:n], idx[n:]
            r_mediana.append(metrics.mape(y[resto], np.full(len(resto), np.median(y[cal]))))
            m = RidgeCV(alphas=np.logspace(0, 5, 30)).fit(E[cal], np.log(y[cal]))
            r_ridge.append(metrics.mape(y[resto], np.exp(m.predict(E[resto]))))
        filas.append({"n_animales_pesados": n, "mediana_local": np.mean(r_mediana),
                      "cnn_adaptada": np.mean(r_ridge), "cnn_adaptada_sd": np.std(r_ridge)})
        print(f"  N={n:3d} | mediana local {np.mean(r_mediana):5.2f}% | "
              f"CNN adaptada {np.mean(r_ridge):5.2f}% (±{np.std(r_ridge):.2f})")

    res = pd.DataFrame(filas)
    res.to_csv(config.RUNS / "adaptacion_mendeley.csv", index=False)
    print(f"\nreferencia: CNN de Bangladesh SIN adaptar sobre Mendeley = ver 24_cross_mendeley.py (regresión: 61%)")
    print(f"referencia: mismo sistema en su propio dominio (AcmeAI test) = 15,3%")
    print(f"-> {config.RUNS / 'adaptacion_mendeley.csv'}")


if __name__ == "__main__":
    main()
