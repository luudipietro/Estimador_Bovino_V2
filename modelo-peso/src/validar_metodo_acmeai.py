"""Valida el metodo (landmarks + calibracion con marcador de referencia) sobre el
dataset publico de Acme AI / BMGF -- mucho mas grande que el propio (4.728 fotos
reales de campo en Bangladesh, peso de bascula, keypoints anotados por
profesionales, y una calcomania de referencia ya segmentada en las mascaras).

Esto NO es el pipeline de produccion propio (esos son entrenar_pose.py +
medidas_desde_keypoints.py / evaluar_holdout.py, sobre keypoints PREDICHOS por
nuestro modelo). Esto valida que el METODO funciona -- con keypoints de
referencia, no los nuestros -- antes de invertir en recolectar el dataset propio
con el marcador incluido.

Resultado (15/09/2026): sin calibrar, MAPE ~22% (apenas mejor que el baseline
tonto). Calibrando con el marcador y usando SOLO 4 distancias elegidas a mano
(altura, largo, girth_f, girth_r), ~14-17% segun banda de peso. Usando TODAS las
distancias posibles entre los 9 keypoints (36 pares, calibradas por el sticker) y
dejando que Boosting elija cuales importan -- en vez de elegir 4 a mano -- el
numero mejora a ~15,3% general y ~12,9% en la banda de 100-200 kg (la mayoria de
los datos), que ya cumple el objetivo del acta (<15%) ahi. Este es el resultado
que se reporta.

Se investigo si el error restante viene de imprecision de calibracion (tamano
aparente del sticker, o que no quede de frente a camara) y NO -- ninguna de las
dos correlaciona con el error. El techo restante parece estructural: una sola
foto lateral con un solo marcador no puede recuperar la circunferencia real del
pecho (solo un proxy de profundidad), ni corregir que cada landmark esta a una
profundidad distinta de la camara. Detalle completo en README.md.

Uso:  python src/validar_metodo_acmeai.py
"""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

STICKER_RGB = {"b2": (255, 240, 0), "b3": (0, 117, 255), "b4": (0, 117, 255)}

# B3 y B4 comparten los mismos 9 landmarks (mismos nombres, orden distinto en el
# json -- por eso se mapea por nombre, no por indice). B2 tiene un esquema mas
# chico (6 puntos, sin shoulderbone ni height) y se trata aparte.
ALIAS = {
    "front_girth_top": "front_top", "front_girth_bottom": "front_bottom",
    "rear_girth_top": "rear_top", "rear_girth_bottom": "rear_bottom",
}


def normalizar(nombre):
    n = nombre.lower().lstrip("0123456789_").replace("-", "_")
    return ALIAS.get(n, n)


def cargar_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    imgs = {im["id"]: im for im in d["images"]}
    nombres = [normalizar(n) for n in d["categories"][0]["keypoints"]]
    return d, imgs, nombres


def sticker_size_px(mask, sticker_rgb, ancho_real, alto_real):
    sticker = (mask == sticker_rgb).all(axis=-1)
    if sticker.sum() < 20:
        return None
    ys, xs = np.where(sticker)
    ancho_mask, alto_mask = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    mh, mw = mask.shape[:2]
    sx, sy = mw / ancho_real, mh / alto_real
    return ((ancho_mask / sx) * (alto_mask / sy)) ** 0.5


def procesar(nombre_batch, json_path, mask_dir, patron, extraer):
    """Devuelve un DataFrame con TODAS las distancias entre pares de keypoints
    (calibradas por el sticker), nombradas por nombre de landmark (no indice) para
    que sean comparables entre batches aunque el orden del json difiera."""
    raiz = config.DATASET_ACMEAI
    d, imgs, nombres = cargar_json(raiz / json_path)
    idx = {n: i for i, n in enumerate(nombres)}
    pares = list(itertools.combinations(sorted(idx), 2))
    mask_dir = raiz / mask_dir

    filas = []
    for ann in d["annotations"]:
        fn = imgs[ann["image_id"]]["file_name"]
        m = patron.match(fn)
        if not m or ann["num_keypoints"] < len(nombres):
            continue
        aid, peso, sexo = extraer(m.groups())
        mpath = mask_dir / f"{fn}___fuse.png"
        if not mpath.exists():
            continue
        mask = np.array(Image.open(mpath).convert("RGB"))
        im = imgs[ann["image_id"]]
        tam = sticker_size_px(mask, STICKER_RGB[nombre_batch], im["width"], im["height"])
        if tam is None:
            continue

        kp = np.array(ann["keypoints"], float).reshape(len(nombres), 3)[:, :2]
        fila = {"batch": nombre_batch, "grupo": f"{nombre_batch}_{aid}_{peso}_{sexo}",
                "peso_kg": float(peso)}
        for a, b in pares:
            fila[f"d_{a}_{b}"] = float(np.linalg.norm(kp[idx[a]] - kp[idx[b]])) / tam
        filas.append(fila)

    print(f"  {nombre_batch}: {len(filas)} filas, {len(pares)} distancias calibradas c/u")
    return pd.DataFrame(filas)


def evaluar(df, cols, etiqueta, franjas=None):
    X = df[cols].to_numpy(float)
    y = df["peso_kg"].to_numpy(float)
    g = df["grupo"].to_numpy()
    base = np.mean(np.abs((y - np.median(y)) / y)) * 100
    print(f"\n=== {etiqueta} (n={len(df)}, peso {y.min():.0f}-{y.max():.0f} kg, "
          f"baseline {base:.2f}%, {len(cols)} features) ===")
    for nombre, m in [
        ("Ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40)))),
        ("Boosting", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=5,
            l2_regularization=1.0, random_state=0)),
    ]:
        cv = GroupKFold(n_splits=5)
        p = cross_val_predict(m, X, y, cv=cv, groups=g)
        mape = np.mean(np.abs((y - p) / y)) * 100
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {nombre:9s} MAPE {mape:5.2f}%   R2 {r2:+.3f}")
        if nombre == "Boosting" and franjas:
            err = np.abs((y - p) / y) * 100
            for lo, hi in franjas:
                s = (y >= lo) & (y < hi)
                if s.sum() >= 10:
                    print(f"      {lo:4d}-{hi if hi < 9999 else '+':>4} kg  "
                          f"n={s.sum():4d}  MAPE {err[s].mean():5.2f}%")


def main():
    if not config.DATASET_ACMEAI.exists():
        print(f"No se encontro el dataset en {config.DATASET_ACMEAI}")
        print("Ver CLAUDE.md para donde conseguirlo.")
        return

    pat_b2 = re.compile(r"^([\d.]+)_s_(\d+)_[\d.]+_([MF])\.jpg$", re.IGNORECASE)
    pat_b3 = re.compile(r"^(\d+)_s_(\d+)_([MF])\.jpg$", re.IGNORECASE)
    pat_b4 = re.compile(r"^(\d+)_(b4-\d+)_s_(\d+)_([MF])\.jpg$", re.IGNORECASE)

    print("Procesando batches...")
    df2 = procesar("b2", "Vector/B2/Side/data/Side/COCO_Side.json",
                    "Pixel/B2/Side/annotations", pat_b2, lambda g: (g[0], g[1], g[2]))
    df3 = procesar("b3", "Vector/B3/Side/data/COCO_Side.json",
                    "Pixel/B3/annotations", pat_b3, lambda g: (g[0], g[1], g[2]))
    df4 = procesar("b4", "Vector/B4/Side/data/coco_b4_side.json",
                    "Pixel/B4/Side/annotations", pat_b4, lambda g: (g[0], g[2], g[3]))

    config.TABLAS.mkdir(parents=True, exist_ok=True)
    df3.to_csv(config.TABLAS / "acmeai_b3_distancias.csv", index=False)
    df4.to_csv(config.TABLAS / "acmeai_b4_distancias.csv", index=False)
    df2.to_csv(config.TABLAS / "acmeai_b2_distancias.csv", index=False)

    # --- resultado principal: B3+B4, comparten los 9 landmarks -> 36 pares ---
    cols_comunes_34 = sorted(set(c for c in df3.columns if c.startswith("d_"))
                              & set(c for c in df4.columns if c.startswith("d_")))
    df34 = pd.concat([df3, df4], ignore_index=True)
    franjas = [(0, 100), (100, 200), (200, 300), (300, 10_000)]
    evaluar(df34, cols_comunes_34,
            "B3+B4, todas las distancias (36 pares) -- NUMERO A REPORTAR", franjas)

    # --- chequeo secundario con B2 incluido: solo el subconjunto de puntos que
    #     tienen los 3 batches (wither, pinbone, front/rear top/bottom) ---
    comunes_3 = {"wither", "pinbone", "front_top", "front_bottom", "rear_top", "rear_bottom"}
    cols_comunes_3batches = sorted(f"d_{a}_{b}" for a, b in itertools.combinations(sorted(comunes_3), 2))
    dfx = pd.concat([df2, df3, df4], ignore_index=True)
    faltan = [c for c in cols_comunes_3batches if c not in dfx.columns]
    if faltan:
        print(f"\n(no se pudo armar el chequeo con B2: faltan columnas {faltan})")
    else:
        evaluar(dfx, cols_comunes_3batches, "B2+B3+B4, subconjunto de puntos comun a los 3")


if __name__ == "__main__":
    main()
