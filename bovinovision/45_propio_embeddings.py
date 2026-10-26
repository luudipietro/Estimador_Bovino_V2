"""Dataset propio, etapa 5b: ¿el EMBEDDING de la CNN actual trae información de peso en estos animales?

La salida del modelo de Bangladesh está saturada (no predice > ~270 kg), pero el embedding previo a la capa final puede
conservar el tamaño del animal. En CID (rango solapado) correlacionaba +0,82 con la altura real; en Mendeley (rango fuera
de lo visto) no (+0,02). Acá se prueba: embedding por foto -> promedio por animal -> Ridge sobre log(peso) con validación
cruzada 5 folds POR ANIMAL, comparado contra predecir la mediana.

Usa las fotos de dataset_propio/curado/ (ya orientadas y a 1024 px) y el mismo recorte que `estimador/`.
Cachea en data/propio/embeddings_v1.npz.   Uso:  uv run python 45_propio_embeddings.py
"""
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

import config

sys.path.insert(0, str(config.ROOT.parent / "estimador"))
from estimador import EstimadorPeso  # noqa: E402

CACHE = config.DATA / "propio" / "embeddings_v1.npz"


def mape(real, pred):
    return float(np.mean(np.abs(real - pred) / real) * 100)


def embeddings(fotos):
    est = EstimadorPeso(config.ROOT.parent / "estimador" / "modelos")
    out = []
    for i, rel in enumerate(fotos, 1):
        im = Image.open(config.PROPIO / rel).convert("RGB")
        escena, _, _ = est._aislar_animal(im)
        res = est.pose.predict(source=escena, imgsz=640, conf=0.05, max_det=3, verbose=False)[0]
        if len(res.boxes) == 0:
            out.append(np.full(est.cnn.num_features, np.nan, dtype=np.float32))
            continue
        k = int(res.boxes.conf.argmax())
        xy = res.keypoints.xy[k].cpu().numpy()
        recorte, _ = est._recortar(escena, xy[:, 0], xy[:, 1])
        x = est.tfm(est._v2.functional.pil_to_tensor(recorte)).unsqueeze(0).to(est.dispositivo)
        with torch.no_grad():
            f = lambda t: est.cnn.forward_head(est.cnn.forward_features(t), pre_logits=True).squeeze(0).cpu().numpy()
            out.append((f(x) + f(torch.flip(x, dims=[3]))) / 2)
        if i % 50 == 0:
            print(f"  {i}/{len(fotos)}", flush=True)
    return np.stack(out)


def main():
    fv = pd.read_csv(config.PROPIO / "fotos.csv", dtype={"animal_id": str})
    d = fv[fv.uso == "incluida"].reset_index(drop=True)
    if CACHE.exists() and list(np.load(CACHE, allow_pickle=True)["fotos"]) == list(d.archivo_curado):
        E = np.load(CACHE, allow_pickle=True)["E"]
    else:
        E = embeddings(list(d.archivo_curado))
        np.savez(CACHE, E=E, fotos=np.array(d.archivo_curado, dtype=object))
    ok = ~np.isnan(E).any(axis=1)
    print(f"{ok.sum()}/{len(d)} fotos con embedding (dim {E.shape[1]})")
    d, E = d[ok].reset_index(drop=True), E[ok]

    # un embedding por animal = promedio de sus fotos
    animales = d.groupby("animal_id", sort=True)
    A = np.stack([E[g.index].mean(axis=0) for _, g in animales])
    real = animales.peso_kg.first().to_numpy()
    ids = animales.size().index.to_numpy()
    y = np.log(real)
    print(f"{len(ids)} animales, {real.min():.0f}-{real.max():.0f} kg\n")

    res = {}
    for nombre, X in [("embedding completo (768)", A), ("embedding estandarizado", (A - A.mean(0)) / (A.std(0) + 1e-6))]:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=ids):
            m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        res[nombre] = mape(real, np.exp(oof))
    oof_med = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(A, y, groups=ids):
        oof_med[te] = np.median(real[tr])
    print("CV 5 folds por animal (MAPE):")
    print(f"  predecir la mediana (referencia) : {mape(real, oof_med):.1f}%")
    for k, v in res.items():
        print(f"  Ridge sobre {k:<26}: {v:.1f}%")

    # --- ¿la mejora es real? 30 repeticiones de CV con particiones distintas + bootstrap de animales ---
    dif, mr, mm = [], [], []
    for seed in range(30):
        oof, oofm = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(A, y, groups=ids):
            oof[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(A[tr], y[tr]).predict(A[te])
            oofm[te] = np.median(real[tr])
        mr.append(mape(real, np.exp(oof)))
        mm.append(mape(real, oofm))
        dif.append(mm[-1] - mr[-1])
    print(f"\n30 repeticiones de CV: Ridge {np.mean(mr):.2f}% ± {np.std(mr):.2f} | mediana {np.mean(mm):.2f}% ± {np.std(mm):.2f} "
          f"| mejora media {np.mean(dif):+.2f} puntos")
    rng = np.random.default_rng(config.SEED)
    e_r = np.abs(real - np.exp(oof)) / real * 100
    e_m = np.abs(real - oofm) / real * 100
    boots = [(e_m[i] - e_r[i]).mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    print(f"bootstrap por animal de la mejora (última repetición): {np.mean(boots):+.2f} puntos, IC95% "
          f"[{np.percentile(boots, 2.5):+.2f}, {np.percentile(boots, 97.5):+.2f}]")

    # --- entrenar con un día y evaluar en el otro (cambia de teléfono, de lote y de luz) ---
    dia = d.groupby("animal_id").dia.first().loc[ids].to_numpy()
    print("\nGeneralización entre jornadas (entrena en un día, evalúa en el otro):")
    for a, b in [("8-10", "22-10"), ("22-10", "8-10")]:
        tr, te = dia == a, dia == b
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(A[tr], y[tr])
        print(f"  {a} -> {b}: Ridge {mape(real[te], np.exp(m.predict(A[te]))):.1f}% | "
              f"mediana de {a} {mape(real[te], np.median(real[tr])):.1f}% | mediana propia de {b} {mape(real[te], np.median(real[te])):.1f}%")


if __name__ == "__main__":
    main()
