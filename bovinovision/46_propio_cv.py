"""Dataset propio, etapa 6: fine-tuning de la CNN sobre las vacas propias con VALIDACIÓN CRUZADA POR ANIMAL.

Por qué CV y no el split train/val/test del dataset: el test tiene 21 animales y cualquier MAPE ahí tiene un intervalo
enorme. Con 5 folds por animal cada una de las 141 vacas se evalúa exactamente una vez, con un modelo que no la vio
(predicciones "out-of-fold", OOF), y el resultado sale de 141 animales en vez de 21.

Todo es ADITIVO: no modifica 20_train_cnn.py, ni estimador/modelos/, ni runs/cnn/. Salidas en runs/propio_cv/<nombre>/.

Qué compara (--init):
  bangladesh : arranca del modelo actual (estimador/modelos/peso_convnext.pt) y lo ajusta a las vacas propias
  imagenet   : arranca de ConvNeXt-Tiny preentrenado en ImageNet (como se entrenó el modelo de Bangladesh)
  (--congelar: control, solo se reentrena la cabeza)
Referencias que imprime siempre: predecir la mediana de los animales de entrenamiento de cada fold, y el Ridge sobre el
embedding congelado (45_propio_embeddings.py: 7,2 %).

Folds estratificados por peso (cada fold ve todo el rango). Un 14 % de los animales de entrenamiento se aparta como
validación interna SOLO para elegir la época; el fold de test nunca interviene en el entrenamiento ni en la elección.

Uso (Fedora, GPU):  HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run --extra rocm python 46_propio_cv.py --init bangladesh
Humo (Windows CPU): uv run python 46_propio_cv.py --epocas 1 --solo-fold 0 --limite-animales 20 --tam 128 --workers 0 --nombre humo
Modelo final con todos los animales:  agregar --final  (se guarda en runs/propio_cv/<nombre>/final/, nunca en estimador/modelos/)
"""
import argparse
import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

import config
from common import metrics

ESTIMADOR = config.ROOT.parent / "estimador"
PESOS_BANGLADESH = ESTIMADOR / "modelos" / "peso_convnext.pt"
CROPS = config.DATA / "propio" / "crops"
MEDIA, DESVIO = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- recortes (se calculan una vez y quedan en caché)
def nombre_crop(rel):
    return rel.replace("/", "__")


def construir_crops(rutas):
    """Recorta cada foto al animal con el MISMO procedimiento del estimador (detector -> pose -> margen) y la guarda."""
    faltan = [r for r in rutas if not (CROPS / nombre_crop(r)).exists()]
    if not faltan:
        return
    print(f"armando {len(faltan)} recortes (una sola vez)...", flush=True)
    sys.path.insert(0, str(ESTIMADOR))
    from estimador import EstimadorPeso
    est = EstimadorPeso(ESTIMADOR / "modelos")
    CROPS.mkdir(parents=True, exist_ok=True)
    for r in faltan:
        im = Image.open(config.PROPIO / r).convert("RGB")
        escena, _, _ = est._aislar_animal(im)
        res = est.pose.predict(source=escena, imgsz=640, conf=0.05, max_det=3, verbose=False)[0]
        recorte = escena
        if len(res.boxes):
            k = int(res.boxes.conf.argmax())
            xy = res.keypoints.xy[k].cpu().numpy()
            recorte, _ = est._recortar(escena, xy[:, 0], xy[:, 1])
        recorte.save(CROPS / nombre_crop(r), quality=95)


class Vacas(Dataset):
    def __init__(self, df, tam, entrenando):
        from torchvision.transforms import v2
        self.df = df.reset_index(drop=True)
        self.entrenando = entrenando
        if entrenando:   # mismo augmentation que 20_train_cnn.py
            self.tfm = v2.Compose([v2.RandomResizedCrop(tam, scale=(0.7, 1.0), ratio=(0.8, 1.25), antialias=True),
                                   v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True),
                                   v2.Normalize(MEDIA, DESVIO), v2.RandomErasing(p=0.5, scale=(0.01, 0.05))])
            self.color = v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)
        else:
            self.tfm = v2.Compose([v2.Resize((tam, tam)), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEDIA, DESVIO)])
            self.color = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from torchvision.transforms import v2
        r = self.df.iloc[i]
        x = v2.functional.pil_to_tensor(Image.open(CROPS / nombre_crop(r.archivo_curado)).convert("RGB"))
        if self.color is not None:
            x = self.color(x)
        return self.tfm(x), torch.tensor(np.log(r.peso_kg), dtype=torch.float32), i


def construir_modelo(init, congelar):
    import timm
    if init == "bangladesh":
        m = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
        m.load_state_dict(torch.load(PESOS_BANGLADESH, map_location="cpu"))
    else:
        m = timm.create_model("convnext_tiny", pretrained=True, num_classes=1)
    if congelar:
        for n, p in m.named_parameters():
            p.requires_grad = "head" in n
    return m


def predecir(modelo, loader, dispositivo, tta=True):
    modelo.eval()
    preds, idxs = [], []
    with torch.no_grad():
        for x, _, i in loader:
            x = x.to(dispositivo)
            with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=dispositivo.type == "cuda"):
                p = modelo(x).squeeze(1).float()
                if tta:
                    p = (p + modelo(torch.flip(x, dims=[3])).squeeze(1).float()) / 2
            preds.append(p.cpu().numpy())
            idxs.append(i.numpy())
    return np.concatenate(idxs), np.exp(np.concatenate(preds))


def entrenar(tr, va, a, dispositivo):
    """Ajusta el modelo en `tr` y devuelve el mejor estado según el MAPE en `va` (validación interna)."""
    modelo = construir_modelo(a.init, a.congelar).to(dispositivo)
    dl = lambda d, ent: DataLoader(Vacas(d, a.tam, ent), batch_size=a.batch, shuffle=ent, num_workers=a.workers,
                                   drop_last=ent and len(d) > a.batch, pin_memory=dispositivo.type == "cuda")
    dl_tr, dl_va = dl(tr, True), dl(va, False)
    cab = [p for n, p in modelo.named_parameters() if p.requires_grad and "head" in n]
    res = [p for n, p in modelo.named_parameters() if p.requires_grad and "head" not in n]
    grupos = ([{"params": res, "lr": a.lr_backbone}] if res else []) + [{"params": cab, "lr": a.lr}]
    opt = torch.optim.AdamW(grupos, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in grupos],
                                                total_steps=a.epocas * max(1, len(dl_tr)))
    crit = nn.L1Loss() if a.loss == "l1" else nn.MSELoss()
    esc = torch.amp.GradScaler(enabled=dispositivo.type == "cuda")
    mejor, mejor_ep, estado, hist = float("inf"), -1, None, []
    for ep in range(a.epocas):
        modelo.train()
        perd = []
        for x, y, _ in dl_tr:
            x, y = x.to(dispositivo), y.to(dispositivo)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=dispositivo.type == "cuda"):
                p = crit(modelo(x).squeeze(1), y)
            esc.scale(p).backward()
            esc.step(opt)
            esc.update()
            sched.step()
            perd.append(p.item())
        idx, pred = predecir(modelo, dl_va, dispositivo, tta=False)
        m = metrics.mape(va.iloc[idx].peso_kg, pred)
        hist.append({"epoca": ep, "perdida": float(np.mean(perd)), "mape_val": m})
        if m < mejor:
            mejor, mejor_ep, estado = m, ep, copy.deepcopy(modelo.state_dict())
        elif ep - mejor_ep >= a.paciencia:
            break
    modelo.load_state_dict(estado)
    return modelo, mejor, mejor_ep, hist


# ---------------------------------------------------------------- folds
def asignar_folds(animales, k, seed):
    """Fold por animal, estratificado por peso: se ordenan por peso (empates al azar) y se reparten en ronda."""
    rng = np.random.default_rng(seed)
    orden = np.argsort(animales.to_numpy() + rng.normal(0, 1e-6, len(animales)))
    f = np.empty(len(animales), dtype=int)
    f[orden] = np.arange(len(animales)) % k
    return pd.Series(f, index=animales.index)


def separar_validacion(animales_tr, seed, cada=7):
    """~14 % de los animales de entrenamiento (uno de cada 7 por peso) quedan como validación interna."""
    rng = np.random.default_rng(seed)
    orden = np.argsort(animales_tr.to_numpy() + rng.normal(0, 1e-6, len(animales_tr)))
    es_val = np.zeros(len(animales_tr), dtype=bool)
    es_val[orden[rng.integers(0, cada)::cada]] = True
    return animales_tr.index[es_val], animales_tr.index[~es_val]


def resumir(oof, salida, nombre):
    """Una predicción por animal (promedio geométrico de sus fotos), con la mediana como referencia y bootstrap por animal."""
    g = oof.groupby("animal_id")
    por = pd.DataFrame({"real": g.peso_kg.first(), "fold": g.fold.first(), "mediana_train": g.mediana_train.first(),
                        "pred": np.exp(g.apply(lambda d: np.log(d.pred).mean(), include_groups=False))})
    err_m = (por.pred - por.real).abs() / por.real * 100
    err_b = (por.mediana_train - por.real).abs() / por.real * 100
    rng = np.random.default_rng(config.SEED)
    dif = [(err_b.to_numpy()[i] - err_m.to_numpy()[i]).mean() for i in (rng.integers(0, len(por), len(por)) for _ in range(2000))]
    r = {"nombre": nombre, "animales": int(len(por)), "mape_modelo": float(err_m.mean()), "mape_mediana": float(err_b.mean()),
         "mejora_puntos": float(err_b.mean() - err_m.mean()),
         "mejora_ic95": [float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))],
         "sesgo_pct": float(((por.pred - por.real) / por.real).mean() * 100), "spearman": float(spearmanr(por.real, por.pred)[0]),
         "pred_min": float(por.pred.min()), "pred_max": float(por.pred.max()),
         "mape_por_fold": {int(k): float(v) for k, v in err_m.groupby(por.fold).mean().items()},
         "mape_fotos": float(metrics.mape(oof.peso_kg, oof.pred))}
    por.to_csv(salida / "oof_por_animal.csv")
    (salida / "resumen.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["bangladesh", "imagenet"], default="bangladesh")
    ap.add_argument("--congelar", action="store_true", help="control: solo se reentrena la cabeza")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tam", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--loss", choices=["l1", "mse"], default="mse")
    ap.add_argument("--paciencia", type=int, default=6)
    ap.add_argument("--semilla", type=int, default=1, help="azar del ENTRENAMIENTO; los folds son fijos (config.SEED)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--solo-fold", type=int, default=None, help="correr un único fold (humo / paralelizar)")
    ap.add_argument("--limite-animales", type=int, default=0, help="humo: usar solo los primeros N animales")
    ap.add_argument("--final", action="store_true", help="además, entrenar con TODOS los animales y guardar el modelo")
    ap.add_argument("--solo-final", action="store_true", help="saltear la validación cruzada y entrenar solo el modelo final")
    a = ap.parse_args()
    nombre = a.nombre or f"{a.init}{'_congelado' if a.congelar else ''}_{a.loss}_s{a.semilla}"
    salida = config.RUNS / "propio_cv" / nombre
    salida.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.semilla)
    np.random.seed(a.semilla)
    print(f"dispositivo={dispositivo} | {nombre} | salida: {salida}", flush=True)

    df = pd.read_csv(config.PROPIO / "fotos.csv", dtype={"animal_id": str})
    df = df[df.uso == "incluida"].reset_index(drop=True)
    animales = df.groupby("animal_id").peso_kg.first().sort_index()
    if a.limite_animales:
        animales = animales.iloc[:a.limite_animales]
        df = df[df.animal_id.isin(animales.index)].reset_index(drop=True)
    construir_crops(list(df.archivo_curado))
    fold = asignar_folds(animales, a.folds, config.SEED)
    df["fold"] = df.animal_id.map(fold)
    print(f"{len(df)} fotos, {len(animales)} animales, {a.folds} folds de {np.bincount(fold)} animales", flush=True)

    filas = []
    a.final = a.final or a.solo_final
    for k in ([] if a.solo_final else [a.solo_fold] if a.solo_fold is not None else range(a.folds)):
        t0 = time.time()
        an_tr = animales[fold != k]
        an_va, an_ent = separar_validacion(an_tr, config.SEED + k)
        tr, va, te = df[df.animal_id.isin(an_ent)], df[df.animal_id.isin(an_va)], df[df.fold == k]
        modelo, mejor, ep, _ = entrenar(tr, va, a, dispositivo)
        idx, pred = predecir(modelo, DataLoader(Vacas(te, a.tam, False), batch_size=a.batch, num_workers=a.workers), dispositivo)
        t = te.iloc[idx].copy()
        t["pred"], t["mediana_train"] = pred, float(an_tr.median())
        filas.append(t[["foto", "animal_id", "fold", "peso_kg", "pred", "mediana_train"]])
        print(f"fold {k}: entrena {len(an_ent)} animales, valida {len(an_va)}, evalúa {len(animales[fold == k])} | "
              f"mejor época {ep} (MAPE val {mejor:.1f}%) | MAPE test fotos {metrics.mape(t.peso_kg, t.pred):.1f}% | {time.time() - t0:.0f}s",
              flush=True)
        del modelo
    r = {"mape_modelo": None, "mape_mediana": None}
    if filas:
        oof = pd.concat(filas, ignore_index=True)
        oof.to_csv(salida / "oof_fotos.csv", index=False)
        r = resumir(oof, salida, nombre)
        print(f"\n=== {nombre} | {r['animales']} animales (OOF) ===")
        print(f"MAPE modelo {r['mape_modelo']:.2f}% | mediana {r['mape_mediana']:.2f}% | mejora {r['mejora_puntos']:+.2f} puntos "
              f"(IC95 {r['mejora_ic95'][0]:+.2f} a {r['mejora_ic95'][1]:+.2f}) | Spearman {r['spearman']:+.2f} | "
              f"predice {r['pred_min']:.0f}-{r['pred_max']:.0f} kg | sesgo {r['sesgo_pct']:+.1f}%")

    if a.final:
        an_va, an_ent = separar_validacion(animales, config.SEED, cada=10)
        modelo, mejor, ep, hist = entrenar(df[df.animal_id.isin(an_ent)], df[df.animal_id.isin(an_va)], a, dispositivo)
        fin = salida / "final"
        fin.mkdir(exist_ok=True)
        torch.save(modelo.state_dict(), fin / "peso_convnext_propio.pt")
        (fin / "config.json").write_text(json.dumps({
            "version": f"{pd.Timestamp.now():%Y-%m-%d} / fine-tuning sobre dataset propio v1 ({len(animales)} animales, {a.init})",
            "archivo_pose": "pose9.pt", "archivo_cnn": "peso_convnext_propio.pt", "archivo_detector": "detector_coco.pt",
            "backbone": "convnext_tiny", "tam": a.tam,
            "rango_entrenamiento_kg": [float(animales.min()), float(animales.max())],
            "cv_mape_oof": r["mape_modelo"], "cv_mape_mediana": r["mape_mediana"],
            "nota": "Entrenado con 141 hembras de 292-485 kg de UN establecimiento. Fuera de ese rango no es confiable."},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"modelo final -> {fin} (época {ep}, MAPE val interno {mejor:.1f}%). NO reemplaza a estimador/modelos/.")


if __name__ == "__main__":
    main()
