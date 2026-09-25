"""Fase 1: CNN fine-tuneada para estimar peso desde la imagen, SIN marcador de referencia.

Reemplaza al Modelo A (que usaba un ResNet18 congelado sobre la imagen completa a 224px y daba 19,6%).
Acá el backbone se reentrena completo, con encuadre y resolución configurables, para la ablación del plan.

Entrada: las imágenes ya reducidas a 1024 px por 06a_preparar_pose.py (data/pose/images/<split>/), así no hay que
releer los JPG originales de 1900 px en cada época.

Opciones de encuadre (--encuadre):
  completa : la imagen tal cual (como el Modelo A actual)
  crop     : recorte al animal con margen, derivado de los KEYPOINTS PREDICHOS (no de la máscara del dataset,
             para que sea honesto en producción)
  crop_mascara : igual que crop, pero agrega la máscara del animal como 4º canal (no borra el fondo: la
             literatura reporta que borrarlo empeora, así que se marca el animal en vez de recortarlo del fondo)

Uso (Fedora):  HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run python 20_train_cnn.py --backbone convnext_tiny --tam 384 --encuadre crop
Humo (CPU):    uv run python 20_train_cnn.py --epocas 1 --limite-train 40 --limite-val 20 --tam 128 --workers 0 --nombre humo
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from common import metrics
from common.acmeai import KP_CANONICOS

LADO_POSE = 1024  # lado mayor de las imágenes preparadas por 06a_preparar_pose.py


class BovinosDataset(Dataset):
    def __init__(self, df, kp, encuadre, tam, entrenando, sin_aug_escala=False, dir_imagenes="images_sin_sticker"):
        self.df = df.reset_index(drop=True)
        self.kp = kp
        self.encuadre = encuadre
        self.tam = tam
        self.entrenando = entrenando
        self.sin_aug_escala = sin_aug_escala
        self.dir_imagenes = dir_imagenes
        from torchvision.transforms import v2
        media, desvio = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if encuadre == "crop_mascara":  # 4º canal: media/desvío neutros para la máscara
            media, desvio = media + [0.5], desvio + [0.25]
        if entrenando:
            geom = ([v2.Resize((tam, tam))] if sin_aug_escala else
                    [v2.RandomResizedCrop(tam, scale=(0.7, 1.0), ratio=(0.8, 1.25), antialias=True)])
            self.tfm = v2.Compose(geom + [
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(media, desvio),
                # borra parches al azar: evita que el modelo se apoye en una mancha concreta (por ejemplo, la
                # zona donde se tapó el sticker) en vez de en la forma del cuerpo
                v2.RandomErasing(p=0.5, scale=(0.01, 0.05)),
            ])
            self.color = v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)
        else:
            self.tfm = v2.Compose([v2.Resize((tam, tam)),
                                   v2.ToDtype(torch.float32, scale=True),
                                   v2.Normalize(media, desvio)])
            self.color = None

    def __len__(self):
        return len(self.df)

    def _caja_animal(self, r):
        """Caja del animal en coordenadas de la imagen de 1024 px, a partir de los keypoints predichos."""
        f = LADO_POSE / max(r.ancho, r.alto)
        k = self.kp.loc[r.image_id]
        xs = np.array([k[f"{n}_x"] for n in KP_CANONICOS]) * f
        ys = np.array([k[f"{n}_y"] for n in KP_CANONICOS]) * f
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        mx, my = 0.10 * (x1 - x0), 0.15 * (y1 - y0)  # margen: deja algo de contexto alrededor
        return x0 - mx, y0 - my, x1 + mx, y1 + my

    def _mascara_1024(self, r):
        """Máscara binaria del animal reducida a 1024 px, cacheada en disco la primera vez que se usa."""
        cache = config.CACHE / "mascaras_1024"
        cache.mkdir(parents=True, exist_ok=True)
        destino = cache / (r.image_id.replace("/", "__")[:-4] + ".png")
        if not destino.exists():
            from common import masks
            rgba = np.array(Image.open(config.ACMEAI / r.ruta_mascara).convert("RGBA"))
            vaca, _ = masks.separar(rgba)
            m = Image.fromarray((vaca * 255).astype(np.uint8))
            f = LADO_POSE / max(m.size)
            if f < 1:
                m = m.resize((round(m.width * f), round(m.height * f)), Image.NEAREST)
            m.save(destino)
        return Image.open(destino)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # el manifest unificado trae su propia carpeta por fila (AcmeAI y los datasets externos conviven)
        carpeta = getattr(r, "dir_imagenes", None) or self.dir_imagenes
        ruta = config.DATA / "pose" / carpeta / r.split / r.image_id.replace("/", "__")
        im = Image.open(ruta).convert("RGB")
        msk = self._mascara_1024(r) if self.encuadre == "crop_mascara" else None

        if msk is not None and msk.size != im.size:
            # ~1% de las máscaras (sobre todo B4) vienen rotadas o con otro tamaño: se alinean a la imagen
            msk = msk.resize(im.size, Image.NEAREST)
        if self.encuadre in ("crop", "crop_mascara"):
            x0, y0, x1, y1 = self._caja_animal(r)
            caja = (max(0, int(x0)), max(0, int(y0)), min(im.width, int(x1)), min(im.height, int(y1)))
            if caja[2] - caja[0] > 10 and caja[3] - caja[1] > 10:
                im = im.crop(caja)
                if msk is not None:
                    msk = msk.crop(caja)

        from torchvision.transforms import v2
        x = v2.functional.pil_to_tensor(im)
        if self.color is not None:
            x = self.color(x)  # solo sobre RGB: el 4º canal se agrega después
        if msk is not None:
            mt = v2.functional.pil_to_tensor(msk)
            if mt.shape[-2:] != x.shape[-2:]:
                mt = v2.functional.resize(mt, list(x.shape[-2:]), antialias=False)
            x = torch.cat([x, mt], dim=0)
        x = self.tfm(x)
        return x, torch.tensor(np.log(r.peso_kg), dtype=torch.float32), i


def construir_modelo(backbone, canales, congelar):
    import timm
    modelo = timm.create_model(backbone, pretrained=True, num_classes=1, in_chans=canales)
    if congelar:
        for nombre, p in modelo.named_parameters():
            if "head" not in nombre and "fc" not in nombre and "classifier" not in nombre:
                p.requires_grad = False
    return modelo


def evaluar(modelo, loader, dispositivo, tta=False):
    modelo.eval()
    preds, idxs = [], []
    with torch.no_grad():
        for x, _, i in loader:
            x = x.to(dispositivo, non_blocking=True)
            with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=dispositivo.type == "cuda"):
                p = modelo(x).squeeze(1).float()
                if tta:  # promedio con la imagen espejada
                    p = (p + modelo(torch.flip(x, dims=[3])).squeeze(1).float()) / 2
            preds.append(p.cpu().numpy())
            idxs.append(i.numpy())
    return np.concatenate(idxs), np.exp(np.concatenate(preds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--encuadre", choices=["completa", "crop", "crop_mascara"], default="crop")
    ap.add_argument("--tam", type=int, default=384)
    ap.add_argument("--epocas", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--loss", choices=["l1", "mse"], default="l1")
    ap.add_argument("--sin-aug-escala", action="store_true",
                    help="usa Resize en vez de RandomResizedCrop: conserva el tamaño aparente")
    ap.add_argument("--congelar", action="store_true", help="control: backbone congelado (como el Modelo A viejo)")
    ap.add_argument("--imagenes", choices=["sin_sticker", "con_sticker"], default="sin_sticker",
                    help="sin_sticker (default, escenario real) | con_sticker (control: cuánto aportaba el marcador)")
    ap.add_argument("--unificado", action="store_true",
                    help="entrena con AcmeAI + CID + Mendeley (36-816 kg) en vez de solo AcmeAI. "
                         "Requiere haber corrido 29_dataset_unificado.py")
    ap.add_argument("--balancear", action="store_true",
                    help="muestrea cada banda de peso con igual probabilidad. Sin esto el modelo comprime hacia "
                         "el centro (medido: nunca predice fuera de 80-267 kg aunque el real llega a 465)")
    ap.add_argument("--semilla", type=int, default=None,
                    help="siembra el entrenamiento (inicialización de la cabeza, orden de los lotes, "
                         "augmentation). El split NO cambia: su incertidumbre se mide aparte por bootstrap "
                         "en 30_intervalos.py")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--paciencia", type=int, default=8)
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--limite-train", type=int, default=0)
    ap.add_argument("--limite-val", type=int, default=0)
    a = ap.parse_args()
    nombre = (a.nombre or f"{a.backbone}_{a.encuadre}_{a.tam}_{a.loss}" + ("_noaug" if a.sin_aug_escala else "")
              + ("_bal" if a.balancear else "") + ("_unif" if a.unificado else "")
              + (f"_s{a.semilla}" if a.semilla is not None else "")
              + ("_CONSTICKER" if a.imagenes == "con_sticker" else ""))

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dispositivo={dispositivo} | {nombre}")
    if a.semilla is not None:
        import random
        torch.manual_seed(a.semilla)
        torch.cuda.manual_seed_all(a.semilla)
        np.random.seed(a.semilla)
        random.seed(a.semilla)
        print(f"semilla de entrenamiento = {a.semilla}")

    sufijo = "unificado" if a.unificado else None
    man = pd.read_parquet(config.TABLAS / (f"manifest_{sufijo}.parquet" if sufijo else "manifest.parquet"))
    kp = pd.read_parquet(config.TABLAS / (f"keypoints_{sufijo}.parquet" if sufijo else
                                          "keypoints_pred.parquet")).set_index("image_id")
    man = man[man.image_id.isin(kp.index)].reset_index(drop=True)
    if a.unificado:
        print(man.groupby("origen").agg(n=("image_id", "size"), min=("peso_kg", "min"), max=("peso_kg", "max")).to_string())
    tr, va, te = [man[man.split == s] for s in ("train", "val", "test")]
    if a.limite_train:
        tr = tr.head(a.limite_train)
    if a.limite_val:
        va, te = va.head(a.limite_val), te.head(a.limite_val)

    canales = 4 if a.encuadre == "crop_mascara" else 3
    dir_img = "images_sin_sticker" if a.imagenes == "sin_sticker" else "images"
    if not (config.DATA / "pose" / dir_img).exists():
        raise SystemExit(f"falta {config.DATA / 'pose' / dir_img} — correr antes 20a_tapar_sticker.py")
    ds = lambda d, ent: BovinosDataset(d, kp, a.encuadre, a.tam, ent, a.sin_aug_escala, dir_img)
    dl = lambda d, ent: DataLoader(ds(d, ent), batch_size=a.batch, shuffle=ent, num_workers=a.workers,
                                   pin_memory=dispositivo.type == "cuda", drop_last=ent)
    if a.balancear:
        from torch.utils.data import WeightedRandomSampler
        bandas = pd.cut(tr.peso_kg, config.BANDAS_PESO, right=False)
        conteo = bandas.value_counts()
        pesos = np.array([1.0 / conteo[b] if conteo[b] else 0.0 for b in bandas], dtype=float)
        print("muestreo balanceado por banda:", {str(k): int(v) for k, v in conteo.sort_index().items()})
        dl_tr = DataLoader(ds(tr, True), batch_size=a.batch, num_workers=a.workers, drop_last=True,
                           pin_memory=dispositivo.type == "cuda",
                           sampler=WeightedRandomSampler(pesos, num_samples=len(tr), replacement=True))
    else:
        dl_tr = dl(tr, True)
    dl_va, dl_te = dl(va, False), dl(te, False)

    modelo = construir_modelo(a.backbone, canales, a.congelar).to(dispositivo)
    cabeza = [p for n, p in modelo.named_parameters() if p.requires_grad and ("head" in n or "fc" in n or "classifier" in n)]
    resto = [p for n, p in modelo.named_parameters() if p.requires_grad and not ("head" in n or "fc" in n or "classifier" in n)]
    opt = torch.optim.AdamW([{"params": resto, "lr": a.lr_backbone}, {"params": cabeza, "lr": a.lr}], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr], total_steps=a.epocas * max(1, len(dl_tr)))
    criterio = nn.L1Loss() if a.loss == "l1" else nn.MSELoss()
    escalador = torch.amp.GradScaler(enabled=dispositivo.type == "cuda")

    salida = config.RUNS / "cnn" / nombre
    salida.mkdir(parents=True, exist_ok=True)
    mejor_mape, mejor_epoca, historial = float("inf"), -1, []

    for epoca in range(a.epocas):
        modelo.train()
        t0, perdidas = time.time(), []
        for x, y, _ in dl_tr:
            x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=dispositivo.type == "cuda"):
                perdida = criterio(modelo(x).squeeze(1), y)
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            sched.step()
            perdidas.append(perdida.item())
        idx, pred = evaluar(modelo, dl_va, dispositivo)
        mape_val = metrics.mape(va.iloc[idx].peso_kg, pred)
        historial.append({"epoca": epoca, "perdida": float(np.mean(perdidas)), "mape_val": mape_val,
                          "seg": round(time.time() - t0, 1)})
        print(f"  época {epoca:3d} | pérdida {np.mean(perdidas):.4f} | MAPE val {mape_val:.2f}% | {time.time() - t0:.0f}s")
        if mape_val < mejor_mape:
            mejor_mape, mejor_epoca = mape_val, epoca
            torch.save(modelo.state_dict(), salida / "mejor.pt")
        elif epoca - mejor_epoca >= a.paciencia:
            print(f"  corte por paciencia (mejor época {mejor_epoca}, MAPE val {mejor_mape:.2f}%)")
            break

    modelo.load_state_dict(torch.load(salida / "mejor.pt"))
    filas, preds_guardar = [], []
    for split, d, loader in (("val", va, dl_va), ("test", te, dl_te)):
        for tta in (False, True):
            idx, pred = evaluar(modelo, loader, dispositivo, tta=tta)
            real = d.iloc[idx]
            filas.append({"split": split, "tta": tta, **metrics.reporte(tr.peso_kg, real.peso_kg, pred, nombre)})
            if tta:
                g = pd.DataFrame({"image_id": real.image_id.values, "split": split,
                                  "peso_kg": real.peso_kg.values, "pred_cnn": pred})
                if "origen" in real.columns:
                    g["origen"] = real.origen.values
                preds_guardar.append(g)
    res = pd.DataFrame(filas)[["split", "tta", "modelo", "n", "mape", "mape_mediana", "mejora_pct"]]
    print(res.round(2).to_string(index=False))
    res.to_csv(salida / "resultados.csv", index=False)
    (salida / "historial.json").write_text(json.dumps({"config": vars(a), "historial": historial}, indent=2))
    pd.concat(preds_guardar).to_parquet(config.TABLAS / f"pred_cnn_{nombre}.parquet", index=False)

    te_pred = preds_guardar[-1]
    print(f"\nMAPE por banda (test, con TTA):")
    print(metrics.mape_por_banda(te_pred.peso_kg, te_pred.pred_cnn).round(2).to_string(index=False))
    print(f"rango de predicciones en test: {te_pred.pred_cnn.min():.0f}-{te_pred.pred_cnn.max():.0f} kg "
          f"(real {te_pred.peso_kg.min():.0f}-{te_pred.peso_kg.max():.0f})")
    if "origen" in te_pred.columns:
        print("\nMAPE por dataset de origen (test):")
        print(te_pred.groupby("origen").apply(
            lambda d: pd.Series({"n": len(d), "mape": metrics.mape(d.peso_kg, d.pred_cnn),
                                 "mape_mediana": metrics.mape(d.peso_kg, np.full(len(d), tr.peso_kg.median()))}),
            include_groups=False).round(2).to_string())

    resumen = config.RUNS / "cnn" / "resumen.csv"
    fila = res[(res.split == "test") & res.tta].assign(nombre=nombre, backbone=a.backbone, encuadre=a.encuadre,
                                                       tam=a.tam, loss=a.loss, sin_aug_escala=a.sin_aug_escala,
                                                       congelar=a.congelar, imagenes=a.imagenes,
                                                       balanceado=a.balancear, unificado=a.unificado,
                                                       semilla=a.semilla,
                                                       mape_val=round(mejor_mape, 3), mejor_epoca=mejor_epoca)
    # se reescribe el CSV entero (en vez de append) para que agregar columnas nuevas no rompa el archivo
    if resumen.exists():
        fila = pd.concat([pd.read_csv(resumen), fila], ignore_index=True)
    fila.to_csv(resumen, index=False)
    print(f"-> {salida}  |  resumen acumulado: {resumen}")


if __name__ == "__main__":
    main()
