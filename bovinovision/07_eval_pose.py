"""Etapa 5b: keypoints predichos por YOLO-pose para TODAS las imágenes -> data/tablas/keypoints_pred.parquet,
y error contra los keypoints anotados (val y test).

Error de cada keypoint = distancia en px / largo corporal GT (wither-pinbone), es decir, en % del largo del animal:
comparable entre imágenes de distinta escala. PCK@5% = fracción de puntos con error < 5% del largo corporal.
Las predicciones sobre 'train' están sobreajustadas: se guardan (para no re-predecir) pero no se usan para evaluar.
Uso:  uv run python 07_eval_pose.py --pesos runs/pose/pose9/weights/best.pt
"""
import argparse

import numpy as np
import pandas as pd

import config
from common.acmeai import KP_CANONICOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=str(config.RUNS / "pose" / "pose9" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4, help="bajar si da OOM en la GPU")
    ap.add_argument("--salida", default="keypoints_pred")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limite", type=int, default=0, help="solo las primeras N imágenes (pruebas)")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = a.device if a.device != "auto" else (0 if torch.cuda.is_available() else "cpu")
    modelo = YOLO(a.pesos)

    man = pd.read_parquet(config.TABLAS / "manifest.parquet").set_index("image_id")
    gt = pd.read_parquet(config.TABLAS / "keypoints_gt.parquet").set_index("image_id")
    dir_pose = config.DATA / "pose" / "images"
    ids = list(gt.index)[: a.limite or None]
    rutas = [dir_pose / man.loc[i, "split"] / i.replace("/", "__") for i in ids]

    # OJO: modelo.predict(source=lista, stream=True, batch=N) NO respeta `batch` cuando `source`
    # es una lista de rutas: arma un solo batch con TODAS las imágenes (confirmado con
    # predictor.dataset.bs). Hay que trocear la lista a mano en grupos de a.batch.
    filas = []
    rutas_str = [str(r) for r in rutas]
    i = 0
    for inicio in range(0, len(rutas_str), a.batch):
        grupo_ids = ids[inicio: inicio + a.batch]
        grupo_rutas = rutas_str[inicio: inicio + a.batch]
        resultados = modelo.predict(
            source=grupo_rutas, stream=False, imgsz=a.imgsz, batch=a.batch, conf=0.05, max_det=3,
            device=device, verbose=False)
        for image_id, res in zip(grupo_ids, resultados):
            i += 1
            fila = {"image_id": image_id, "n_det": len(res.boxes)}
            if len(res.boxes):
                k = int(res.boxes.conf.argmax())  # detección más confiable
                xyn, kc = res.keypoints.xyn[k].cpu().numpy(), res.keypoints.conf[k].cpu().numpy()
                fila["det_conf"] = float(res.boxes.conf[k])
                for j, nombre in enumerate(KP_CANONICOS):
                    fila[f"{nombre}_x"] = float(xyn[j, 0] * man.loc[image_id, "ancho"])
                    fila[f"{nombre}_y"] = float(xyn[j, 1] * man.loc[image_id, "alto"])
                    fila[f"{nombre}_conf"] = float(kc[j])
            filas.append(fila)
            if i % 500 == 0:
                print(f"  {i}/{len(ids)}")
    pred = pd.DataFrame(filas).set_index("image_id")
    pred.reset_index().to_parquet(config.TABLAS / f"{a.salida}.parquet", index=False)

    gt = gt.loc[ids]
    largo = np.hypot(gt.wither_x - gt.pinbone_x, gt.wither_y - gt.pinbone_y)
    sp = man.loc[ids, "split"]
    print(f"sin detección: {pred.det_conf.isna().sum()} de {len(pred)}")
    for split in ("val", "test"):
        m = (sp == split).to_numpy()
        filas = []
        for nombre in KP_CANONICOS:
            err = np.hypot(pred[f"{nombre}_x"] - gt[f"{nombre}_x"], pred[f"{nombre}_y"] - gt[f"{nombre}_y"]) / largo
            e = err[m]
            filas.append({"keypoint": nombre, "err_med_%largo": 100 * e.median(), "err_p90_%largo": 100 * e.quantile(0.9),
                          "PCK@5%": (e < 0.05).mean() * 100, "conf_media": pred.loc[m, f"{nombre}_conf"].mean()})
        t = pd.DataFrame(filas)
        print(f"\n== {split} ({m.sum()} imágenes) ==")
        print(t.round(2).to_string(index=False))
        t.to_csv(config.RUNS / f"eval_pose_{split}.csv", index=False)


if __name__ == "__main__":
    main()
