"""Etapa 5: entrena YOLO-pose (9 keypoints) sobre data/pose. Pensado para la GPU AMD en Fedora.

El .yaml se regenera en cada corrida con la ruta ABSOLUTA de esta copia del proyecto: así funciona igual en
Windows y Fedora sin depender del directorio de trabajo (Ultralytics resuelve 'path' contra el cwd).
Uso (Fedora):  HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run python 06_train_pose.py --modelo yolo11s-pose.pt --epochs 150
Prueba rápida: uv run python 06_train_pose.py --epochs 1 --imgsz 320 --fraccion 0.02 --nombre humo
"""
import argparse

import yaml

import config
from common.acmeai import KP_CANONICOS


def escribir_yaml() -> str:
    d = config.DATA / "pose"
    assert (d / "images" / "train").exists(), "falta data/pose: correr 06a_preparar_pose.py y copiar data/"
    cfg = {
        "path": d.as_posix(), "train": "images/train", "val": "images/val",
        "kpt_shape": [len(KP_CANONICOS), 3],
        "flip_idx": list(range(len(KP_CANONICOS))),  # ningún keypoint tiene par izq/der: el espejado no los permuta
        "names": {0: "bovino"},
    }
    ruta = d / "pose.yaml"
    ruta.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(ruta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="yolo11s-pose.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--fraccion", type=float, default=1.0, help="fracción del train a usar (pruebas rápidas)")
    ap.add_argument("--nombre", default="pose9")
    ap.add_argument("--device", default="auto", help="auto | cpu | 0")
    ap.add_argument("--sin-amp", action="store_true", help="desactivar precisión mixta si falla en ROCm")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = a.device if a.device != "auto" else (0 if torch.cuda.is_available() else "cpu")
    print(f"device={device} | torch {torch.__version__} | hip={getattr(torch.version, 'hip', None)}")

    YOLO(a.modelo).train(
        data=escribir_yaml(), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
        patience=a.patience, fraction=a.fraccion, device=device, amp=not a.sin_amp,
        project=str(config.RUNS / "pose"), name=a.nombre, exist_ok=True, seed=config.SEED, plots=True,
    )
    print(f"pesos: {config.RUNS / 'pose' / a.nombre / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
