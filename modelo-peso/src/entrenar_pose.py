"""Entrena YOLO-pose para localizar los 7 landmarks anatomicos.

Con 72 imagenes hay que apoyarse fuerte en el modelo preentrenado y en la
augmentacion; por eso arrancamos del checkpoint de pose de COCO y congelamos
el backbone las primeras epocas.

Uso:  python src/entrenar_pose.py
"""
import os
import sys
from pathlib import Path

from ultralytics import YOLO

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))


def main():
    # ultralytics resuelve el 'path' del yaml contra el cwd del proceso, no
    # contra la ubicacion del yaml. Nos paramos en modelo-peso/ para que
    # 'path: data/pose' del yaml apunte siempre al mismo lugar, sin importar
    # desde donde se invoque este script.
    os.chdir(AQUI.parent)

    modelo = YOLO("yolo11n-pose.pt")
    modelo.train(
        data=str(AQUI / "pose_dataset.yaml"),
        epochs=300,
        imgsz=1024,
        batch=4,
        patience=60,
        freeze=10,          # backbone congelado: 72 imagenes no alcanzan para moverlo
        fliplr=0.5,         # valido: flip_idx es identidad
        degrees=8,
        translate=0.12,
        scale=0.35,         # simula la variacion de distancia de captura
        shear=3,
        hsv_v=0.4,          # sol fuerte y sombra dura, como en el campo
        mosaic=0.0,         # mosaico rompe la geometria del animal, no sirve aca
        project=str(AQUI.parent / "runs"),
        name="pose7",
        device="cpu",
    )
    print("\nPesos en runs/pose7/weights/best.pt")


if __name__ == "__main__":
    main()
