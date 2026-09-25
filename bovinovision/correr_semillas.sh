#!/usr/bin/env bash
# Repite la MEJOR configuración con distintas semillas, para poder reportar el resultado con un intervalo
# de confianza en vez de un número suelto ("15,1% ± algo") y saber si las diferencias entre configuraciones
# son reales o ruido.
#
# Qué varía: la inicialización de la cabeza de regresión, el orden de los lotes y el augmentation.
# Qué NO varía: el split train/val/test, que queda fijo. Su incertidumbre ya se mide aparte por bootstrap
# (uv run python 30_intervalos.py), remuestreando animales del test. Se deja fijo a propósito: el modelo de
# pose (YOLO) se entrenó con este split, así que cambiarlo metería animales "ya vistos" por el detector en el
# conjunto de evaluación.
#
#   export HSA_OVERRIDE_GFX_VERSION=10.3.0
#   bash correr_semillas.sh 2>&1 | tee runs/semillas.log
#
# Son 8 corridas (~10 h). Para acortar, correr solo el primer bloque (4 corridas, ~5 h).
# Al terminar:  uv run python 31_resumen_semillas.py
set -u
BATCH="${BATCH:-16}"
correr () {
  echo "=============== $* ==============="
  uv run --extra rocm python 20_train_cnn.py --backbone convnext_tiny --tam 384 --encuadre crop --loss mse \
      --epocas 25 --batch "$BATCH" "$@" || echo "!! falló: $*"
}

# Bloque 1: la mejor configuración (sin balanceo), 4 semillas
for s in 1 2 3 4; do correr --semilla "$s"; done

# Bloque 2: la variante con balanceo, 4 semillas (para ver si la mejora del balanceo es real o ruido)
for s in 1 2 3 4; do correr --balancear --semilla "$s"; done

echo
echo "===== listo. Ahora: uv run python 31_resumen_semillas.py ====="
