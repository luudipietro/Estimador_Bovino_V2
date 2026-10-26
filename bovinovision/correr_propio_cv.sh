#!/usr/bin/env bash
# Fine-tuning de la CNN sobre el dataset propio con validación cruzada por animal (46_propio_cv.py).
# ADITIVO: no pisa runs/cnn/, ni estimador/modelos/, ni nada anterior. Todo va a runs/propio_cv/<nombre>/.
#
#   export HSA_OVERRIDE_GFX_VERSION=10.3.0
#   bash correr_propio_cv.sh 2>&1 | tee runs/propio_cv.log
#
# Qué corre (cada una = 5 folds sobre los 141 animales, ~5-10 min en la RX 6700 XT):
#   1. Bloque A: arranca del modelo actual (Bangladesh), 3 semillas  -> la corrida principal
#   2. Bloque B: arranca de ImageNet, 3 semillas                      -> ¿sirve haber preentrenado en Bangladesh?
#   3. Bloque C: Bangladesh con backbone congelado (solo cabeza)       -> control: ¿hace falta reentrenar todo?
# Al terminar:  uv run python 47_propio_resumen_cv.py
# El MODELO FINAL (todos los animales) se entrena aparte, solo si el resultado lo justifica:
#   uv run --extra rocm python 46_propio_cv.py --init bangladesh --semilla 1 --solo-final --nombre final_bangladesh
set -u
BATCH="${BATCH:-16}"
correr () {
  echo "=============== $* ==============="
  uv run --extra rocm python 46_propio_cv.py --batch "$BATCH" "$@" || echo "!! falló: $*"
}

for s in 1 2 3; do correr --init bangladesh --semilla "$s"; done
for s in 1 2 3; do correr --init imagenet --semilla "$s"; done
correr --init bangladesh --congelar --semilla 1

echo
echo "===== listo. Ahora: uv run python 47_propio_resumen_cv.py ====="
