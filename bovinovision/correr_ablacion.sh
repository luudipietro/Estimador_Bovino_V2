#!/usr/bin/env bash
# Fase 1 del plan: ablación completa de la CNN sin marcador. Pensado para Fedora con la GPU AMD.
#
#   export HSA_OVERRIDE_GFX_VERSION=10.3.0
#   bash correr_ablacion.sh 2>&1 | tee runs/ablacion.log
#
# Cada corrida agrega una fila a runs/cnn/resumen.csv. Si una corrida se cae (por VRAM, por ejemplo),
# las siguientes igual se ejecutan. Para bajar memoria: exportar BATCH=8 antes de correr.
set -u
BATCH="${BATCH:-16}"
EPOCAS="${EPOCAS:-25}"
correr () {
  echo "=============== $* ==============="
  uv run --extra rocm python 20_train_cnn.py --epocas "$EPOCAS" --batch "$BATCH" "$@" || echo "!! falló: $*"
}

# 1) Encuadre (la pregunta más importante: ¿recortar al animal o dejar el contexto?)
correr --backbone convnext_tiny --tam 384 --encuadre completa
correr --backbone convnext_tiny --tam 384 --encuadre crop
correr --backbone convnext_tiny --tam 384 --encuadre crop_mascara

# 2) Resolución (con el mejor encuadre de arriba; por defecto crop)
correr --backbone convnext_tiny --tam 224 --encuadre crop

# 3) Backbone
correr --backbone tf_efficientnetv2_s --tam 384 --encuadre crop
correr --backbone resnet18 --tam 384 --encuadre crop --congelar   # control: es el Modelo A viejo, debería ser el peor

# 4) Augmentation: ¿conviene invarianza a escala o conservar el tamaño aparente?
correr --backbone convnext_tiny --tam 384 --encuadre crop --sin-aug-escala

# 5) Loss
correr --backbone convnext_tiny --tam 384 --encuadre crop --loss mse

# 6) CONTROL CIENTÍFICO: mismas condiciones pero con el sticker visible.
#    La diferencia contra la corrida equivalente sin sticker mide cuánto aportaba realmente el marcador.
correr --backbone convnext_tiny --tam 384 --encuadre crop --imagenes con_sticker

echo
echo "===== resumen ====="
column -s, -t runs/cnn/resumen.csv 2>/dev/null || cat runs/cnn/resumen.csv
