#!/usr/bin/env bash
# Fase 3/4c: atacar la SATURACIÓN del modelo en el rango alto de peso.
#
# Diagnóstico que motiva esta corrida (medido en la Fase 1):
#   la CNN entrenada solo con AcmeAI nunca predice fuera de 80-267 kg, aunque el real llega a 465 kg.
#   Subestima 31% a los animales de más de 300 kg y falla por completo en Mendeley (341-644 kg).
#   Causa: el entrenamiento tiene 1.394 imágenes entre 150-200 kg y solo 22 por encima de 300.
#
# Dos remedios, por separado y combinados:
#   --balancear : muestrea cada banda de peso con igual probabilidad (no agrega datos, los reparte mejor)
#   --unificado : suma CID (150-816 kg) y Mendeley (341-644 kg). El rango 300+ pasa de 22 a 127 imágenes.
#
# Referencia a batir: convnext_tiny_crop_384_mse = 15,50% en test de AcmeAI (14,56% en val).
#
#   export HSA_OVERRIDE_GFX_VERSION=10.3.0
#   bash correr_ablacion2.sh 2>&1 | tee runs/ablacion2.log
#
# IMPORTANTE antes de correr: si data/tablas/manifest_unificado.parquet no existe, correr primero
#   uv run python 29_dataset_unificado.py && uv run python 29a_tapar_vara.py
#
# Sobre la vara de referencia de CID: ya se tapó por inpainting (29a_tapar_vara.py), igual que el sticker, para
# que la CNN no pueda usarla como referencia de escala. Aun así, CID son fotos de una estación fija con fondo
# constante (barandas, plataforma), que da algo de escala implícita imposible de borrar sin arruinar la imagen.
# Por eso el número que vale para nuestro objetivo sigue siendo el MAPE del origen 'acmeai' (fotos de campo),
# que el script imprime desglosado por dataset.
set -u
BATCH="${BATCH:-16}"
correr () {
  echo "=============== $* ==============="
  uv run --extra rocm python 20_train_cnn.py --backbone convnext_tiny --tam 384 --encuadre crop --loss mse \
      --batch "$BATCH" "$@" || echo "!! falló: $*"
}

# 1) Solo balanceo (mismos datos, mejor repartidos)
correr --epocas 25 --balancear

# 2) Solo datos nuevos
correr --epocas 25 --unificado

# 3) Las dos cosas juntas (la apuesta principal)
correr --epocas 30 --unificado --balancear

# 4) Igual que 3 pero con más épocas, por si con más datos el modelo aún no convergió
correr --epocas 45 --unificado --balancear --nombre convnext_unif_bal_45ep

echo
echo "===== resumen ====="
column -s, -t runs/cnn/resumen.csv 2>/dev/null || cat runs/cnn/resumen.csv
echo
echo "Qué mirar:"
echo "  - mape_val de cada corrida (así se elige, nunca por test)"
echo "  - el desglose 'MAPE por dataset de origen': la fila 'acmeai' es la comparable con 15,50%"
echo "  - 'rango de predicciones en test': si sigue topando en ~267 kg, la saturación no se resolvió"
