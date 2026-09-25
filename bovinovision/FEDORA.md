# Correr bovinovision en Fedora (GPU AMD RX 6700 XT)

Todo lo que necesita GPU se corre en Fedora: entrenamiento de YOLO-pose (`06`), su evaluación (`07`) y la
profundidad monocular (`10`). El resto (manifest, features, baselines) ya está hecho en Windows y sus tablas
viajan dentro de `data/`.

## 1. Nada que copiar: dual boot con el proyecto en el disco D

Windows y Fedora están en dual boot y el proyecto vive en el disco `D:` (NTFS), compartido entre ambos. No hace
falta copiar nada: se monta esa partición desde Fedora y se trabaja directo ahí. Los resultados que se generen
(pesos entrenados, tablas nuevas) van a estar disponibles al volver a Windows sin ningún paso extra.

```bash
lsblk -f                                  # identificar la partición NTFS del disco D (tamaño la delata)
sudo mkdir -p /mnt/D
sudo mount -t ntfs3 /dev/DISPOSITIVO /mnt/D    # reemplazar DISPOSITIVO por lo que muestre lsblk
ls "/mnt/D/claude_code/estimador_peso_bovino/bovinovision"   # debería listar config.py, common/, etc.
```

Para que quede montada en cada arranque (opcional):
```bash
sudo blkid /dev/DISPOSITIVO   # copiar el UUID
echo 'UUID=TU-UUID-ACA  /mnt/D  ntfs3  defaults,uid=1000,gid=1000,windows_names  0  0' | sudo tee -a /etc/fstab
sudo mount -a                 # no debería dar error
```

El dataset AcmeAI ya está al lado de `bovinovision/` en esa misma partición
(`/mnt/D/claude_code/estimador_peso_bovino/www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/`), así que
`config.py` lo encuentra por ruta relativa sin tocar `ACMEAI_DIR`.

**Nota de rendimiento:** `ntfs3` anda bien para leer/escribir, pero el dataloader de `06_train_pose.py` lee miles
de archivos chicos y puede ir más lento que en ext4 nativo. Probar primero directo en `/mnt/D`; si resulta muy
lento, copiar solo `data/pose/` a una carpeta ext4 del home y apuntar el `.yaml` ahí.

## 2. Entorno (una vez)
```bash
sudo dnf install -y python3.13 git             # ROCm wheels verificadas para cp313
curl -LsSf https://astral.sh/uv/install.sh | sh
cd "/mnt/D/claude_code/estimador_peso_bovino/bovinovision"
uv python install 3.13
uv sync --extra rocm --python 3.13             # torch 2.14 + ROCm 7.2 (índice download.pytorch.org/whl/rocm7.2)
```
Tu usuario debe estar en los grupos `video` y `render` para acceder a la GPU:
```bash
sudo usermod -aG video,render $USER            # cerrar sesión y volver a entrar
```

## 3. Verificar la GPU (primer paso, antes de entrenar nada)
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0         # gfx1031 (6700 XT) -> gfx1030, que sí trae kernels
uv run python 00_check_gpu.py
```
Debe mostrar `hip=7.x`, el nombre de la GPU y `backend=cuda` (así se expone ROCm en PyTorch), y las 5 matmul bastante
más rápidas que en CPU (en Windows/CPU dieron 1,5 s). Si dice `Sin GPU visible`: revisar grupos `video/render`,
`rocminfo` y que el override esté exportado en la MISMA terminal.

Para no repetir el export: `echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc`.

## 4. Entrenar YOLO-pose
```bash
uv run python 06_train_pose.py --modelo yolo11s-pose.pt --epochs 150 --imgsz 640 --batch 16 --nombre pose9
# si falla por precisión mixta (AMP) en ROCm:   añadir  --sin-amp
# si se queda sin VRAM (12 GB):                 bajar --batch a 8
```
Los pesos quedan en `runs/pose/pose9/weights/best.pt`. Después:
```bash
uv run python 07_eval_pose.py --pesos runs/pose/pose9/weights/best.pt
```
Imprime error por keypoint (% del largo corporal, PCK@5%) en val y test, y escribe `data/tablas/keypoints_pred.parquet`.

## 5. Profundidad monocular (puerta de decisión)
```bash
uv run python 10_depth_run.py --modelo pro     --n 300 --fp16   # Depth Pro (métrica + focal), ~5 GB de VRAM en fp16
uv run python 10_depth_run.py --modelo dav2l   --n 300          # Depth Anything V2 metric (outdoor, large)
uv run python 10_depth_run.py --modelo dav2l-in --n 300         # variante indoor, por si outdoor sigue sesgada
```
Los pesos se descargan de Hugging Face la primera vez (Depth Pro ≈ 1,9 GB; requiere internet).
Licencia de Depth Pro: solo investigación / no comercial.

## 5-bis. Fase 1 del plan sin sticker: ablación de la CNN (lo que hay que correr ahora)

**Antes**: tiene que existir `data/pose/images_sin_sticker/` (4.540 imágenes con el marcador borrado por
inpainting). Si no está, generarla con `uv run python 20a_tapar_sticker.py` (~1 h, solo CPU).
Sin ese paso el experimento no vale: la CNN puede aprender a usar el sticker como referencia de escala.

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
bash correr_ablacion.sh 2>&1 | tee runs/ablacion.log     # ~9 corridas; si falta VRAM: BATCH=8 bash correr_ablacion.sh
```
Cada corrida agrega una fila a `runs/cnn/resumen.csv` (MAPE en test con TTA, más el MAPE de validación con el que
se eligió el modelo). Una corrida suelta, para probar:
```bash
uv run python 20_train_cnn.py --backbone convnext_tiny --tam 384 --encuadre crop --epocas 25
```
Qué mirar al terminar: la fila con menor `mape_val` (no elegir por test), y la diferencia contra la corrida
`--imagenes con_sticker`, que es cuánto aportaba realmente el marcador.

## 5-ter. Fase 3/4c: atacar la saturación en el rango alto (lo que hay que correr AHORA)

La Fase 1 ya cumplió el objetivo sin marcador (15,50% en test; ensemble 15,27%). El problema que queda es que el
modelo **nunca predice fuera de 80–267 kg**: subestima 31% a los animales de más de 300 kg, porque el
entrenamiento tiene 1.394 imágenes en 150–200 kg y solo 22 por encima de 300.

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
uv run python 29_dataset_unificado.py          # solo si falta data/tablas/manifest_unificado.parquet
uv run python 29a_tapar_vara.py                # idem: tapa la vara graduada de CID
bash correr_ablacion2.sh 2>&1 | tee runs/ablacion2.log
```
(En este disco ya están hechos los dos pasos previos: el manifest apunta a `extra_sin_vara`.)
Son 4 corridas (~6 h): balanceo solo, datos nuevos solos, las dos juntas, y las dos juntas con más épocas.
El dataset unificado suma CID (150–816 kg) y Mendeley (341–644 kg): el rango 300+ pasa de 22 a **127** imágenes.

Qué mirar al terminar:
- `mape_val` de cada corrida (así se elige el modelo, nunca por test).
- El desglose **MAPE por dataset de origen**: la fila `acmeai` es la comparable con el 15,50% actual.
- **`rango de predicciones en test`**: si sigue topando cerca de 267 kg, la saturación no se resolvió.
- Aviso: las fotos de CID traen una vara graduada en la escena, así que su MAPE puede ser optimista. El número
  que vale para nuestro objetivo es el de `acmeai`.

## 5-quater. Semillas: para reportar con intervalo de confianza

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
bash correr_semillas.sh 2>&1 | tee runs/semillas.log     # 8 corridas, ~10 h (o solo el bloque 1: 4 corridas, ~5 h)
```
Repite la mejor configuración con 4 semillas, y la variante con balanceo con otras 4. Sirve para dos cosas:
saber cuánto varía el resultado por el puro azar del entrenamiento, y si la mejora del balanceo
(15,34% vs 15,50%) es real o ruido.

El split **no** cambia entre semillas, a propósito: el modelo de pose se entrenó con ese split, así que
moverlo metería animales ya vistos por el detector en la evaluación. La incertidumbre del split se mide aparte,
por bootstrap, y no necesita GPU (`30_intervalos.py`, ya corrido: IC 95% de 14,0 a 16,3).

Al terminar, en cualquiera de los dos sistemas: `uv run --extra cpu python 31_resumen_semillas.py`
(en Fedora, `--extra rocm`), que combina ambas fuentes de incertidumbre y hace el test estadístico.

## 6. Volver a Windows
No hay que copiar nada: al reiniciar a Windows, `runs/pose/pose9/weights/best.pt`, `data/tablas/keypoints_pred.parquet`
y `data/tablas/depth_*.parquet` ya están en `D:\claude_code\estimador_peso_bovino\bovinovision\`, porque es la misma
partición. Con eso las etapas siguientes (`08_features_2d.py --kp keypoints_pred`, baselines, gate de profundidad)
corren en Windows sin GPU.
