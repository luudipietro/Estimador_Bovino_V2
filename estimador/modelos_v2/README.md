# Modelo v2 (ajustado con el dataset propio)

Lo que viaja en git acá es solo `config.json` y `SHA256SUMS.txt`. **Los `.pt` no están en el repositorio** (el de pesos pesa
111 MB y GitHub rechaza más de 100 MB): se pasan aparte y se verifican con los hashes de abajo.

| Archivo | En git | Qué es |
|---|---|---|
| `config.json` | sí | versión, **rango válido 292–485 kg**, nombres de los archivos |
| `SHA256SUMS.txt` | sí | huellas de los 3 `.pt` (el servicio **no arranca** si alguna no coincide) |
| `peso_convnext_propio.pt` | **no** (111 MB) | la CNN ajustada con 141 vacas. **Es lo único nuevo.** |
| `pose9.pt` | no | el mismo del v1 (`modelos/`), no cambió |
| `detector_coco.pt` | no | el mismo del v1 (`modelos/`), no cambió |

`pose9.pt` y `detector_coco.pt` se bajan con `python ../estimador/descargar_modelos.py` (carpeta `modelos/`) y se copian acá.
Solo hay que conseguir `peso_convnext_propio.pt`.

## Cómo dejarlo andando

1. Poner los 3 `.pt` en esta carpeta.
2. Verificar (cualquiera de las dos):
   ```bash
   cd estimador/modelos_v2 && sha256sum -c SHA256SUMS.txt          # Linux / Git Bash
   Get-FileHash peso_convnext_propio.pt -Algorithm SHA256           # PowerShell: comparar con SHA256SUMS.txt
   ```
3. Apuntar el servicio a esta carpeta y arrancarlo (ver `servicio-estimacion/README.md`):
   ```bash
   export CARPETA_MODELOS=../estimador/modelos_v2
   export RESOLUCION_MINIMA_MP=0.5      # las fotos de 1024 px tienen ~0,8 MP y el valor por defecto (3 MP) las rechaza
   ```

El adaptador del servicio lee la versión y el rango de `config.json`: **cambiar de v1 a v2 no requiere tocar código**.

## Qué es y qué NO es

- Partió del modelo de Bangladesh (`modelos/`) y se ajustó con 141 hembras de 292–485 kg de un establecimiento.
- Validación cruzada de 5 folds por animal: **7,2 %** de error contra **8,3 %** de predecir siempre la mediana (mejora de
  1,2 puntos, IC95 +0,4 a +1,9). Los números y cómo se midieron están en `bovinovision/runs/propio_cv/`.
- **No extrapola:** fuera de 292–485 kg no es confiable. En una jornada aparte (6/10, con toros de hasta 544 kg) dio 23,7 %
  de error contra 17,5 % de predecir siempre un peso fijo. El servicio marca `confiable: false` con `fuera_de_rango` (R-20).
- El v1 (`modelos/`) declara 36–621 kg pero no pasa de ~270, así que sobre vacas argentinas da pesos equivocados **marcados
  como confiables**. No usarlo para demostrar nada con animales locales.
