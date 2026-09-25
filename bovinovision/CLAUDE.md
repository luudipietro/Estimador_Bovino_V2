# bovinovision — contexto de trabajo (actualizado 22/09/2026)

> **CAMBIO DE RUMBO (22/09/2026): se saca el sticker.** Los productores dijeron que pegarle un marcador a cada
> animal no es viable en campo. Se acepta menos precisión a cambio de simplicidad: **objetivo MAPE 15–19% SIN
> marcador** (el número exacto lo define el usuario). Todo lo de abajo sobre el sticker queda como evidencia
> histórica: sirve para justificar por qué el sticker ayudaba, no como diseño vigente.
> Plan completo: `C:\Users\Luciano\.claude\plans\estuve-pensando-en-un-rustling-thunder.md`.
>
> **Estado sin sticker**: Fase 0 lista → **17,77% MAPE en test** (fusión A + C'), banda 100–200 kg **14,11%**.
> Falta la Fase 1 (CNN fine-tuneada, en Fedora) que es donde está el salto grande.
>
> Decisiones de producto tomadas: la app **pedirá categoría + raza + edad/dentición**, **registrará los sensores
> del teléfono y el EXIF** en cada captura, y el cómputo pesado va en Fedora con GPU.
>
> **Trampa crítica ya resuelta**: el sticker está pegado al animal en casi todas las fotos de AcmeAI. Entrenar una
> CNN sobre las imágenes tal cual permitiría que aprenda a usarlo como referencia de escala, y el MAPE no valdría
> para el escenario real. Por eso `20a_tapar_sticker.py` genera `data/pose/images_sin_sticker/` (inpainting con
> dilatación aleatoria, para que el tamaño de la zona tapada tampoco delate la escala) y **todo entrenamiento usa
> esas imágenes por defecto**. La variante `--imagenes con_sticker` queda solo como control científico.

---


Reescritura desde cero del estimador de peso bovino (Módulo 2 de BovinoVision, UTN-FRT). Reemplaza la investigación
de `../modelo-peso/` (que queda intacto como evidencia). Responder siempre en español.

## Objetivo y diseño
Diseño de referencia: `../Plan_estimacion_peso_bovino_vision_celular.docx` (video corto de celular → mejor frame →
segmentación + keypoints + profundidad monocular → medidas métricas → regresión → peso + "no sé"). Hipótesis a probar:
la escala se puede recuperar **sin sticker en campo** usando profundidad métrica. Meta del acta: MAPE < 15%.
Restricción: un solo celular RGB, animales libres. Sin hardware propio.

Plan aprobado (etapas 0–15, un script por etapa, cada una con su prueba): ver
`C:\Users\Luciano\.claude\plans\estuve-pensando-en-un-rustling-thunder.md` (en Windows) — resumen de lo que importa abajo.

## Dataset base: AcmeAI/BMGF (Bangladesh)
`../www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/` (o `ACMEAI_DIR`). Lo usamos solo lotes **B3+B4 vista lateral**:
4.544 imágenes, 9 keypoints, máscaras con sticker, peso de báscula en el nombre. Otros datasets evaluados: Mendeley 72
bovinos (ya local, sirve para validar medidas en cm) y **CowDatabase2** (GitHub `ruchaya/CowDatabase2`, RGB-D real + peso;
único que permitiría medir el error de profundidad contra profundidad real; licencia/descarga sin verificar).

### Trampas de los datos (ya manejadas en `common/acmeai.py`; NO volver a caer)
1. **Orden de keypoints distinto entre B3 y B4** → siempre mapear por nombre (`kp_por_nombre`), nunca por posición.
2. **B3 guarda keypoints en la resolución original de la foto**, no en la de la imagen entregada (1900 px) → se reescala por
   imagen con `ancho_real / ancho_json` (`04_keypoints_gt.py`). B4 no necesita reescalado.
3. **El "id" del nombre NO identifica un animal único** (mismo id con varios pesos; 514 ids en B3 con ~5 fotos c/u). Se agrupa
   por id numérico entre lotes (conservador). Diagnóstico: CV por imagen 18,63% vs por animal 18,53% → sin fuga aparente.
4. B2 queda fuera de la v1 (esquemas de 6 y 23 keypoints, nombres irregulares).
5. Máscaras RGBA de 3 colores: fondo (0,255,193), sticker (0,117,255), vaca (255,30,249); aparece negro (0,0,0) = sin etiquetar
   (se trata como fondo). 6 imágenes sin sticker; 28 con sticker recortado (redondez < 0,9).
6. EXIF: solo 71% trae focal equivalente 35 mm (26–28 mm). El PDF de Acme NO da el tamaño real del sticker (Depth Pro sugiere ~10 cm).
7. Split fijo por animal (seed 42): train 360 / val 77 / test 77 animales (3194/680/670 imágenes).

## Resultados medidos hasta ahora (test, 670 imágenes)
Con keypoints anotados a mano (`features_gt`, techo de la etapa de regresión):
- Mediana (baseline): **23,5%** MAPE (este dataset sí discrimina; en Mendeley la mediana ya daba 14,9%). Por banda: 0–100 kg 87%, 100–200 16%.
- Sin escala (ratios, "Modelo B"): 19,7%. Distancias px crudas: 19,6% (casi igual → la distancia de captura varía poco en este dataset).
- Con sticker (distancias en diámetros de sticker): **16,6%** HGB (16,1% con ratios agregados). Por banda: 100–200 kg 13,5% (cumple), 0–100 kg 44%, 200–300 18%, >300 solo 7 imágenes.
- Reproduce ~15,8% del trabajo viejo (no exacto: otro split/conjunto).
- Aun con escala perfecta el MAPE global queda en ~16% > 15%: el cuello de botella no es solo la escala (terneros, calidad de keypoints).

**Con keypoints PREDICHOS por YOLO-pose** (`features_pred`, el número honesto — ver estado de pose abajo): prácticamente
idéntico al techo GT. Sin sticker: 19,15–19,85% (ridge/hgb). Con sticker: 16,51–16,83%. Por banda (hgb, con sticker):
0–100 kg 44,5%, 100–200 kg **13,7%** (cumple), 200–300 kg 18,8%, >300 kg 24,3% (n=7). CV5 por animal 18,99% (sin fuga
relevante). **Conclusión: el modelo de pose entrenado NO es el cuello de botella** — casi no degrada el MAPE respecto
de keypoints anotados a mano.

### Pose entrenado (`06_train_pose.py`, YOLO11s-pose, 114 épocas, GPU AMD en Fedora, ROCm)
Convergencia limpia (train y val loss bajan juntas, sin señal de sobreajuste), mAP50-95(pose) ≈ 0,85.
Error por keypoint en test (`runs/eval_pose_test.csv`), en % del largo corporal (wither→pinbone):
front_girth_bottom 2,0% (mejor) ... height_bottom 3,7% (peor); PCK@5% entre 71% (height_bottom) y 94% (front_girth_top).
0 imágenes sin detección en las 4.540. Muy por debajo del ruido que ya tolera el modelo de regresión (~5%, ver abajo) →
consistente con que predichos ≈ GT en el resultado final.

### Profundidad: Depth Pro corrida real (300 imágenes, GPU Fedora) — `11_depth_scale_check.py`
**LA PUERTA NO PASA.** Error de escala del sticker (constante real, tamaño estimado ~10,3 cm — plausible): mediana
**7,9%**, P90 **18,2%** (umbral era mediana ≤3,5%, P90 ≤8%). Por lote: B3 mediana 7,1%/P90 16,1%, B4 mediana 9,6%/P90 23,0%
(algo peor, sin explicar por qué). Según la tabla de sensibilidad (`10a`), σ≈8% cae entre "aporta un punto sobre no usar
escala" (σ=5–10%, entrenando con el mismo ruido) y "es peor que no usar escala" (si se entrena limpio y se despliega con
este ruido, como sería el uso real de un modelo de profundidad general no afinado). **Diagnóstico secundario también
malo**: la profundidad varía 21% (mediana) entre los 9 keypoints de una misma imagen, a pesar de que en vista lateral el
animal es casi plano — el mapa de Depth Pro no es solo un error de escala global, también es ruidoso punto a punto (el
umbral de la puerta ya asumía esto como optimista, y se confirma).
**Decisión: el sticker sigue siendo requisito para el sistema.** Congruente con la nota del 15/09 en `../CLAUDE.md` del
proyecto (TP5): NO hay que revertir esa decisión. Falta probar DAV2 (large/indoor) por si alguna variante da mejor
escala antes de cerrar el tema del todo, pero con Depth Pro (el candidato con mejor prior, porque estima su propia focal)
ya no alcanza.

### Sensibilidad a error de escala (`10a_sensibilidad_escala.py`, `runs/sensibilidad_escala.csv`)
Error de escala global por imagen ~N(0,σ). Entrenando con el mismo ruido: σ=5% → 17,7%; 10% → 18,6%; 20% → 19,5%; 30% → 19,8% (converge al Modelo B).
Entrenando limpio y desplegando ruidoso: σ=10% → 22,3% (peor que no usar escala). **Implicancia: la escala de profundidad debe entrar al entrenamiento.**
**Umbral propuesto de la puerta (etapa 8):** seguir sin marcador solo si el error de escala de la profundidad tiene **mediana ≤ ~3,5% y P90 ≤ ~8%**
(σ≈5%); entre 5% y 10% el sticker sigue valiendo la pena; >10% la profundidad no sirve como fuente de escala.
Limitación: simula error global; la profundidad real también deforma proporciones entre puntos (umbral optimista).

## Estado de scripts (todos en este directorio; `uv run python <script>`) — TODO el plan v1 (etapas 0–15) está escrito y corrido
| Script | Estado |
|---|---|
| `config.py`, `common/{metrics,splits,geometry,acmeai,masks,features}.py`, `tests/` | listos, 16 tests en verde (`uv run pytest -q`) |
| `00_check_gpu.py` | corrido en Fedora, ROCm reconoce la RX 6700 XT (`HSA_OVERRIDE_GFX_VERSION=10.3.0`, ver `FEDORA.md`) |
| `01_manifest.py`, `02_split.py`, `03_eda.py`, `04_keypoints_gt.py`, `05_mascaras.py` | corridos → `data/tablas/{manifest,keypoints_gt,mascaras}.parquet` |
| `06a_preparar_pose.py`, `06_train_pose.py`, `07_eval_pose.py` | corridos en Fedora → `runs/pose/pose9/weights/best.pt`, `data/tablas/keypoints_pred.parquet` |
| `08_features_2d.py`, `09_baselines.py` | aceptan `--kp`/`--features`; corridos con GT y con pred → `features_{gt,pred}.parquet` |
| `10a_sensibilidad_escala.py`, `10_depth_run.py` (Depth Pro), `11_depth_scale_check.py` | corridos — puerta de profundidad NO pasa (ver arriba) |
| `13_model_C.py` | corrido → `runs/model_c.csv`, `runs/models/model_c.joblib`, `data/tablas/pred_modelo_c.parquet` |
| `14_model_A.py` | corrido (embeddings ResNet18 congelado, cacheados en `data/cache/embeddings_resnet18.parquet`) → `runs/model_a.csv`, `runs/models/model_a.joblib` |
| `15_fusion.py` | corrido (stacking Ridge sobre predicciones OOF de A y C) → `runs/fusion.csv`, `runs/models/fusion_stacker.joblib`, `data/tablas/predicciones_finales.parquet` |
| `16_calidad.py` | corrido → `runs/calidad_mape_vs_cobertura.csv` (ver limitación: score muy concentrado, no discrimina bien en este dataset) |
| `17_predict.py` | pipeline imagen→peso end-to-end, probado sobre una imagen del dataset (falta segmentador propio para fotos nuevas, ver nota en el script) |
| `18_reporte.py` | corrido → `runs/reporte_final_{global,por_banda}.csv`, `runs/reporte_final_por_banda.png` (gráfico para el hito) |
| `12_measures_metric.py` (medidas en cm reales) | **no escrito**: el PDF de AcmeAI no documenta el diámetro físico del sticker, así que no hay forma confiable de convertir a cm absolutos con este dataset. Se usaron unidades relativas al sticker en su lugar (ya es una escala real, ver Modelo C) |
| **`20a_tapar_sticker.py`** (sin sticker) | escrito; genera `data/pose/images_sin_sticker/` (~1 h en CPU). **Correr antes que cualquier CNN** |
| **`20_train_cnn.py`** (sin sticker, Fase 1) | escrito y probado en CPU con los 3 encuadres; **falta la corrida real en Fedora**. Flags: `--encuadre {completa,crop,crop_mascara}`, `--tam`, `--backbone`, `--loss`, `--sin-aug-escala`, `--congelar`, `--imagenes {sin_sticker,con_sticker}` |
| **`correr_ablacion.sh`** | las 9 corridas de la Fase 1 en un solo comando (ver `FEDORA.md`) |
| **`23_mendeley_manifest.py`** | corrido → `manifest_mendeley.parquet` (72 animales, 341–644 kg, con medidas en cm) |
| **`24_cross_mendeley.py`** | corrido → generalización a otro país/raza/rango (ver resultados arriba) |
| Datasets externos bajados | `../datasets_externos/cid_dataset.csv` (513 animales con raza/edad/altura/peso) y `cid_images.tar.gz` (833 MB, 2.052 fotos reales + frames de YouTube) — **sin descomprimir todavía** |

**Sobre CattleNet-XAI (el paper que reporta 6,2% sin marcador)**: usa justamente CID. De sus 17.864 imágenes,
**15.812 son frames de YouTube** del mismo animal (hasta 30 por animal) y solo 2.052 son fotos reales. Si el split
fue por imagen y no por animal, su número está inflado por fuga. No usarlo como referencia sin esa aclaración.

## RESULTADO PRINCIPAL (23/09/2026): la CNN sin marcador cumple el objetivo
Fase 1 corrida en Fedora (8 corridas válidas, `runs/cnn/resumen.csv`, ablación en `runs/ablacion.log`).
Modelos elegidos por MAPE de **validación**; se reporta test con TTA (espejado).

| Configuración | MAPE val | MAPE test |
|---|---|---|
| ConvNeXt-Tiny, crop 384, MSE | 14,56% | **15,50%** |
| ConvNeXt-Tiny, crop 384, L1 sin aug de escala | 14,63% | 15,86% |
| ConvNeXt-Tiny, crop 384, L1 | 14,96% | 15,88% |
| ConvNeXt-Tiny, **imagen completa** 384, L1 | 15,10% | 16,45% |
| ConvNeXt-Tiny, crop **224**, L1 | 15,51% | 17,07% |
| EfficientNetV2-S, crop 384, L1 | 19,51% | 19,25% |
| ResNet18 **congelado**, crop 384 | 30,10% | 29,39% |
| *(control)* ConvNeXt-Tiny crop 384 L1 **CON sticker visible** | 13,90% | 15,19% |
| **Ensemble de las 3 mejores sin sticker** | 14,15% | **15,27%** |

Conclusiones de la ablación:
- **Fine-tuning es todo**: congelado 29,4% vs fine-tuneado 15,9% (el congelado ni siquiera gana a la mediana, 23,5%).
- **Recortar al animal ayuda** (~0,6 pts) y **la resolución importa** (384 vs 224: ~1,2 pts).
- **El augmentation de escala da igual** (15,86 vs 15,88): coherente con que el encuadre ya borró la señal de tamaño.
- **ConvNeXt-Tiny >> EfficientNetV2-S** con los mismos hiperparámetros (este último quizá necesitaba otro LR).
- **EL STICKER AHORA VALE SOLO 0,30 puntos** (15,19% con él vs 15,50% sin él, misma configuración). Antes, con
  features geométricas, valía 2 puntos. **La CNN recupera casi todo lo que aportaba el marcador.**
- `crop_mascara` (máscara como 4º canal) **falló en ROCm** con `miopenStatusUnknownError` (convolución de 4
  canales); iba en 15,6% de val cuando se cortó. Pendiente si se quiere, no es crítico.

**Comparación con el rumbo anterior**: el pipeline completo CON sticker daba 15,72%. La CNN sin sticker da 15,27%.
**Ya somos mejores sin marcador que el mejor resultado con marcador.** Objetivo 15–19% del usuario: cumplido.

### Reporte consolidado (`28_reporte_sin_sticker.py` → `runs/reporte_sin_sticker{_global.csv,_por_banda.csv,.png}`)
| Modelo (test, 670 img) | MAPE | 0–100 | 100–200 | 200–300 | 300+ |
|---|---|---|---|---|---|
| Mediana (no mira la foto) | 23,5% | 86,9 | 16,1 | 28,2 | 51,1 |
| B: ratios de keypoints | 19,2% | 55,5 | 14,5 | 22,8 | 49,1 |
| C': ratios + forma + sexo | 19,1% | 54,6 | 14,6 | 22,5 | 49,2 |
| A: CNN congelada (punto de partida) | 19,6% | 58,6 | 15,1 | 21,7 | 39,5 |
| CNN fine-tuneada (mejor sola) | 15,5% | 31,9 | 13,1 | 19,2 | 31,4 |
| **CNN ensemble (3 mejores)** | **15,3%** | 35,8 | **12,3** | 19,6 | 34,3 |
| *(control)* CNN con sticker | 15,2% | 33,0 | 12,3 | 20,4 | 39,4 |

**Costo de sacar el marcador: +0,08 puntos de MAPE.** Es el número para la defensa: el sticker dejó de ser
necesario, no por un rodeo metodológico sino porque un modelo visual entrenado como corresponde lo reemplaza.
Sigue abierto el error alto en los extremos (0–100 kg y 300+ kg), que es donde hay pocos datos de entrenamiento.

### La geometría dejó de aportar (`25_fusion_cnn.py`)
Fusionar la CNN con el Modelo C' (ratios de keypoints + forma) **empeora**: CNN sola 15,27%, fusión 15,54%. El
combinador le da peso 1,11 a la CNN y 0,10 a la geometría. La CNN ve todo lo que los keypoints capturaban.
**Implicancia de arquitectura**: en producción no hace falta el camino geométrico; YOLO-pose queda solo para
recortar al animal (y para eso alcanzaría con un detector de caja, más simple).

## RESUELTO (23/09/2026): el problema es el RANGO DE PESO, no el dominio visual (`27_cross_cid.py`)
CID permite separar las dos causas porque **se solapa con AcmeAI entre 150 y 300 kg** (Mendeley no se solapaba,
por eso ahí no se podía distinguir). 513 animales, una vista lateral por animal elegida automáticamente por la
relación ancho/alto de los keypoints; detección 100%.

**a) Transferencia directa, sin adaptar** (CNN entrenada en AcmeAI aplicada a CID):

| Subconjunto | n | CNN | Mediana local |
|---|---|---|---|
| CID completo | 513 | 21,1% | 15,8% |
| **Dentro del rango común (150–300 kg)** | 447 | 17,2% | 11,5% |
| Fuera del rango común | 66 | 47,2% | 14,6% |

Predice **151–281 kg cuando el real va de 150 a 816 kg**. La calibración no transfiere.

**b) Con calibración local** (Ridge sobre el embedding de la CNN, usando N animales del lugar pesados en balanza):

| N animales locales | CNN calibrada | Mediana de esos N |
|---|---|---|
| 10 | 14,3% | 16,6% |
| 25 | **10,9%** | 15,8% |
| 50 | **9,5%** | 16,2% |
| 200 | **7,9%** | 15,8% |

**c) El embedding SÍ ve el tamaño real del animal**: correlación de Spearman **+0,82 con la altura medida con
cinta** (en Mendeley era +0,02). Y compitiendo contra la metadata (CV 5-fold dentro de CID):

| Entrada | MAPE |
|---|---|
| Mediana | 15,8% |
| Metadata del productor (raza + edad + sexo) | 13,8% |
| Metadata + altura real medida con cinta | 8,9% |
| **Embedding de la CNN (solo la foto)** | **7,3%** |
| Embedding + metadata | 7,3% (no agrega nada) |

### Conclusiones que bajan al producto
1. **Lo que no transfiere es la CALIBRACIÓN, no la capacidad de ver al animal.** La CNN generaliza a un dominio
   visualmente muy distinto (CID son fotos de estación fija, con plataforma, vara graduada y menor resolución).
2. **El modelo satura en el rango que vio.** Por eso falló en Mendeley (341–644 kg): es incapaz de predecir arriba
   de 267 kg. **El dataset argentino tiene que cubrir el rango local (400–500 kg) o el modelo va a saturar igual.**
3. **Con 25–50 animales pesados en balanza en el establecimiento se llega a 9–11% de MAPE.** Es la respuesta a
   "cuántos animales hay que pesar" y es una puesta en marcha perfectamente viable para un productor.
4. **No hace falta que el productor cargue raza ni edad**: la foto sola (7,3%) supera a la metadata (13,8%) y
   sumarlas no aporta nada. Esto simplifica la app respecto de lo que se había decidido el 22/09.
   *Matiz honesto*: CID tiene fondo constante y vara de referencia en la escena; parte de la facilidad puede venir
   de ahí. Conviene confirmarlo con el dataset propio.

## DECISIÓN (23/09/2026): una sola foto, vista lateral
**La app saca UNA foto lateral por animal. No se usa vista trasera.** Decidido por el usuario.
Razones: con una sola foto ya se cumple el objetivo (15,3%, rango aceptable 15–19%); pedir una segunda foto
complica la captura en corral con animales sueltos y compite con el requisito de <60 s por evento; y agrega
fricción justo después de que los productores rechazaron el sticker por engorroso.
Queda **sin usar** (por si se retoma): 4.541 pares lateral↔trasera en AcmeAI, las 4 vistas de CID y 71 traseras de
Mendeley. La trasera aportaría el ancho corporal (su relación ancho/alto ya correlaciona 0,33–0,44 con el peso).
Si se retomara, el diseño propuesto es **lateral obligatoria + trasera opcional** ("sacá una segunda foto para
mayor precisión"), midiendo antes cuánto aporta con los pares que ya existen, sin recolectar nada nuevo.

## Ablación 2 (24/09/2026): el balanceo ayuda poco y los datos externos NO sirven (atajo por dominio)
4 corridas (`runs/ablacion2.log`). **Comparadas sobre el MISMO conjunto** (val/test de AcmeAI), porque el
`mape_val` que guarda `resumen.csv` NO es comparable entre corridas: las unificadas validan también sobre CID y
Mendeley, que son más fáciles, y eso las hacía parecer mejores.

| Corrida | val AcmeAI | test AcmeAI | 200–300 | 300+ | Predice (en AcmeAI) |
|---|---|---|---|---|---|
| baseline (solo AcmeAI, MSE) | **14,51** | 15,50% | 19,2% | 31,4% | 80–267 kg |
| **+ balanceo por banda** | 14,62 | **15,34%** | **16,7%** | **27,3%** | 78–286 kg |
| + balanceo + datos externos | 14,79 | 15,76% | 16,3% | 28,6% | **91–273 kg** |
| + datos externos, 45 épocas | 15,74 | 16,30% | — | — | — |

**Conclusión 1 — el balanceo sirve, poco**: mejora el rango alto (300+ de 31,4% a 27,3%) sin costo. Se queda.

**Conclusión 2 — los datos externos NO resuelven la saturación: el modelo toma un ATAJO POR DOMINIO.**
Con el dataset unificado el modelo llega a predecir 640 kg… pero **solo en fotos de CID/Mendeley**. Sobre fotos de
AcmeAI sigue topando en 273 kg, aunque ahí hay animales de hasta 465. Es decir: aprendió a **reconocer de qué
dataset viene la foto** y ajustar a la distribución de pesos de ese dataset, en vez de aprender a medir animales
grandes. Síntomas coherentes: CID da 6,28% (fondo fijo, trivial de identificar) y Mendeley 15,6% (estaba en el
entrenamiento), mientras AcmeAI no mejora nada y empeora 0,26 puntos.

**Consecuencia para el proyecto (importante)**: sumar datasets ajenos **no sustituye** tener datos propios del
rango que se quiere estimar. Para estimar novillos argentinos de 400–500 kg hacen falta fotos de novillos de
400–500 kg **capturadas igual que las de producción**, no fotos de otro país/estación.

### Resultado final consolidado (`28_reporte_sin_sticker.py`, test de AcmeAI, 670 imágenes)
| Modelo | MAPE | 0–100 | 100–200 | 200–300 | 300+ |
|---|---|---|---|---|---|
| Mediana | 23,5% | 86,9 | 16,1 | 28,2 | 51,1 |
| B: ratios de keypoints | 19,2% | 55,5 | 14,5 | 22,8 | 49,1 |
| CNN fine-tuneada (mejor sola) | 15,5% | 31,9 | 13,1 | 19,2 | 31,4 |
| **CNN ensemble (3 mejores por val de AcmeAI)** | **15,07%** | 34,1 | **12,5** | 17,9 | 30,5 |
| *(control)* CNN con sticker | 15,19% | 33,0 | 12,3 | 20,4 | 39,4 |

**El ensemble sin marcador (15,07%) ya supera al control con marcador (15,19%).** Sacar el sticker dejó de tener
costo medible. Queda sin resolver el error en los extremos (0–100 kg y 300+ kg), que es donde el dataset tiene
pocos datos y donde el modelo sigue comprimiendo hacia el centro.

## NÚMERO FINAL con incertidumbre (25/09/2026, `correr_semillas.sh` + `31_resumen_semillas.py`)
8 corridas en Fedora: la mejor configuración con 4 semillas, y la variante con balanceo con otras 4.

| Configuración | MAPE medio (4 semillas) | Desvío | Corridas |
|---|---|---|---|
| Sin balanceo | **15,60%** | ±0,18 | 15,84 / 15,63 / 15,42 / 15,52 |
| Con balanceo | 15,83% | ±0,21 | — |

**El balanceo NO sirve** (Mann-Whitney p=0,200; de hecho sale levemente peor). La "mejora" que habíamos visto
(15,34% vs 15,50%) **era ruido de una sola corrida**. Se descarta `--balancear`.

**Ensembles** (promedio geométrico, sobre el test de AcmeAI):
- 4 semillas de la misma configuración: **15,20%**
- 4 semillas + 2 configuraciones: 15,15%
- (el 15,07% reportado antes salía de elegir 3 corridas entre muchas por su val: tenía algo de suerte de selección)

**Cómo reportarlo en el informe**:
> MAPE 15,6% con un modelo único (±0,2 por el azar del entrenamiento) o **15,2% con un ensemble de 4 semillas**.
> Intervalo de confianza del 95% por muestreo de animales: **14,0 – 16,3**.

Dato relevante: **la incertidumbre por qué animales tocan en el test (±1,2) es 6 veces mayor que la del azar del
entrenamiento (±0,2)**. Es decir, el límite hoy no es el modelo ni el entrenamiento: es que el test tiene apenas
77 animales. Otro argumento fuerte para el dataset propio.

## Intervalos de confianza (`30_intervalos.py`, ya corrido)
Bootstrap de 2.000 repeticiones **remuestreando animales** (no imágenes: las fotos del mismo animal están
correlacionadas y hacerlo por imagen daría un intervalo falsamente angosto). El test tiene 670 imágenes pero
solo **77 animales**, y de ahí sale la amplitud.

| Modelo | MAPE | IC 95% |
|---|---|---|
| **CNN ensemble (3)** | **15,07%** | 14,0 – 16,3 |
| (control) con sticker | 15,19% | 14,1 – 16,4 |
| CNN + balanceo | 15,34% | 14,2 – 16,5 |
| CNN (mejor sola) | 15,50% | 14,4 – 16,7 |

**Los intervalos del ensemble sin marcador y del control con marcador se superponen ampliamente: con estos datos
no se puede afirmar que el sticker mejore el resultado.** Es la forma estadísticamente correcta de defender la
decisión de sacarlo (mejor que decir "cuesta 0,08 puntos", que es ruido).
Pendiente en Fedora: `correr_semillas.sh` (variabilidad del entrenamiento) + `31_resumen_semillas.py`, que
combina ambas fuentes y testea si la mejora del balanceo es real.

## Dataset unificado para cubrir el rango alto (`29_dataset_unificado.py`, `29a_tapar_vara.py`)
AcmeAI + CID + Mendeley: **5.125 imágenes, 1.099 animales, 36–816 kg**. Una lateral por animal en CID y Mendeley
(elegida por la relación ancho/alto de los keypoints). Split por animal en cada fuente; el de AcmeAI se respeta
tal cual para poder comparar con los resultados anteriores.

Imágenes de entrenamiento por banda, antes y después:

| Banda | Solo AcmeAI | Unificado |
|---|---|---|
| 0–100 | 191 | 191 |
| 150–200 | 1.392 | 1.431 |
| 200–250 | 449 | 619 |
| 250–300 | 61 | 156 |
| **300+** | **22** | **127** |

**La vara graduada de CID está tapada** (`29a_tapar_vara.py`: detección por color HSV + inpainting con dilatación
aleatoria, 511 de 513 imágenes), por el mismo motivo por el que se tapó el sticker: es un objeto de tamaño
conocido y la CNN podría usarlo como referencia de escala. Detalle del filtro: separa la vara del cabestro
amarillo por **tamaño** (área ≥ 1200 px) y no por posición — filtrarla por el rectángulo del animal dejaba sin
tapar el tramo de abajo, porque la vara pasa justo por detrás del cuerpo.
*Límite honesto*: aunque la vara ya no esté, CID son fotos de estación fija con fondo constante (barandas,
plataforma), que da algo de escala implícita imposible de quitar. Por eso el número que vale para el objetivo
sigue siendo el MAPE del origen `acmeai`, y el entrenamiento lo reporta desglosado por fuente.

## Sesgo de compresión hacia el centro (diagnóstico del error en los extremos)
Medido en el test de AcmeAI con la mejor CNN: **nunca predice fuera de 80–267 kg**, aunque el real va de 51 a 465.

| Banda real | n | Predicho medio | Sesgo | Imágenes de entrenamiento en esa banda |
|---|---|---|---|---|
| 0–100 kg | 52 | 113 kg | **+31%** | 191 |
| 150–200 kg | 293 | 162 kg | −6% | 1.394 |
| 250–300 kg | 15 | 205 kg | −24% | 61 |
| 300+ kg | 7 | 226 kg | **−31%** | 22 |

Ordena bien (Spearman 0,67) pero no calibra la magnitud donde hay pocos datos. Mitigación ya implementada y
pendiente de correr en Fedora: `20_train_cnn.py --balancear` (muestreo con igual probabilidad por banda de peso).

## Primer diagnóstico de transferencia, con Mendeley (Fase 4, `24_cross_mendeley.py`, `26_adaptacion.py`)
Sobre Mendeley (72 bovinos de Mongolia, 341–644 kg, otra raza y cámara):

| Prueba | Resultado |
|---|---|
| YOLO-pose (localizar keypoints) | transfiere bien: 0 fallos, confianza 0,935 |
| Regresión geométrica sin adaptar | MAPE 61% (predice 136–255 kg cuando el real es 341–644) |
| **CNN como extractor + Ridge con N animales locales** | N=5: 17,6% · N=10: 16,1% · N=25: 15,8% · N=40: 16,3% |
| **Mediana de esos mismos N animales locales** | N=5: 18,2% · N=10: 15,9% · N=25: **15,4%** · N=40: 16,0% |
| Embedding de la CNN vs peso real, dentro de Mendeley | Spearman **−0,13** (ruido) |
| Embedding vs altura real / perímetro torácico real | +0,02 / −0,00 (**cero señal**) |
| Referencia biológica: perímetro torácico real vs peso | Spearman 0,93 |

**En Mendeley la CNN adaptada no le gana a predecir la mediana local**: el embedding tiene señal cero ahí.
**Causa identificada después con CID (ver arriba): es el RANGO, no el dominio.** Mendeley va de 341 a 644 kg y el
modelo satura en 267 kg, así que sus características no discriminan en esa zona. Cuando el rango se solapa (CID),
el mismo embedding correlaciona +0,82 con la altura real y la calibración con 25–50 animales da 9–11%.
Este experimento queda como el caso límite: **qué pasa si el dataset de entrenamiento no cubre el rango local**.

## HALLAZGO de producto: la referencia puede estar en el corral, no en el animal
Las fotos del dataset CID (el mismo proyecto Acme AI / Gates Foundation) se toman en una **estación fija con una
vara graduada amarilla** al lado del animal, sobre una plataforma, con 4 vistas por animal. O sea: el proyecto
real **no le pega un marcador a cada animal**, pone **una referencia fija en el lugar de captura**.
Esto abre una tercera opción entre "sticker por animal" (rechazada por los productores) y "solo la imagen":
**una vara o marca de altura conocida en la manga/corral por donde ya pasan los animales**, que se instala una
sola vez, no toca al animal y no agrega trabajo por pesaje. Vale la pena consultarlo con el productor, porque
recupera la escala física que hoy no tenemos.

## Generalización: primera medición (Fase 4a, `24_cross_mendeley.py`)
Modelo entrenado en Bangladesh (AcmeAI) aplicado **tal cual** a Mendeley (72 bovinos de Mongolia Interior,
341–644 kg, otra raza, otra cámara):

| | Resultado |
|---|---|
| YOLO-pose (localizar los 9 keypoints) | **transfiere bien**: 0 fallos de detección, confianza media 0,935 |
| Regresión de peso (ratios, sin marcador) | **MAPE 61%**, MAE 301 kg |
| Predice 136–255 kg cuando el real es 341–644 kg | **no extrapola**: se queda en el rango que vio en entrenamiento |
| Correlación de los ratios con el peso dentro de Mendeley | máx 0,18, mediana 0,08 (casi nula) |
| Mediana de AcmeAI aplicada a Mendeley | 65,7% |
| **Mediana de la propia Mendeley** | **14,9%** |

**Lectura**: lo que no generaliza es la *regresión de peso*, no la visión. Y la última fila es la clave: con solo
conocer la mediana local, el error cae de 61% a 14,9%. O sea, el grueso del problema es el desajuste de
distribución, no que el modelo no vea al animal. **Conclusión para el proyecto: el dataset propio argentino es
imprescindible** y hay que dimensionarlo (Fase 4b del plan). Evidencia directa para el riesgo nuevo de TP5.

## Cuánto aporta la metadata que cargaría el productor (dataset CID, 513 animales, validación cruzada)
| Entrada (SIN imagen) | MAPE |
|---|---|
| Mediana | 15,8% |
| Solo edad + dentición | 16,7% (no aporta) |
| Edad + sexo + color | 16,9% |
| **+ raza** | **14,2%** |
| + altura real medida con cinta | 9,7% |

La **raza sí aporta** (~1,6 puntos). La **edad no discrimina** en este dataset (casi todos tienen 2–3 años), así que
habría que revalidarlo con animales de edades variadas antes de hacerla obligatoria en la app. Una sola medida
física real vale 6 puntos: es la escala que se pierde al sacar el marcador.

## Resultados SIN STICKER (rumbo vigente, Fase 0 del plan nuevo — test, 670 imágenes)
| Modelo | MAPE | 0–100 kg | 100–200 kg | 200–300 kg |
|---|---|---|---|---|
| Mediana | 23,5% | 86,9% | 16,1% | 28,2% |
| Solo sexo + lote (sin mirar la foto) | 21,7% | — | — | — |
| B: ratios de keypoints | 19,2% | 55,5% | 14,5% | 22,8% |
| C': ratios + forma + sexo + confianzas (`--escala animal`) | 19,1% | 54,6% | 14,6% | 22,5% |
| A: visual (ResNet18 congelado) | 19,6% | 58,6% | 15,1% | 21,7% |
| **Fusión A + C'** | **17,8%** | 48,6% | **14,1%** | 19,5% |

Con sticker la fusión daba 15,7%, así que **hoy el marcador vale ~2 puntos de MAPE**. La Fase 1 (CNN) apunta a
recuperarlos sin él. Ojo: el aporte de las features de forma sobre los ratios es mínimo (19,15 → 19,09): la
geometría de keypoints está saturada, el margen está en el modelo visual.

## Resultado del pipeline CON sticker (histórico, ya no es el diseño vigente)
| Modelo | MAPE global | 0–100 kg | 100–200 kg | 200–300 kg | >300 kg (n=7) |
|---|---|---|---|---|---|
| Mediana (baseline) | 23,5% | 86,9% | 16,1% | 28,2% | 51,1% |
| A: visual puro (CNN congelada) | 19,6% | 58,6% | 15,1% | 21,7% | 39,5% |
| B: keypoints, sin escala | 19,2% | 55,5% | 14,5% | 22,8% | 49,1% |
| C: keypoints + sticker + forma | 16,5% | 42,6% | 13,5% | 18,1% | 24,3% |
| **Fusión A+C** | **15,7%** | 41,7% | **12,8%** | 17,0% | 23,0% |

Cada capa (A→B→C→fusión) baja el error de forma consistente. La fusión confirma la hipótesis de la sec. 12 del
plan (dos caminos, geométrico y visual, se complementan). **Objetivo del acta (MAPE<15%) NO se cumple en el global
(15,7%), pero SÍ en la banda 100–200 kg (12,8%, el 78% del dataset)** — coincide con el patrón que ya se había visto
con el dataset externo (13,1% en esa misma banda, ver README de `../modelo-peso/`). Los terneros (0–100 kg) y los
animales grandes (>300 kg, solo 7 casos en test) siguen siendo el cuello de botella y explican casi todo el
faltante para el objetivo global.

`16_calidad.py` mostró una limitación real: el score de calidad (confianza de keypoints + sticker) casi no
discrimina en este dataset, porque el modelo de pose es confiado en casi toda imagen (mediana del score 0,97,
mínimo 0,47 en 670 imágenes) — este dataset no tiene suficiente variabilidad de condiciones de captura para probar
bien la regla "no sé"; haría falta el dataset propio (con más heterogeneidad real de campo) para eso.

## Qué sigue
1. Investigar por qué el error sube tanto en 0–100 kg (42–59% según el modelo) — no investigado todavía la causa
   (ya estaba anotado como pendiente en `../CLAUDE.md` del proyecto para el dataset de AcmeAI en general).
2. Reportar con varias semillas/splits e intervalo de confianza antes de cerrar números para el informe.
3. (Opcional) `10_depth_run.py --modelo dav2l --n 300` en Fedora, por completitud — Depth Pro ya tenía el mejor
   prior de los dos candidatos (estima su propia focal) y no pasó la puerta.
4. `17_predict.py` depende hoy de la máscara que ya trae el dataset (segmentación GT); falta entrenar un
   segmentador propio (YOLO-seg u otro) para que el pipeline funcione sobre una foto nueva, no solo sobre imágenes
   del dataset AcmeAI — es lo que falta para que sea usable en campo de verdad.
5. **Actualizar TP5**: cargar el riesgo del marcador de referencia (ya redactado en `../CLAUDE.md` del proyecto, con
   impacto/probabilidad pendientes) confirmando que sigue vigente — esta corrida es la evidencia que faltaba. No
   editar el xlsx sin pedido explícito del usuario.
6. Frame-selection de video (sec. 4 del plan) y componente fuzzy (sec. 11) siguen fuera de esta primera vuelta,
   como se había decidido en el plan original.

## Decisiones/criterios vigentes
- Siempre reportar MAPE **junto con** la mejora sobre la mediana y por banda de peso.
- **El sticker sigue siendo requisito** (puerta de profundidad no pasó con Depth Pro). Afecta Módulo 1 (el operario debe colocar el marcador) y el riesgo del marcador en TP5, que ya estaba redactado como "probable" — ahora hay evidencia numérica para confirmarlo, no para revertirlo.
- Depth Pro: licencia solo investigación/no comercial (documentar). Ultralytics AGPL.
- No editar `../TP5/PMI_Registro_de_Riesgos.xlsx` sin pedido explícito del usuario. No borrar `../modelo-peso/`.
- Windows/bash: evitar caracteres no-cp1252 (σ, ±) en `print` (rompe la consola); los scripts de este repo ya lo respetan salvo `±` en un log intermedio de `10a`.
- El entorno se reinstala automáticamente al cambiar de SO (`.venv` no es portable entre Windows/Linux): correr `uv sync --extra cpu` (Windows) o `--extra rocm` (Fedora) al volver a cada uno si `uv run` se queja.
