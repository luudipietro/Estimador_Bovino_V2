# Módulo 2 — Estimación de peso bovino

Prueba de concepto del pipeline de inferencia de peso, validada sobre un dataset
público con peso de báscula antes de salir a recolectar datos propios.

## Enfoque

**Leer las secciones fechadas en orden antes que esto.** El texto tachado de acá
abajo es la primera hipótesis, descartada con evidencia el 08/09. Después el
proyecto pasó por tres conclusiones más:

1. **"El marcador de referencia SÍ hace falta"** (15/09) — calibrar con una
   calcomanía pegada al animal baja el MAPE de ~22% a 15,8%.
2. **"El sticker puede ser un callejón sin salida"** (15/09) — hipótesis: ese
   15,8% sería un techo del sticker, y la profundidad monocular métrica lo
   rompería sin tocar al animal.
3. **"La profundidad monocular no reemplaza al sticker"** (16/09, más abajo) —
   se corrió el experimento: sin sticker da 19,5%, y sumarle profundidad al
   sticker casi no mueve el número (15,70% → 15,56%). **Decisión vigente: se usa
   el sticker.**

~~Sin escala métrica. No se mide en centímetros ni se usa marcador, giroscopio ni
LiDAR. El peso se infiere de la forma de la silueta, aprovechando que un bovino
no escala uniformemente: un ternero de 200 kg tiene proporciones distintas a una vaca
de 500 kg.~~ Es lo que hacen algunos productos comerciales (Peso Certo en Brasil
reporta ~8,6% de error por animal con una sola foto lateral) — pero como se detalla
abajo, a nosotros nos hizo falta agregar un marcador físico de referencia para
acercarnos a ese nivel de precisión.

El objetivo del acta es MAPE < 15%. El techo conocido del 2D (con marcador) es ~8,6%.

## Pipeline

```
imagen → YOLO-seg (clase 'cow' de COCO, sin entrenar) → máscara
       → descriptores invariantes a escala → regresión → kg
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `config.py` | Rutas. El dataset vive en `data/`, ignorado en git (2,29 GB) |
| `src/preparar.py` | Descomprime el zip e inspecciona la estructura real |
| `src/segmentar.py` | YOLO-seg sobre cada imagen → máscara binaria |
| `src/features.py` | 71 descriptores de forma, ninguno en centímetros |
| `src/entrenar.py` | GroupKFold por animal + Ridge/Boosting + MAPE por franja |
| `src/validar_metodo_acmeai.py` | Valida el método **con sticker** sobre AcmeAI (15,8%) |
| `src/validar_profundidad_monocular.py` | Valida la alternativa **sin sticker**, con profundidad monocular métrica (19,5%: no alcanza) |

## Paso manual: bajar el dataset (cada integrante, una vez)

El dataset **no está en el repo** (pesa 2,3 GB comprimido, GitHub no lo acepta) y
Mendeley está detrás de Cloudflare, así que tampoco se puede automatizar la descarga.
Cada uno lo baja a mano, una sola vez, después de clonar el repo:

1. Abrir https://data.mendeley.com/datasets/h2s22wr5py/2
2. Botón **Download All** (2,29 GB)
3. Guardar como `h2s22wr5py-2.zip` dentro de `modelo-peso/data/`
   (la carpeta `data/` está en `.gitignore`, así que queda local a cada uno)
4. Correr `python src/preparar.py` para descomprimirlo e inspeccionar la estructura

Son 72 bovinos de Mongolia Interior, vista lateral y trasera, fotografiados con
iPhone 13, con keypoints, perímetro torácico, altura a la cruz, largo corporal,
largo de grupa **y peso corporal medido por expertos**. Licencia CC BY 4.0.

## Correr

```bash
python src/preparar.py     # descomprime e inspecciona
python src/segmentar.py    # ~144 imágenes, unos minutos en CPU
python src/features.py     # instantáneo
python src/entrenar.py     # instantáneo
```

## Decisiones de diseño que conviene recordar

**Partición por animal, no por imagen.** `GroupKFold(groups=animal_id)`. Si las fotos
del mismo animal caen en train y test, el modelo lo memoriza y el MAPE reportado es
falso. El estudio de vista dual particiona explícitamente por identidad del animal.

**Las dos orientaciones.** Un animal puede mirar a izquierda o derecha. Ninguna
heurística de canonicalización resultó confiable (probamos comparar el grosor de los
tercios: falla en toros de tren delantero pesado). En vez de adivinar, se emiten las
dos orientaciones de cada silueta y el modelo aprende la invarianza.

**Tres modelos, no uno.** El script compara Schaeffer sobre las medidas reales (piso),
regresión sobre las medidas reales en cm (techo) y regresión sobre la forma sin escala
(lo que nos interesa). Si el último no llega, los otros dos dicen si el problema está
en las features o en el regresor.

## Resultados de la primera corrida (08/09/2026)

Dataset: 72 bovinos adultos de Mongolia Interior, 341–644 kg (media 482, sd 86).
Validación con `GroupKFold` por animal.

| Modelo | Entrada | MAPE | R² |
|---|---|---|---|
| Baseline tonto | nada (predice la mediana) | **14,90%** | −0,03 |
| Schaeffer | medidas reales en cm, sin entrenar | 5,72% | 0,83 |
| Ridge | medidas reales en cm | 4,43% | 0,89 |
| Boosting | medidas reales en cm | **2,59%** | 0,94 |
| Ridge | forma de la silueta, sin escala | 16,89% | **−0,13** |
| Boosting | forma de la silueta, sin escala | 18,20% | **−0,42** |
| Ridge | forma + tamaño aparente en px | 23,95% | −2,56 |

### Tres conclusiones

**1. El objetivo del acta (MAPE < 15%) no discrimina en este rango de pesos.**
Un modelo que ignora la imagen y predice siempre la mediana da 14,90%. Hay que
reportar la *mejora sobre el baseline*, no el MAPE solo, y el dataset propio tiene
que incluir terneros para que la métrica signifique algo.

**2. El enfoque escala-libre por descriptores de silueta falló.** R² negativo:
peor que una constante. No es falta de datos (con 72 animales y 4 medidas reales,
Ridge da 4,43%), ni de segmentación (las máscaras son excelentes).

**3. La causa es la pose.** Diagnóstico:
- El tamaño en píxeles no correlaciona con ninguna medida real (r < 0,25) →
  la distancia de captura **no** fue constante, aunque el paper diga 1 m.
- La silueta no puede recuperar ni la altura a la cruz (R² = −1,37), que es
  puramente geométrica y visible en la foto.
- Las fotos tienen animales con la cabeza arriba y otros pastando, patas abiertas
  o juntas. Cabeza, cuello, patas y cola se mueven independientemente del peso y
  dominan los descriptores de contorno.
- Aislar el torso por erosión morfológica empeora todo (R² −35): es inestable.

### Qué implica

El camino de **keypoints anatómicos no es un refinamiento, es necesario**. Las medidas
que predicen el peso están definidas entre landmarks específicos (cruz, encuentro,
isquion, línea superior e inferior del pecho), y son robustas a la pose de una manera
que el contorno crudo no lo es.

El experimento cuantificó exactamente cuánto cuesta no tenerlos: **17% contra 4%**.

## La escala métrica no hace falta (08/09/2026)

Segunda tanda de experimentos, sobre las medidas reales, con 15 particiones aleatorias
distintas para verificar estabilidad:

| Entrada | MAPE | Estabilidad |
|---|---|---|
| Medidas absolutas en cm | **2,88%** | ± 0,31 |
| **Solo ratios entre medidas (escala-libre)** | **5,53%** | ± 0,48 |
| Control negativo: ratios mezclados al azar | 19,37% | ± 0,97 |
| Baseline tonto | 14,90% | — |

**Los ratios entre landmarks bastan.** Ningún ratio individual correlaciona con el peso
(máximo |r| = 0,26), pero un modelo no lineal sobre el conjunto llega a 5,53%. Es
alometría: los animales grandes no son animales chicos escalados, tienen proporciones
distintas, y esa relación es no lineal (Ridge se queda en 13%, Boosting la captura).

El control negativo se va a 19,37%, peor que el baseline: la señal es real y no un
artefacto del procedimiento de validación.

**Consecuencia para el producto: no hacen falta marcador ArUco, giroscopio, telémetro
ni LiDAR.** Toda la discusión de recuperación de escala se resuelve sola si los
landmarks están bien localizados. Una distancia de captura acotada sigue siendo
deseable por calidad de imagen, pero no por escala.

Advertencia honesta: el 5,53% se midió con medidas *perfectas*, tomadas a mano por
expertos. Los keypoints predichos por un modelo van a tener error y eso degrada el
número. Es el techo del enfoque, no lo que van a obtener. El margen hasta el 15% es
grande, pero hay que medirlo, no suponerlo.

## El marcador de referencia SÍ hace falta (15/09/2026)

La advertencia de arriba se cumplió: al etiquetar las 72 imágenes propias a mano y
medir contra el ratio real (`largo_corporal_cm / altura_cruz_cm`), la correlación
entre el ratio calculado desde la foto y el ratio real del mismo animal fue **r =
0,035** — prácticamente nula. No era un problema de etiquetado (se verificó
visualmente, y el mapeo animal↔peso), sino de que **la distancia y el ángulo de
cámara no estaban controlados**: dos distancias de una misma foto sí comparten un
factor de escala común (r = 0,58-0,71 entre sí), pero ese factor casi no correlaciona
con el tamaño real del animal (r = 0,15-0,22) — la mayor parte de lo que varía en
píxeles de foto a foto es ruido de cámara/pose, no información real, y un ratio entre
dos cantidades mayormente-ruido no cancela el ruido, lo combina.

**Validación externa e independiente**: el dataset público de Acme AI/BMGF
(`www.acmeai.tech Dataset - BMGF-LivestockWeight-CV`, no versionado en este repo —
ver más abajo cómo conseguirlo) — 4.728 imágenes reales de campo en Bangladesh, con
peso de báscula y 9 keypoints anatómicos etiquetados profesionalmente — confirma el
mismo patrón a escala mucho mayor:

| Enfoque | n | Baseline | MAPE |
|---|---|---|---|
| Ratios entre landmarks, sin calibrar | 4.728 | 22,5% | ~22% (sin mejora real) |
| Calibrado, 4 distancias elegidas a mano | 4.538 | 22,5% | 16,9% |
| **Calibrado, las 36 distancias posibles entre los 9 puntos** | 4.538 | 22,5% | **15,8%** |
| — banda 100-200 kg (78% de los datos) | 3.518 | — | **13,1%** ✅ bajo objetivo |

Reproducible con `src/validar_metodo_acmeai.py`.

Calibrar con el marcador (dividir cada distancia por el tamaño en píxeles de una
calcomanía de tamaño fijo pegada al animal, detectada en la segmentación) recuperó
la señal: R² pasó de +0,14 (sin calibrar) a +0,48 (calibrado, con las 36 distancias).
Dejar que Boosting elija entre las 36 distancias posibles en vez de elegir 4 a mano
mejoró el número de forma consistente en todas las bandas de peso — no hacía falta
una red nueva para esto, alcanzó con darle al modelo que ya funciona más información
de entrada.

Se probaron además dos ideas que **no mejoraron nada**:
- Agregar una segunda foto trasera para capturar el ancho del animal (tercera
  dimensión que un solo lateral no puede ver): el ancho correlaciona fuerte con las
  medidas laterales (r = 0,57-0,67), es información redundante por alometría.
- Más cantidad de datos y mejores keypoints (9 en vez de 7, 65x más imágenes) **sin
  calibrar con el marcador**: sigue dando ~22%, confirma que el problema es de
  información faltante (escala), no de cantidad de datos ni precisión de etiquetado.

**Decisión revisada**: sigue sin hacer falta ArUco, giroscopio, telémetro ni LiDAR —
pero **sí hace falta un marcador de referencia físico simple** (una calcomanía de
tamaño y color fijos, no electrónica) en cada foto de captura. No cambia la
Exclusión de TP2 de "no habrá app nativa": no es hardware ni sensor del teléfono.

Especificación del marcador para el dataset propio:
- Forma circular, color que no aparezca en el pelaje (verde o naranja flúor) —
  se detecta por umbral de color simple, no hace falta entrenar segmentación para esto.
- Tamaño fijo y consistente entre todos los animales (no hace falta saber el tamaño
  real en cm — el modelo aprende la relación en "unidades de sticker"; documentar el
  tamaño real igual es buena práctica).
- Ubicación estandarizada y consistente en el cuerpo (ej. mitad del costado), visible,
  sin obstrucciones.
- Una sola foto lateral alcanza (se descartó la necesidad de una segunda foto trasera)
  — el evento de pesaje se mantiene en <60s.

Toda la extracción + calibración + evaluación de esta sección está formalizada en
`src/validar_metodo_acmeai.py` — correrlo reproduce estos números (necesita tener
`www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/` en la raíz del proyecto, ver
`CLAUDE.md`).

Se recomienda además migrar de los 7 landmarks actuales a los 9 que usa AcmeAI
(agrega `shoulderbone` y separa `height_top`/`height_bottom` como par vertical
dedicado, en vez de nuestro `cruz`→`pezuña` diagonal) — ver `etiquetado/README.md`
para la guía a actualizar.

## El sticker puede ser un callejón sin salida (15/09/2026)

> **Superada por el resultado del 16/09 (sección siguiente).** Esta sección es la
> hipótesis y el diseño del experimento, y se deja como evidencia. Dos cosas de
> acá abajo no se sostuvieron al correrlo: que el techo del sticker se deba a un
> escorzo que la profundidad métrica pueda corregir, y que `z_robusto()` ya
> cubriera el problema del borde de la silueta (no lo cubría: la máscara del
> animal estaba mal leída, ver el bug 2 del 16/09).

Antes de comprar la calcomanía y repetir la captura, conviene mirar dos veces la
conclusión de la sección anterior. Está bien medida pero está mal nombrada: el
problema no es "falta escala", es **falta geometría 3D**. Una foto es una
proyección perspectiva, no una semejanza, y eso rompe dos cosas distintas:

| | Qué es | ¿Lo arregla el sticker? |
|---|---|---|
| **Escala global** | la distancia de captura varía → todo el animal crece o achica | ✅ sí |
| **Escorzo / profundidad por punto** | el animal no está perpendicular al eje óptico, y `encuentro` está ~40 cm más cerca de la cámara que `isquion` | ❌ **no, nunca** |

Lo segundo es exactamente lo que quedó documentado como el techo restante en el
docstring de `validar_metodo_acmeai.py`: *"cada landmark está a una profundidad
distinta de la cámara"*. El sticker calibra **un** factor global medido a **una**
profundidad (el flanco). Por construcción no puede corregir lo otro.

Consecuencia incómoda: el sticker no es solo molesto operativamente, **tiene
techo en ~15%**. Aunque se consiga la calcomanía, se actualice la guía de
etiquetado y se repita la captura entera, se choca contra el mismo límite.

### La alternativa: profundidad monocular métrica

Los modelos de profundidad monocular **métrica** (Depth Pro, UniDepth, Metric3D)
salen de una sola RGB y devuelven Z(u,v) **en metros**, más la focal en píxeles
—Depth Pro la predice él mismo cuando no hay EXIF—. Con eso cada keypoint se
retroproyecta a 3D:

```
X = (u−cx)·Z/f      Y = (v−cy)·Z/f      Z
```

y las 36 distancias dejan de estar en píxeles o en "unidades de sticker": pasan a
ser **metros reales en 3D**. Resuelve los dos problemas de la tabla de arriba con
un solo modelo, y **no hay que tocar al animal**.

De yapa, con la máscara y Z salen dos features que ninguna distancia captura:
**área proyectada en m²** y un **volumen en m³**. El volumen es la variable que
uno querría de entrada — es la única con unidades de masa/densidad.

Evidencia de la literatura, toda posterior a la primera tanda de experimentos:

| Trabajo | Resultado |
|---|---|
| *J. Anim. Sci.* 2026 — keypoints + profundidad monocular zero-shot, peso bovino | **R² 0,95 / RMSE 24,2 kg**, contra R² 0,90 / RMSE 32,9 kg con los keypoints 2D solos |
| *Expert Syst. Appl.* 2024 — keypoints + profundidad monocular **a distancias variables** | error 6,75% altura, 7,55% largo, 8,00% profundidad de pecho |
| LaWE (*Eng. Appl. AI* 2025) — "readily photos" desde el celular | usa **stickers circulares** para normalizar escala |
| PickAMoo 2025 — celular en el campo | usa **LiDAR para controlar la distancia**, normaliza a 2,00 m |

Las dos últimas confirman que la conclusión de la sección anterior no estaba mal:
el marcador es el estado de la práctica. Simplemente no es la única salida, y es
la de menor techo.

### El experimento decisivo no necesita una sola foto nueva

El dataset de AcmeAI tiene el sticker en las 4.538 fotos útiles. Eso permite
usarlo como **regla de validación** en vez de como entrada del modelo:

- el sticker tiene tamaño real fijo y desconocido S, así que `sticker_px = f·S/Z`
  → la escala "píxeles por metro" a la profundidad del animal es **proporcional a
  `sticker_px`**;
- la red de profundidad estima esa misma cantidad como **`f / Z_animal`**.

Si las dos correlacionan fuerte, la red mide lo que mide el sticker y el sticker
se puede tirar. Es un coeficiente de correlación contra 4.538 fotos con peso de
báscula, **sin entrenar nada y sin recolectar nada**.

`src/validar_profundidad_monocular.py` hace ese diagnóstico y además el A/B
completo contra el 15,8% actual, en la misma corrida:

```bash
uv run python src/validar_profundidad_monocular.py --limite 400
uv run python src/validar_profundidad_monocular.py --solo-evaluar   # reusa el CSV cacheado
```

Criterio de lectura, fijado **antes** de correrlo para no racionalizar después:

- `r > 0,90` y CV del factor red/sticker `< 10%` → el sticker es prescindible.
- `r < 0,70` → la red no recupera la escala acá, se compra la calcomanía y listo.

La geometría del script está verificada con una escena sintética (un "bovino"
plano de 1,80 × 1,20 m): las distancias 3D dan 0,19% de error, el área da exacto,
y **el mismo animal fotografiado a 4 m y a 7 m arroja la misma área con 0,28% de
desvío — en píxeles crudos habría cambiado 67%**. Ese par de números es el
argumento entero.

Detalle que casi arruina el experimento y quedó cubierto: `wither`, `pinbone` y
los de *girth* caen **sobre el borde** de la silueta, donde Z salta al fondo. Un
muestreo puntual daba 30 m en vez de 4 m. Por eso `z_robusto()` toma la mediana
restringida a los píxeles de adentro de la máscara.

### Complementos que tampoco tocan al animal

Si la profundidad sola no alcanza, estos se suman y son gratis:

1. **Controlar la distancia en vez de calibrarla.** Una estaca, una soga de 3 m o
   una raya pintada en el piso del corral. La escala pasa a ser constante y el
   modelo la absorbe. Es lo que hace PickAMoo con LiDAR, pero con una soga.
2. **Mover el marcador de la vaca al ambiente.** Marcas pintadas en el barral de
   la manga, o una tabla de tamaño fijo atrás. Cumple la restricción al 100% y es
   setup de una sola vez, no por animal.
3. **El giroscopio entra dentro de la Exclusión de TP2.** `DeviceOrientationEvent`
   funciona desde una web con HTTPS (en iOS pide permiso tras un gesto del
   usuario, en Android es directo): no es app nativa. Da pitch/roll → línea de
   horizonte → metrología de vista única con altura de cámara conocida, y además
   permite **rechazar fotos mal tomadas en el momento de sacarlas**.

Se descartó la **caravana SENASA** como regla de referencia: la botón-botón es de
21 mm **±5 mm** por Resolución 754/2006. ±24% de variación es peor que el ruido
que se quiere eliminar.

### Decisión

~~**No comprar el sticker todavía.** Correr primero el diagnóstico de escala: son
unas horas de cómputo sobre datos que ya están bajados, y define la arquitectura
del trabajo entero.~~ Ya se corrió: el sticker se queda (ver sección siguiente).

## La profundidad monocular no reemplaza al sticker (16/09/2026)

Se corrió `src/validar_profundidad_monocular.py` sobre el dataset completo de
AcmeAI: B3+B4, **4.536 fotos** de las 4.538 que usa el método con sticker (las 2
de diferencia son de B3, con keypoints fuera de la imagen). Criterio fijado de
antemano: `r > 0,90` → el sticker sobra; `r < 0,70` → se compra la calcomanía.

**Veredicto: r = 0,59 → el sticker se queda.**

### Qué modelo se usó, y por qué no Depth Pro

La máquina de la corrida no tiene CUDA (GPU AMD RX 6700 XT, que PyTorch no usa en
Windows), así que todo corrió en CPU. Medido sobre la misma foto, a 1024 px:

| Modelo | s/imagen | Z mediana | Focal |
|---|---|---|---|
| Depth Pro (`apple/DepthPro-hf`) | 440 | 2,24 m | 814,5 px (predicha) |
| **Depth Anything V2 Metric Indoor Small** | **0,3** | **2,23 m** | 803,7 px (FOV 65° asumido) |
| Depth Anything V2 Metric Outdoor Small | 0,4 | 10,61 m ❌ | 803,7 px (FOV 65° asumido) |

Con Depth Pro, el dataset completo serían ~3 semanas de CPU. Se usó el Indoor
Small porque coincide con Depth Pro en la distancia al animal, y porque el FOV de
65° que el script asume cuando el modelo no predice la focal queda a 1,3% de la
que predice Depth Pro para esta cámara. El Outdoor se descartó: está entrenado con
escenas de manejo y pone al animal a 10 m.

**Salvedad**: esa equivalencia se comprobó sobre una sola foto y un solo escalar.
Con r = 0,59 el resultado queda debajo del umbral, pero no de forma aplastante.
Confirmarlo con Depth Pro sobre ~100 imágenes son ~12 h de CPU (o mucho menos en
una máquina con GPU NVIDIA). Queda pendiente, no bloqueante.

### Diagnóstico de escala

| | n | r | r (log) |
|---|---|---|---|
| Global | 4.536 | +0,254 | +0,505 |
| **Global, sin el 1% de cada cola** | 4.443 | **+0,594** | +0,605 |
| Solo B3 | 2.597 | +0,288 | +0,534 |
| Solo B4 | 1.939 | +0,216 | +0,481 |

El +0,254 crudo está hundido por outliers de segmentación del sticker (hay uno de
587 px, con mediana de 57 px). El número a leer es el recortado: **0,59**. Se
descartaron los dos confusores que podían ensuciar la lectura:

- **No es que B3 y B4 usen calcomanías de distinto tamaño**: separados por batch
  da lo mismo que juntos.
- **No es que falte escala que recuperar**: el tamaño aparente del sticker varía
  p95/p5 = 1,49× (CV 11,7%).

Cómo falla es informativo: la escala que estima la red varía **en la magnitud
correcta** (p95/p5 = 1,45×, CV 11,4%), pero solo comparte el 35% de su varianza
con la real (r² = 0,35). No se queda corta: varía lo que tiene que variar, pero
por los motivos equivocados.

### A/B contra el método con sticker, sobre las mismas 4.536 filas

| Entrada | MAPE | R² | 100-200 kg (n=3.517) |
|---|---|---|---|
| Baseline tonto | 22,45% | — | — |
| **A) Con sticker — 36 distancias calibradas** | **15,70%** | +0,498 | **13,01%** ✅ |
| B) Sin sticker — 36 distancias 3D en metros | 19,48% | +0,205 | 15,16% |
| C) Sin sticker — 3D + área m² + volumen m³ | 19,09% | +0,242 | 14,94% |
| D) Sin sticker — solo área m² + volumen m³ | 20,01% | +0,186 | 15,58% |
| **E) Sticker + profundidad** | **15,56%** | +0,510 | 12,97% |

Boosting en todas las filas; Ridge da el mismo orden. La fila A reproduce el
15,8% / 13,1% de la sección del 15/09, así que las dos corridas son comparables.
En terneros de menos de 100 kg (n=277) la profundidad empeora mucho: 67% sin
sticker contra 47% con sticker.

### Qué dice esto sobre la hipótesis del escorzo

La sección anterior sostenía que el sticker tiene techo porque calibra un solo
factor global y no puede corregir que cada landmark esté a distinta profundidad,
y que la profundidad métrica rompería ese techo. **La fila E no lo acompaña**:
dándole al modelo, encima del sticker, las 36 distancias en metros reales en 3D
más área y volumen, el MAPE baja 0,14 puntos.

Hay dos lecturas posibles: o el escorzo por punto no es lo que limita, o esta red
no mide la profundidad por punto con la precisión necesaria para corregirlo. Este
experimento no las separa. Pero la conclusión práctica es la misma en las dos:
**con profundidad monocular no aparece una salida de mayor techo que el sticker.**

### Dos bugs del script, corregidos antes de medir

Ninguno era de rendimiento: los dos daban un resultado falso sin avisar.

1. **Coordenadas de B3.** El JSON de B3 declara 4160×3120 y los keypoints están
   en ese sistema, pero el jpg en disco es de 1900×1425. Al escalar por el tamaño
   del archivo, los puntos caían fuera de la imagen: **se descartaba el 98,8% de
   B3 en silencio** y el experimento quedaba reducido a B4. Ahora se escala contra
   las dimensiones que declara el JSON. `validar_metodo_acmeai.py` no tiene este
   problema: nunca abre el jpg y trabaja todo en coordenadas del JSON.
2. **Máscaras.** Los `___fuse.png` no son binarios sobre fondo negro: son mapas de
   3 colores planos — fondo `(0,255,193)`, animal `(255,30,249)`, sticker
   `(0,117,255)` — sin un solo píxel negro. Tomar "cuerpo = todo lo que no es
   negro" daba la foto entera (99,6%), lo que anulaba el recorte de `z_robusto()`
   en el borde de la silueta y convertía `area_m2` y `volumen_m3` en el área de la
   imagen. El color del animal se verificó contra los keypoints: el 74-76% cae
   adentro y el resto justo sobre el borde, como se esperaba. Tampoco afecta a
   `validar_metodo_acmeai.py`, que de la máscara solo usa el color del sticker.

Otros cambios al script, ninguno cambia los números:

- `--ancho` (1024 por defecto): redimensiona antes de inferir. Es exacto, no una
  aproximación: escalar u, v y f por el mismo factor deja X/Z, Y/Z y Z iguales.
  Abarata la corrida porque el mapa de profundidad, la máscara, el área y el
  volumen escalan con el tamaño de la foto, no con el del modelo.
- Un cache por modelo, `data/tablas/acmeai_profundidad_<modelo>.csv`, para que
  Depth Pro y Depth Anything no se pisen.
- Los descartes se cuentan por motivo y se imprimen siempre, y una silueta de
  tamaño absurdo (fuera del 2-60% de la imagen) se descarta con motivo explícito,
  para que una paleta distinta en otro batch no vuelva a pasar callada.

### Reproducir

```bash
uv sync
uv run python src/validar_profundidad_monocular.py --modelo depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
uv run python src/validar_profundidad_monocular.py --modelo depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf --solo-evaluar
```

Son ~50 min en CPU (Ryzen de 12 hilos) para las 4.536 fotos. El script imprime
el diagnóstico crudo y las filas A a D. **La fila E, la correlación sin colas y la
separación por batch salieron de un análisis aparte sobre el CSV cacheado, que
todavía no está en el repo.**

## Estado

- [x] Entorno: torch 2.14 CPU, OpenCV 5, scikit-learn 1.9, ultralytics 8.4
- [x] Pipeline completo corriendo end-to-end
- [x] Segmentación: vista lateral 72/72 (100%), vista trasera 14/72 (19%)
- [x] Baseline, Schaeffer y modelos morfométricos medidos
- [x] Hipótesis escala-libre por silueta probada y **descartada con evidencia**
- [x] Hipótesis escala-libre por ratios entre landmarks **confirmada** (5,53%)
- [x] Proyecto de etiquetado listo: 72 imágenes exportadas + guía de 7 landmarks
- [x] Config y script de YOLO-pose listos
- [x] Etiquetar las 72 imágenes (hecho, un solo integrante etiquetó las 72)
- [x] Entrenar YOLO-pose y medir el error real con keypoints predichos
      (`evaluar_holdout.py`, sin contaminar con animales ya vistos en el entrenamiento)
- [x] Diagnosticar por qué el ratio sin calibrar da peor que el baseline
      (r=0,035 entre ratio en foto y ratio real — ver sección de arriba)
- [x] Validar la hipótesis del marcador de referencia con dataset externo (AcmeAI,
      4.728 imágenes) — **confirmada**: 14-17% MAPE calibrado, bajo objetivo en la
      banda de 100-200 kg
- [x] Escribir el A/B sin marcador (`validar_profundidad_monocular.py`), con la
      geometría verificada sobre una escena sintética
- [x] Correrlo y decidir con el criterio ya fijado — **r = 0,59 < 0,70: el sticker
      se queda** (16/09). Antes se corrigieron dos bugs que falseaban el resultado
      (coordenadas de B3 y color de las máscaras)
- [ ] **Bloqueante**: definir y conseguir el marcador físico (sticker) antes de
      salir a recolectar el dataset propio
- [ ] Actualizar `etiquetado/README.md` a 9 landmarks + posición del sticker
- [ ] Etiquetar el dataset propio con el sticker desde el primer registro
- [ ] Llevar al repo el análisis de la fila E, la correlación sin colas y la
      separación por batch
- [ ] (No bloqueante) Confirmar el veredicto con Depth Pro sobre ~100 imágenes
      (~12 h de CPU, o una máquina con GPU NVIDIA)

## Cómo sigue

El diagnóstico de escala ya salió (16/09): el protocolo de captura lleva sticker.
Lo primero es definirlo y conseguirlo, con la especificación de la sección del
15/09 (circular, color flúor que no aparezca en el pelaje, tamaño fijo, posición
estandarizada en el flanco).

**Después:**

```bash
python src/exportar_etiquetado.py          # ya corrido: 72 imagenes en etiquetado/imagenes
# etiquetar en CVAT o Roboflow -> exportar formato YOLO pose
python src/entrenar_pose.py                # YOLO-pose, 7 (o 9) keypoints
python src/evaluar_holdout.py              # MAPE real, sin contaminar train/val
python src/medidas_desde_keypoints.py      # keypoints -> ratios -> peso (evaluación)
```

No salir a recolectar el dataset propio sin el sticker: si se arranca a
fotografiar sin marcador hay que repetir la captura completa.
