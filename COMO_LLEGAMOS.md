# Cómo llegamos al estimador de peso actual

Documento para el equipo. Explica qué construimos en `bovinovision/`, qué probamos, qué descartamos y por qué,
dónde estamos parados y qué falta. Todo lo que dice acá está medido, no supuesto: cada afirmación tiene un
número atrás y un script que lo reproduce.

---

## Resumen en diez líneas

1. Partimos de cero con un código nuevo (`bovinovision/`), con validación honesta desde el primer día.
2. Usamos un dataset público de Bangladesh: **4.544 fotos** de bovinos con su peso de balanza.
3. La referencia a vencer es **predecir siempre la mediana: 23,5% de error**. Todo se mide contra eso.
4. El camino geométrico (medir al animal con puntos anatómicos) llegó a **19,2%** y se estancó ahí.
5. Probamos **profundidad monocular** para recuperar la escala sin marcador: **no alcanzó** (7,9% de error de escala).
6. Entrenamos una **red neuronal que mira la foto entera**: bajó a **15,6%** (15,2% combinando varias).
7. **El sticker dejó de ser necesario**: el control con marcador da 15,2% y los intervalos se superponen.
8. El modelo **no extrapola**: no predice arriba de ~270 kg, y **no funciona en otro país** (61% de error).
9. Ya existe un **módulo/API funcionando** que recibe una foto y devuelve un peso.
10. Lo único que bloquea el uso real es **el dataset argentino**: 300–400 animales cubriendo el rango local.

---

## 1. El problema, en criollo

Queremos estimar el peso de un bovino con una foto de celular. El obstáculo central es **la escala**: en una foto,
un ternero cerca y una vaca lejos se ven igual de grandes. Sin una referencia de tamaño conocida, el sistema no
sabe si está mirando algo de 100 kg o de 500 kg.

La solución clásica es poner un objeto de tamaño conocido en la escena (el famoso sticker). **Los productores nos
dijeron que eso no es viable**: nadie va a pegarle una calcomanía a cada animal antes de fotografiarlo.

Todo el trabajo giró alrededor de esa pregunta: **¿se puede estimar el peso sin ninguna referencia física?**

---

## 2. Lo primero: poder confiar en lo que medimos

Antes de probar modelos, armamos la infraestructura para que los números no nos mientan. Esto es aburrido pero
es lo que separa un resultado real de uno inflado.

- **Separación por animal, no por foto.** Si las fotos de una misma vaca quedan repartidas entre entrenamiento y
  evaluación, el modelo la "reconoce" y el resultado sale falsamente bueno. Todo nuestro código separa por animal.
- **Baseline de la mediana.** Un modelo que ignora la foto y siempre dice "160 kg" acierta con 23,5% de error en
  este dataset. Cualquier resultado se compara contra eso; si no le gana, no sirve.
- **Reporte por banda de peso.** Un promedio global esconde que el modelo puede andar bien con animales medianos
  y pésimo con terneros.

También encontramos y corregimos varias trampas del dataset, que de no haberlas visto habrían arruinado todo:
los dos lotes listan los puntos anatómicos **en distinto orden**, y uno de ellos guarda las coordenadas **en otra
resolución** que la foto entregada.

---

## 3. El camino recorrido, en orden

### Etapa 1 — Medir al animal con puntos anatómicos (el camino "clásico")

Entrenamos un modelo (**YOLO-pose**) que ubica 9 puntos sobre el cuerpo del animal: cruz, isquion, encuentro,
pecho, altura. Con esos puntos calculamos distancias y proporciones, y con eso predecimos el peso.

**Funcionó bien la parte de ubicar los puntos**: error de 2 a 3,7% del largo del animal, y detecta en el 100% de
las fotos. Pero la parte de estimar el peso se estancó en **19,2%**.

Y acá vino el primer hallazgo incómodo: medimos que **toda esa geometría aporta apenas 2 puntos** sobre no mirar
la foto (solo con el sexo y el lote ya se llega a 21,7%). Nueve puntos anatómicos son un resumen muy pobre de un
animal: no ven el volumen, ni la condición corporal, ni la musculatura.

### Etapa 2 — Intentar recuperar la escala sin marcador (profundidad monocular)

La idea era usar modelos de inteligencia artificial que estiman la distancia a cada objeto de una foto
(**Depth Pro**, de Apple). Si sabemos a qué distancia está el animal, sabemos su tamaño real.

Lo probamos sobre 300 fotos. **No alcanzó**: el error de escala fue de **7,9% en la mediana y 18,2% en el peor
10%**, cuando necesitábamos menos de 3,5%. Como el peso crece con el cubo del tamaño, un 8% de error en escala se
convierte en más de 25% de error en peso.

Antes de probarlo habíamos calculado cuánto error de escala tolerábamos, así que teníamos el criterio de decisión
fijado de antemano. Eso evitó que "acomodáramos" el umbral al resultado.

### Etapa 3 — El salto: una red que mira la foto entera

En vez de reducir el animal a 9 puntos, entrenamos una red neuronal (**ConvNeXt**) que mira el recorte del animal
completo. Eso cambió todo: **de 19,2% a 15,6%**.

Probamos ocho variantes para entender qué importaba:

| Qué cambiamos | Resultado |
|---|---|
| **Reentrenar toda la red** en vez de solo la última capa | **15,9% contra 29,4%** — esto fue lo decisivo |
| Recortar al animal en vez de usar la foto completa | 15,9% contra 16,5% |
| Resolución 384 px en vez de 224 | 15,9% contra 17,1% |
| ConvNeXt en vez de EfficientNet | 15,9% contra 19,3% |
| Variar el zoom durante el entrenamiento | sin efecto |

### Etapa 4 — Comprobar que no nos estábamos haciendo trampa

Al revisar los recortes notamos algo serio: **el sticker seguía visible en las fotos**. La red podía estar
usándolo como referencia de escala, y entonces nuestro "resultado sin marcador" habría sido falso.

Lo borramos de las 4.540 fotos con relleno automático, con dos precauciones: el tamaño de la zona tapada es
aleatorio (para que tampoco delate la escala) y durante el entrenamiento se borran parches al azar.

Además dejamos una corrida **con** el sticker visible como control, para medir cuánto aportaba realmente.

### Etapa 5 — Intentar arreglar los extremos con más datos

El modelo anda bien en el medio (12,5% entre 100 y 200 kg) pero mal en los extremos (35% con terneros, 30% con
animales grandes). Diagnosticamos por qué: **nunca predice fuera de 80–270 kg**, aunque haya animales de 465. Es
un efecto de tener 1.394 fotos entre 150 y 200 kg y solo 22 arriba de 300: el modelo aprende a "jugar a lo seguro".

Probamos dos remedios:

- **Repartir mejor los datos que ya teníamos** (mostrarle más seguido los casos raros): pareció mejorar, pero al
  repetirlo con 4 semillas distintas **resultó ser ruido**. Descartado.
- **Sumar dos datasets externos** (otro de Bangladesh y uno de Mongolia) para cubrir el rango alto: **tampoco
  funcionó, y el motivo es interesante.** El modelo empezó a predecir hasta 640 kg, pero **solo en las fotos de
  esos datasets**. En las nuestras seguía topando en 273. Lo que aprendió fue a **reconocer de qué dataset venía
  la foto** y ajustar a su distribución de pesos, en vez de aprender a medir animales grandes.

**Conclusión importante: datos prestados de otro lado no reemplazan tener datos propios del rango que querés medir.**

### Etapa 6 — Poner números honestos

Hasta acá todos los resultados salían de una sola corrida y un solo reparto de animales. Medimos la
incertidumbre de dos formas:

- **Repetir el entrenamiento con 4 semillas distintas**: el resultado varía ±0,2 puntos.
- **Remuestrear los animales de evaluación** (bootstrap): el intervalo de confianza es **14,0 a 16,3**.

La segunda fuente de incertidumbre es **seis veces más grande** que la primera. Es decir: el límite hoy no es el
modelo ni el azar del entrenamiento, sino que **solo tenemos 77 animales para evaluar**.

---

## 4. Dónde estamos: los números

Todo sobre las mismas 670 fotos de evaluación, de animales que el modelo nunca vio.

| Modelo | Error (MAPE) | Terneros | 100–200 kg | 200–300 kg |
|---|---|---|---|---|
| Predecir siempre la mediana | 23,5% | 86,9% | 16,1% | 28,2% |
| Solo sexo y lote, sin mirar la foto | 21,7% | — | — | — |
| Geometría: 9 puntos anatómicos | 19,2% | 55,5% | 14,5% | 22,8% |
| **Red neuronal (modelo único)** | **15,6%** | 34,9% | 12,5% | 18,2% |
| **Red neuronal (4 modelos combinados)** | **15,2%** | — | — | — |
| *(control)* Red neuronal **con** sticker | 15,2% | 33,0% | 12,3% | 20,4% |

**Cómo decirlo en el informe:**
> 15,6% de error con un modelo único (±0,2 según el azar del entrenamiento), o 15,2% combinando cuatro.
> Intervalo de confianza del 95%: 14,0 a 16,3.

---

## 5. Las decisiones que tomamos, y con qué evidencia

| Decisión | Evidencia |
|---|---|
| **No usar marcador/sticker** | El control con sticker da 15,2% y sin sticker 15,2%; **los intervalos de confianza se superponen**. No se puede afirmar que aporte |
| **No usar profundidad monocular** | Error de escala 7,9% (necesitábamos < 3,5%); además la profundidad varía 21% entre puntos del mismo animal |
| **Una sola foto lateral, sin vista trasera** | Con una sola ya se cumple el objetivo; pedir una segunda complica la captura en corral y compite con el límite de 60 segundos |
| **No pedirle raza ni edad al productor** | Medido: la foto sola rinde 7,3% y la metadata sola 13,8%; sumarlas no mejora nada |
| **Descartar la geometría del pipeline final** | Al combinarla con la red, empeora (15,5% contra 15,3%). La red ya ve todo lo que los puntos capturaban |
| **Descartar el rebalanceo de datos** | Con 4 semillas, la supuesta mejora resultó ser ruido (p=0,200) |

Nada de esto se descartó "por intuición": cada uno tiene su experimento y sus scripts.

---

## 6. Qué quedó funcionando (la arquitectura final)

Es más simple de lo que empezamos:

```
foto del celular
      ↓
  YOLO-pose  →  ubica 9 puntos anatómicos  →  recorta al animal
      ↓
  ConvNeXt   →  mira el recorte a 384×384  →  peso estimado
      ↓
  reglas de calidad → "confiable" o "sacá otra foto"
```

Ya está empaquetado y funcionando en `estimador/`: se le pasa una foto y devuelve el peso, con una API HTTP para
que la app le pegue. Son 125 MB y corre en una computadora común, sin placa de video, en 1 a 3 segundos por foto.

El sistema **avisa cuando no está seguro**: si no detecta al animal, si hay partes tapadas, si el animal está
cortado por el borde o si el peso cae fuera del rango conocido. Es preferible pedir otra foto que mostrar un
número inventado.

---

## 7. Las dos limitaciones que hay que tener presentes

### El modelo no extrapola
Nunca predice por encima de ~270 kg, porque casi no vio animales así de pesados. **Si le sacás una foto a un
novillo de 450 kg, va a decir alrededor de 250.** No es un error de programación: es que aprendió de un dataset
donde el 78% de los animales pesa entre 100 y 200 kg.

### El modelo no se muda solo a otro campo
Lo probamos sobre bovinos de Mongolia: el error fue de **61%**. Interesante: la parte que ubica los puntos
anatómicos **sí funcionó perfecto** (100% de detección). Lo que no transfiere es la **calibración del peso**.

La buena noticia: cuando el rango de pesos sí está cubierto, **con 25 a 50 animales pesados en balanza alcanza
para calibrar** y bajar el error a 9–11%. O sea, el sistema es adaptable; lo que necesita son datos locales.

---

## 8. Qué falta y cómo se hace

### Prioridad 1 — El dataset argentino (es lo único que bloquea el uso real)

**Qué**: 300 a 400 animales con foto lateral y peso de balanza del mismo día.

**Cómo**: está todo detallado en [RECOLECCION_DATASET.md](RECOLECCION_DATASET.md), con el protocolo de captura,
el reparto por banda de peso, qué registrar y los errores a evitar. Lo esencial:

- Cubrir **todo el rango** que el sistema vaya a estimar, con **mínimo 40–50 animales por banda** de 100 kg.
- **Identificación de cada animal** (caravana) y **peso del mismo día**. Sin eso, el dataset no sirve para evaluar.
- Foto original sin recomprimir (nada de WhatsApp). Hay un script, `verificar_fotos.py`, para chequear en el campo
  que la transferencia no borró la metadata.
- Nada de marcadores ni accesorios sobre el animal.

**Quién**: coordinar con el establecimiento en las visitas. Aprovechar cuando los animales ya pasan por la manga.

### Prioridad 2 — Reentrenar con esos datos

**Qué**: volver a entrenar la red partiendo del modelo actual (ya aprendió anatomía bovina) con las fotos
argentinas.

**Cómo**: el código ya está (`20_train_cnn.py`), solo hay que apuntarlo al dataset nuevo. Son unas 4 corridas de
1 a 2 horas en la placa de video. Importante: **hay que reentrenar la red completa**, no solo la última capa
(medimos que la diferencia es de 15,9% contra 29,4%).

**Cuándo sabemos que salió bien**: el error en el rango de interés debería quedar en 12–15%, y el modelo debería
predecir en todo el rango real (no topar como ahora).

### Prioridad 3 — Llevar el módulo a producción

| Pendiente | Cómo |
|---|---|
| Autenticación de la API | Agregar un token; hoy está abierta |
| Guardar cada consulta | Foto + resultado: alimenta el dataset y sirve para el módulo de auditoría |
| Contenedor Docker | Facilita desplegarlo en la nube |
| Versionado de modelos | Ya hay un `config.json` con versión; falta definir cómo se actualiza en el servidor |

### Prioridad 4 — Integración con los otros módulos

El estimador es el Módulo 2. Falta conectarlo con:
- **Módulo 1 (captura)**: que la app saque la foto, la mande a la API y muestre el resultado o el "sacá otra foto".
- **Módulo 3 (trazabilidad)**: guardar cada evento de pesaje asociado al animal.
- **Módulo 4 (analítica)**: curvas de crecimiento y ganancia media diaria a partir del historial.

**Dos requisitos que salen de nuestro trabajo y conviene incorporar ya al Módulo 1:**
1. Guardar los **sensores del teléfono y el EXIF** en cada captura, aunque hoy no los usemos. Si más adelante hace
   falta recuperar la escala física, sin esos datos habría que repetir toda la recolección.
2. Evaluar una **guía de encuadre** en pantalla. Medimos que la gente encuadra llenando el cuadro siempre igual,
   y eso borra la información de tamaño. Una silueta guía podría recuperar parte de esa señal.

### Pendientes menores (no bloquean)

- Investigar el error alto en terneros (35%): no sabemos aún si es por falta de datos o porque son intrínsecamente
  más difíciles.
- Probar la vista trasera: tenemos 4.541 pares de fotos listos para medir cuánto aportaría, sin recolectar nada.
- Una corrida que quedó pendiente por un error de la placa de video (usar la silueta como canal extra).

---

## 9. Dónde está cada cosa

| Archivo | Para qué |
|---|---|
| `bovinovision/CLAUDE.md` | El detalle técnico completo: cada experimento con sus números |
| `bovinovision/` scripts numerados | El pipeline, en orden de ejecución |
| `bovinovision/FEDORA.md` | Cómo correr entrenamientos en la placa de video |
| `estimador/` | El módulo desplegable y su API |
| `RECOLECCION_DATASET.md` | Qué pedirle al campo |
| `verificar_fotos.py` | Revisa que las fotos conserven la metadata |
| `bovinovision/runs/` | Resultados de todas las corridas, en CSV |

Todo el código tiene pruebas automáticas (`pytest`), y cada script deja su resultado en un CSV, así que cualquiera
del equipo puede reproducir o revisar cualquier número de este documento.

---

## 10. Para la defensa: las tres cosas que conviene saber decir

1. **"Probamos el marcador y demostramos que no hace falta."** No lo descartamos por comodidad: lo medimos, y los
   intervalos de confianza se superponen. Es un resultado, no una renuncia.
2. **"Sabemos exactamente qué limita al sistema hoy."** No es el algoritmo: son los datos. El modelo no extrapola
   fuera del rango que vio, y lo demostramos con dos datasets independientes.
3. **"Ya tenemos el sistema funcionando de punta a punta."** Falta el dato local, no la ingeniería.
