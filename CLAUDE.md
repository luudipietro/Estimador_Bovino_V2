# BovinoVision — Sistema de Estimación de Peso Bovino por Visión Artificial

Proyecto Final de Ingeniería en Sistemas (UTN-FRT). Trabajo práctico grupal, cátedra Proyecto Final. Repo local (no es un repositorio git) con los entregables de cada TP.

> ## ⚠️ ESTADO ACTUAL (25/09/2026) — leer antes que el resto de este archivo
> **EL STICKER / MARCADOR DE REFERENCIA SE ELIMINÓ DEL DISEÑO.** Los productores dijeron que pegarle un marcador
> a cada animal no es viable en campo. Se probó y se resolvió: **una CNN entrenada correctamente lo reemplaza**.
> Todo lo que este archivo dice más abajo sobre el marcador (secciones del 15/09) es **historia, no diseño vigente**.
>
> - **Resultado actual: 15,6% de MAPE** con un modelo único (±0,2 entre semillas), **15,2% con ensemble**.
>   IC 95%: 14,0–16,3. Sin marcador, una sola foto lateral, sin sensores.
> - El control **con** marcador da 15,2% y **sus intervalos se superponen**: estadísticamente el sticker no aporta.
> - **Módulo desplegable listo**: `estimador/` recibe una foto y devuelve un peso (ver su README).
> - **Limitación medida**: el modelo no extrapola (no predice arriba de ~270 kg) y no transfiere a otro país/rango.
>   Para usarlo en Argentina hay que **reentrenarlo con dataset propio** (ver `RECOLECCION_DATASET.md`).
>
> **Dónde está cada contexto**:
> | Archivo | Qué contiene |
> |---|---|
> | **`COMO_LLEGAMOS.md`** | **explicación para el equipo**: el recorrido completo, las decisiones y qué falta |
> | este archivo | el proyecto: equipo, alcance, TPs, riesgos |
> | `bovinovision/CLAUDE.md` | **todo el detalle técnico** del modelo: experimentos, resultados, decisiones |
> | `bovinovision/FEDORA.md` | cómo correr entrenamientos en la GPU |
> | `RECOLECCION_DATASET.md` | qué pedirle al campo para armar el dataset propio |
> | `estimador/README.md` | el módulo/API desplegable |
> | `verificar_fotos.py` | revisa que las fotos del campo conserven la metadata |
>
> **Impacto en los TP** (pendiente de volcar a los documentos):
> - **TP2/alcance**: ya no hay marcador físico; la captura es una sola foto lateral, sin vista trasera.
> - **TP5/riesgos**: **retirar** el riesgo del marcador de referencia (ya no aplica). **Dar de alta** uno nuevo:
>   *"el modelo no generaliza a razas ni rangos de peso fuera del dataset de entrenamiento"*, con plan de
>   respuesta = dataset propio argentino de 300–400 animales cubriendo el rango local.
> - **Objetivo 2 del Acta** (MAPE < 15%): hoy estamos en 15,6% sobre datos ajenos. Reportar siempre con su
>   intervalo y aclarar que el número definitivo sale del dataset propio.

## Equipo y roles
- **Carlino, Luciano Ezequiel** — Director de Proyecto / Desarrollador-Analista
- **de la Torre, Máximo César** — Desarrollador/Diseñador
- **Di Pietro, Luciano** — Desarrollador/Diseñador
- **Miranda López, Víctor Manuel** — Desarrollador/Analista
- Tutor: **Patricio Moreno** — Cotutora: **Alexandra Dufour**
- Cliente: Productor ganadero (externo)

## Qué es el proyecto
Plataforma de software que estima el peso de bovinos a partir de imágenes/videos capturados con smartphones (o cámara fija), usando visión artificial, evitando el uso de balanza tradicional. Objetivo: reducir tiempo, costos y estrés animal en el proceso de pesaje, y dar seguimiento histórico/analítico del ganado.

Duración: **13/05/2026 → 18/11/2026** (7 meses), fechas fijas por calendario académico de la cátedra (no modificables unilateralmente).

## Los 6 módulos del sistema (alcance del producto)
1. **Captura móvil** — smartphone Android/iPhone, registra evento de pesaje en <60s, valida calidad de imagen, guarda offline si no hay conexión.
2. **Procesamiento e inferencia de peso** — pipeline de visión artificial en la nube, segmenta al bovino, extrae medidas morfométricas (perímetro torácico, largo corporal, altura), estima peso con **MAPE < 15%**.
3. **Gestión y trazabilidad** — repositorio de eventos de pesaje por animal (id, fecha/hora, peso, imagen), soporta ≥50 animales simultáneos, agrupables en lotes.
4. **Analítica y reportes** — curvas de crecimiento, Ganancia Media Diaria (GMD), alertas por variaciones anómalas, reportes visuales.
5. **Gestión de usuarios** — ABM de usuarios y roles/permisos.
6. **Auditoría** — registro de acciones relevantes (quién/qué/cuándo) sobre la información del sistema.

## Exclusiones clave del alcance
- No se desarrolla hardware/sensores propios (solo smartphones y cámaras web comerciales).
- No hay integración con ERP, plataformas ganaderas externas ni mercados de comercialización.
- No se estima grasa, musculatura, estado sanitario ni diagnóstico veterinario.
- No habrá app nativa (APK/IPA); es multiplataforma.
- No se garantiza precisión para razas fuera del dataset de entrenamiento.
- No se contempla despliegue productivo comercial dentro del alcance del proyecto académico.
- No incluye facturación ni conexión a lectores de caravanas electrónicas ni balanzas.

## Restricciones y supuestos importantes
- Presupuesto limitado a planes gratuitos/bajo costo (sin financiamiento externo); tecnologías de código abierto o gratuitas.
- Dataset mínimo de 200 registros etiquetados, ≥3 razas, condiciones de iluminación variadas (hito: 26/08/2026).
- Depende de disponibilidad de un establecimiento ganadero colaborador para dataset y prueba piloto.
- Depende de disponibilidad y estabilidad de precios de proveedores externos: infraestructura cloud (AWS/Hostinger) y servicios de IA/LLM (Ultralytics y otros).
- Los 4 integrantes deben mantener disponibilidad horaria compatible con el cronograma hasta el cierre.
- Cada animal tiene identificador único propio del establecimiento o de SENASA.

## Objetivos específicos (Acta de Constitución) — con métricas de éxito
1. MVP con los 4 módulos centrales integrados en ≤7 meses.
2. Modelo de estimación con MAPE < 15% (validado con ≥100 registros con peso real en báscula) antes del Release Final (02/11/2026).
3. Captura compatible con gama media, evento de pesaje completo en ≤60s.
4. 100% de eventos de pesaje trazables, seguimiento simultáneo de ≥50 animales.
5. Participación activa de ≥85% de los interesados formalmente identificados.
6. ≥80% de los hitos del cronograma cumplidos en fecha y presupuesto.
7. Todos los riesgos críticos identificados, con planes de respuesta para ≥80% de ellos, antes del 26/08/2026 (**este es el objetivo que cubre el TP5 actual**).
8. 100% de funcionalidades aprobadas en pruebas de calidad + SUS > 70 puntos con ≥5 usuarios.

## Interesados relevantes (más allá del equipo)
- Productor ganadero (cliente final), Cristian Avila (ingeniero zootecnista, operario de campo)
- Compradores de ganado, Decano UTN-FRT, Gustavo Carlino (investigador/zootecnista)
- EEAOC, UNT (Facultad de Agronomía/Zootecnia/Veterinaria), Senasa (supervisión regulatoria)
- Competidores identificados: **Albor**, **Básculas Magris** (empresas privadas)
- ONG **Igualdad Animal** (protección animal)
- Proveedor de infraestructura cloud, proveedor de servicios de IA/LLM

## Situación financiera (TP4)
- Esfuerzo total: 1155 hs. Precio de venta estimado (sin IVA): ~$23,14 M ARS. Margen: 20%.
- Costos: personal ~$14,49M, subcontrataciones (nube + IA) ~$2,81M, costes varios (equipo, internet, licencias) ~$1,0M, otros gastos (cámara/soporte, viajes de campo, dataset) ~$0,97M.
- Ingresos proyectados en 3 hitos (mes 1, mes 4, mes 10); hay meses con margen bruto negativo (mes 5 a 9) — dependencia de flujo de caja ajustado.
- El archivo de planificación financiera tiene hojas con fórmulas rotas (`#REF!`) en "Indicadores", "Situación", "Evolución" y "Cierre" — struct heredada de plantilla, no completada para este proyecto (dato a tener en cuenta si se referencia).

## Estado actual del Registro de Riesgos (TP5) — `PMI_Registro_de_Riesgos.xlsx`
**Archivo de solo lectura para el asistente — no debe modificarse directamente, solo leerse para dar contexto.**

13 riesgos cargados, 5 clasificados como **Inaceptables** (id 1, 2, 4, 5, 11) y 8 como **Aceptables** (id 3, 6, 7, 8, 9, 10, 12, 13). Planes de Respuesta al Riesgo ya redactados (hojas separadas) para los riesgos 1 a 5 (Evitar/Mitigar/Transferir + Acciones de Contingencia). Falta plan para el riesgo 11 (Inaceptable) y para cualquier riesgo nuevo que resulte inaceptable.

Riesgos ya registrados (resumen):
1. Error en estimación del cronograma — **Inaceptable**
2. Error en estimación de costos — **Inaceptable**
3. Error en elección de tecnologías/herramientas — Aceptable
4. No se encuentran imágenes suficientes/adecuadas para el dataset — **Inaceptable**
5. No se logra MAPE < 15% en desarrollo — **Inaceptable**
6. Aparición de una aplicación similar (competencia) — Aceptable
7. Incapacidad del productor de adaptar sus procesos de negocio — Aceptable
8. Rechazo por parte de autoridades a la solución — Aceptable (estado: Finished)
9. Disminución importante de la producción ganadera — Aceptable
10. Incompatibilidad del hardware propuesto con procesos de negocio — Aceptable
11. Desconfianza de productores en la precisión del modelo — **Inaceptable**
12. Rechazo de los animales a los drones — Aceptable
13. Tiempos de procesamiento en la nube excesivos — Aceptable

**Meta pendiente para el TP5**: llegar a ≥20 riesgos totales, con al menos la mitad (≥10) clasificados como Inaceptables, y desarrollar plan de respuesta para cada riesgo Inaceptable (objetivo 7 del Acta de Constitución).

## Consideraciones para asistir con TP5
- No editar `TP5/PMI_Registro_de_Riesgos.xlsx` salvo pedido explícito del usuario.
- Los riesgos nuevos deben alinearse con: alcance/exclusiones (TP2), restricciones/supuestos (TP2), EDT (TP2), plan financiero (TP4) y objetivos del Acta (TP1) — para que cada riesgo tenga trazabilidad real al proyecto, no genérico.
- Categorías del proyecto poco cubiertas hasta ahora en el registro: seguridad de la información/datos, riesgos regulatorios (protección de datos, bienestar animal), riesgos cambiarios/inflación (impacto en costos dólar de AWS/IA en contexto argentino), riesgos de recursos humanos del equipo (bus factor, disponibilidad), riesgos de conectividad rural, riesgos de integración entre los 6 módulos, riesgos éticos/reputacionales (ONGs, prensa).
- **Riesgo nuevo pendiente de cargar** (surgió de la investigación del Módulo 2, ver más abajo): *"El error de localización de keypoints degrada la estimación de peso por encima del 15%"*. Impacto 4, Probabilidad 0,45, VME 1,8 → **Inaceptable**, siguiendo la misma escala que los riesgos 1-13. Falta su plan de respuesta.
- **Segundo riesgo nuevo pendiente de cargar (15/09/2026, reemplaza la nota anterior sobre ArUco)**: la decisión de "no hace falta marcador" se revirtió con evidencia — ahora el sistema SÍ depende de un marcador físico de referencia (sticker, no ArUco/hardware) en cada foto. Esto introduce un riesgo operativo nuevo que antes no aplicaba: *"El marcador de referencia no está presente, se pierde o el productor no lo coloca correctamente, degradando o impidiendo la estimación de peso"*. Falta estimarle Impacto/Probabilidad/VME y clasificarlo. Ver sección de Módulo 2 más abajo para el detalle completo de por qué se revirtió la decisión.

## Módulo de estimación de peso — investigación y estado (`modelo-peso/`)

Carpeta con el código de la prueba de concepto del Módulo 2 (procesamiento e inferencia
de peso). Tiene su propio repo git, ya con commits, pusheado a
`github.com/luudipietro/estimador_peso_bovino` (rama `main`).

### Decisión de arquitectura — REVISADA el 15/09/2026 (leer esto, no la versión vieja)

La hipótesis original (comparando ArUco, giroscopio, telemetría acústica, telémetro
Bluetooth, autofocus, estéreo dual-cámara y LiDAR) descartó correctamente todo ese
hardware/sensores — eso sigue en pie. Pero la conclusión de que **"los ratios entre
landmarks alcanzan sin ningún marcador de referencia" se probó y se descartó con
evidencia**, en dos datasets distintos:

1. **Dataset propio (72 animales, etiquetados a mano por un integrante)**: el ratio
   calculado desde la foto (`largo_corporal_px/altura_cruz_px`) correlaciona **r=0,035**
   con el ratio real medido en cm del mismo animal — prácticamente nulo. Causa
   diagnosticada: la distancia/ángulo de cámara no está controlada entre fotos, y esa
   variación pesa más que la señal biológica real (que ya era frágil incluso con
   medidas perfectas: máximo |r|=0,26 por ratio individual).
2. **Dataset externo de validación** (Acme AI / Bill & Melinda Gates Foundation,
   `www.acmeai.tech Dataset - BMGF-LivestockWeight-CV`, no versionado en este repo,
   4.728 imágenes reales de campo en Bangladesh con peso de báscula): sin calibrar,
   ~22% de MAPE (apenas mejor que el baseline). **Calibrando con un marcador físico de
   referencia** (una calcomanía de tamaño fijo pegada al animal, usada para convertir
   distancias en píxeles a una escala real) y usando las 36 distancias posibles entre
   los 9 keypoints en vez de 4 elegidas a mano, el MAPE baja a **15,8%**, y en la
   banda de 100-200 kg (el 78% de esos datos) llega a **13,1%, por debajo del
   objetivo**. Reproducible con `modelo-peso/src/validar_metodo_acmeai.py`.

**Decisión revisada**: el sistema **sí necesita un marcador de referencia físico**
(un sticker de tamaño y color fijos, no electrónico) en cada foto de captura, además
de los landmarks anatómicos. No es lo mismo que ArUco/giroscopio/LiDAR — es una
calcomanía de papel, no hardware ni sensor del teléfono, así que **la Exclusión N.º 4
de TP2 ("no se desarrollará app nativa") se mantiene sin cambios** igual que antes.
Pero si se retoma la discusión de "no hace falta nada especial en la foto", hay que
saber que **eso específicamente sí se descartó con evidencia** — lo que se mantiene
descartado es el hardware/sensores, no un marcador físico simple.

Implicancia operativa concreta: el Módulo 1 (captura móvil) tiene que contemplar que
el productor/operario coloque el marcador en el animal antes de la foto, en una
ubicación estandarizada — esto es nuevo respecto al diseño anterior y probablemente
haya que reflejarlo en los requerimientos de ese módulo si no está ya. Ver
`modelo-peso/README.md`, sección "El marcador de referencia SÍ hace falta (15/09/2026)"
para el detalle completo de la evidencia y la especificación propuesta del marcador.

### La arquitectura de estimación

Son **dos modelos entrenados por separado**, que no se tocan entre sí en tiempo de
entrenamiento:

1. **YOLO-pose** — aprende a ubicar 7 keypoints anatómicos sobre la foto lateral del
   animal. Se entrena con imagen + puntos marcados a mano. No usa el peso para nada.
2. **Ridge / HistGradientBoosting** — aprende a predecir el peso a partir de los
   *ratios* calculados entre esos 7 puntos (7 números de entrada, sin imagen). Se
   entrena con ratios + peso real de báscula.

En producción: foto → YOLO-pose ubica los 7 puntos → se calculan distancias en píxeles
entre pares específicos → se calculan ratios (adimensionales) → el segundo modelo
predice el peso en kg.

Los 7 landmarks actuales, en el orden que usa todo el código (`flip_idx` es la
identidad `[0,1,2,3,4,5,6]` porque ninguno tiene par izquierda/derecha):
`0 cruz, 1 pecho_sup, 2 pecho_inf, 3 encuentro, 4 lumbar, 5 isquion, 6 pezuna`.
Definición exacta de cada uno: `modelo-peso/etiquetado/README.md`.

**Pendiente (15/09/2026)**: se recomienda migrar a los 9 landmarks validados con el
dataset de AcmeAI (agrega `shoulderbone`, y separa `height_top`/`height_bottom` como
par vertical dedicado en vez del `cruz`→`pezuña` diagonal actual, que mezclaba altura
con la posición de la pata delantera). Falta actualizar `etiquetado/README.md` con
esto y con dónde va el marcador de referencia.

### Dataset usado para la prueba de concepto

**Cattle side view and back view dataset** (Mendeley, DOI `10.17632/h2s22wr5py.2`,
licencia CC BY 4.0). 72 bovinos adultos de Mongolia Interior, 341-644 kg, fotografiados
con iPhone 13, con medidas reales (altura a la cruz, largo corporal, perímetro
torácico, largo de grupa) y peso de báscula en `measurements.xlsx`. **No trae
keypoints** aunque el paper los mencione — hay que etiquetarlos.

**Ubicación del dataset — ya resuelto (10/09/2026):** `config.py` usa una ruta
**relativa al propio archivo** (`Path(__file__).resolve().parent / "data"`), no
absoluta ni ligada a un usuario. Cada integrante del equipo descarga el zip a mano
(Mendeley bloquea la descarga automatizada) y lo coloca en `modelo-peso/data/`, dentro
de su propio clon del repo — instrucciones en `modelo-peso/README.md`. La carpeta
`data/` está en `modelo-peso/.gitignore`, así que nunca se sube: cada persona la tiene
localmente pero el repo queda liviano.

- `data/h2s22wr5py-2.zip` (2,3 GB) y `data/crudo/` descomprimido (otros 2,3 GB).
- `data/mascaras/` — siluetas segmentadas con YOLO-seg (668 KB).
- `data/tablas/` — CSVs intermedios (`segmentacion.csv`, `features.csv`, `dataset.csv`,
  `keypoints.csv`, y los `acmeai_*.csv` del dataset externo, ver abajo).

**Segundo dataset, usado para validar la hipótesis del marcador (15/09/2026)**:
`www.acmeai.tech Dataset - BMGF-LivestockWeight-CV`, en la raíz del proyecto (afuera
de `modelo-peso/`, **no versionado, no tiene .gitignore propio todavía** — pendiente
de decidir si se referencia por instrucciones de descarga como el de Mendeley).
4.728 imágenes reales de campo (Bangladesh), peso real en el nombre del archivo,
keypoints en formato COCO JSON, y una calcomanía de referencia física ya segmentada
en las máscaras — es el dataset que permitió confirmar la necesidad del marcador. Viene
con un PDF explicativo (`www.acmeai.tech BMGF - LivestockWeight - CV.pdf`) del proyecto
real de Acme AI / Bill & Melinda Gates Foundation en Bangladesh del que salió. Los CSVs
ya procesados (`acmeai_b2_b3_b4_side.csv`, etc.) están en `modelo-peso/data/tablas/`
para no tener que reprocesar las máscaras cada vez (tarda ~2 min por batch).

**Estado físico — resuelto (10/09/2026)**: el proyecto ya vive en
`D:\claude_code\estimador_peso_bovino` (fuera de OneDrive), y los ~4,6 GB de
`C:\Users\Luciano\bovino-data\` ya se copiaron a `modelo-peso/data/` (se dejó el
original como respaldo, sin borrar). El repo de GitHub ya existe y está pusheado.

**`.gitignore` de `modelo-peso/` ya creado**, ignora: `data/` (el dataset), `*.pt`
(pesos preentrenados y entrenados, incluido `yolo11n-seg.pt` que ya estaba suelto en la
raíz — ultralytics los re-descarga solo la primera vez que hace falta), `runs/`
(salida de `entrenar_pose.py`) y `__pycache__/`.

**Ojo con un detalle de Ultralytics ya resuelto**: `pose_dataset.yaml` tiene un campo
`path` que YOLO resuelve **contra el directorio de trabajo desde donde se ejecuta el
script, no contra la ubicación del propio yaml** (se confirmó leyendo el código fuente
de `ultralytics.data.utils.check_det_dataset`). Por eso `entrenar_pose.py` hace
`os.chdir(AQUI.parent)` antes de entrenar, y `path:` en el yaml es `data/pose` (relativo
a `modelo-peso/`, no a `src/`). Si se reescribe ese script sin el `chdir`, el
entrenamiento puede fallar en silencio buscando el dataset en el lugar equivocado.

### Resultados de la investigación (medidos, no supuestos)

Validación siempre con `GroupKFold` por `animal_id` (nunca por imagen, para que las dos
orientaciones —normal y espejada— de un mismo animal no queden repartidas entre train
y test).

| Entrada | MAPE | Nota |
|---|---|---|
| Baseline tonto (predice la mediana, ignora la imagen) | 14,90% | **Ya cumple el objetivo del acta** — ver advertencia abajo |
| Descriptores de silueta completa (71 features, sin escala) | 17-18% | **Descartado**: R² negativo, peor que el baseline |
| Fórmula de Schaeffer sobre medidas reales | 5,72% | Piso clásico, sin entrenar |
| Ridge/Boosting sobre medidas reales en cm | 2,59-4,43% | Techo con medidas perfectas, no alcanzable en producción sin cinta métrica |
| **Ratios entre medidas reales (sin cm), 15 particiones aleatorias** | **5,53% ± 0,48** | **El enfoque elegido** — techo teórico con keypoints "perfectos" |
| Control negativo: ratios mezclados al azar respecto al peso | 19,37% | Confirma que la señal de los ratios es real, no un artefacto |

**Advertencia importante sobre el criterio de aceptación**: en el rango de pesos de
este dataset (341-644 kg, todos adultos), un modelo que ignora la foto y predice
siempre la mediana ya da 14,90% de MAPE — por debajo del objetivo del 15% del acta.
Hay que reportar siempre la *mejora sobre ese baseline*, no el MAPE aislado, y el
dataset propio que recolecte el equipo tiene que incluir terneros para que la métrica
discrimine de verdad. Se propuso reforzar el criterio de aceptación de TP1/TP2 con esta
salvedad (pendiente, ver checklist de la reunión con el equipo).

**Por qué falló la silueta completa**: se diagnosticó (no se supuso) mirando las
máscaras — son excelentes, pero la pose del animal (cabeza arriba vs. pastando, patas
abiertas o juntas) domina cualquier descriptor de contorno y tapa la señal de peso.
También se descartó que fuera un problema de escala: el tamaño en píxeles no
correlaciona con ninguna medida real (r < 0,25), lo que probó que la distancia de
captura del dataset no fue constante pese a lo que dice el paper.

**Actualización 15/09/2026**: la fila "Ratios entre medidas reales... 5,53%" de la
tabla de arriba es el techo teórico con medidas de experto en cm — no se sostiene con
keypoints marcados sobre fotos sin marcador de referencia (se midió r=0,035, ver
sección "Decisión de arquitectura — REVISADA" más arriba). El resultado real, con
marcador de referencia y a mayor escala (4.538 imágenes externas, todas las
distancias entre los 9 keypoints), es **15,8% MAPE** (13,1% en la banda 100-200 kg).
Detalle completo en `modelo-peso/README.md`.

### Pipeline de código, en orden de ejecución

```
src/preparar.py             -> descomprime el zip, inspecciona estructura real
src/segmentar.py            -> YOLO-seg (clase 'cow' de COCO, sin entrenar) -> mascaras/
src/exportar_etiquetado.py  -> exporta 72 imagenes laterales a 1024px para etiquetar
   (etiquetado manual en CVAT/Roboflow, 7 keypoints, ver etiquetado/README.md)
src/organizar_export_cvat.py -> arma data/pose/ (train/val) a partir del export de CVAT
src/entrenar_pose.py        -> entrena YOLO-pose sobre las etiquetas -> runs/pose7/
src/medidas_desde_keypoints.py --pesos runs/pose7/weights/best.pt
                             -> keypoints -> ratios -> MAPE (mezcla animales vistos y no vistos)
src/evaluar_holdout.py      -> MAPE honesto: separa animales que el modelo de pose
                                nunca vio, evita inflar el numero por memorizacion
```

`src/features.py`, `src/unir.py` y `src/entrenar.py` corresponden al camino de silueta
completa (descartado) y al de medidas en cm (usado solo para el techo teórico). No van
al pipeline de producción, pero **no borrarlos**: son la evidencia del experimento que
justifica por qué se eligió el camino de keypoints (útil para la defensa si preguntan
por qué no se usó el enfoque más simple).

### Estado al 15/09/2026 — qué falta

- [x] Pipeline completo corriendo end-to-end sobre el dataset de Mendeley
- [x] Segmentación: vista lateral 72/72 (100%), vista trasera 14/72 (19%, no se usa)
- [x] Hipótesis de silueta completa probada y descartada con evidencia
- [x] Hipótesis de ratios sin marcador probada y **descartada con evidencia** (r=0,035
      entre ratio en foto y ratio real; confirmado en dataset externo de 4.728 imágenes)
- [x] Hipótesis del marcador de referencia **confirmada con evidencia** (15,8% MAPE
      calibrado, 13,1% en la banda 100-200 kg, contra ~22% sin calibrar, en el
      dataset de AcmeAI)
- [x] Probado: más keypoints (9) + más datos (65x) SIN marcador — sigue en ~22%,
      confirma que el problema es de información de escala, no de cantidad de datos
- [x] Mejora de feature engineering: usar las 36 distancias posibles entre los 9
      keypoints en vez de 4 elegidas a mano (Boosting elige cuáles importan) —
      mejor que agregar una CNN nueva, mismo modelo, más información de entrada
- [x] Decisión de arquitectura revisada: sí hace falta marcador físico de referencia,
      sigue sin hacer falta ArUco/giroscopio/LiDAR (ver sección de arriba)
- [x] Etiquetado de las 72 imágenes propias completo (un solo integrante etiquetó todas)
- [x] `.gitignore`, `pyproject.toml`+`uv.lock` (entorno con `uv`) y repo en GitHub, todo listo
- [x] YOLO-pose entrenado (`runs/pose7/weights/best.pt`, 279 épocas, ~1h en CPU)
- [x] Evaluación honesta sin contaminar train/val (`src/evaluar_holdout.py`) — separa
      los animales que el modelo de pose ya vio de los que no, para no inflar el número
- [ ] **Bloqueante actual**: definir y conseguir el marcador físico (sticker) antes de
      salir a recolectar el dataset propio — si se arranca sin él hay que repetir la
      captura completa. Ver especificación en `modelo-peso/README.md`.
- [ ] Actualizar `etiquetado/README.md`: migrar a 9 landmarks + posición del sticker
- [ ] Reentrenar/reetiquetar el dataset propio con el marcador incluido desde el
      primer registro, y volver a medir el MAPE real con `evaluar_holdout.py`
      — **este es el número que se presenta en el hito del 30/09/2026**
- [ ] Cargar el riesgo nuevo sobre el marcador de referencia (ver arriba) y su plan de
      respuesta en TP5, y actualizar/retirar el riesgo viejo sobre ArUco si corresponde
- [ ] Decidir estrategia de `.gitignore`/Git LFS antes de mover el dataset al repo único
- [ ] Investigación exploratoria pendiente (no bloqueante, se puede retomar después):
      por qué el error sube mucho en terneros <100 kg en el dataset de AcmeAI (52% MAPE
      ahí, contra 14% en la banda 100-200 kg) — no se investigó la causa todavía
