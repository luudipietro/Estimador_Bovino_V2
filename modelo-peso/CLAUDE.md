# Contexto para trabajar en este repo

Módulo 2 de la materia: estimar el peso de un bovino a partir de una foto.
El objetivo del acta es **MAPE < 15%**, con la salvedad —medida, no supuesta— de
que un baseline tonto que predice siempre la mediana ya da 14,90% en el dataset
de Mendeley y 22,5% en el de AcmeAI. **Siempre reportar la mejora sobre el
baseline, nunca el MAPE solo.**

Historia completa de hipótesis probadas y descartadas: `README.md`. Este archivo
es solo lo operativo.

## Entorno

```bash
uv sync
```

Todo se corre **parado en la raíz del repo**, no adentro de `src/`:

```bash
uv run python src/<script>.py
```

`pyproject.toml` fija `torch` desde el índice CPU y `environments = ["sys_platform == 'win32'"]`.
Si alguien trabaja en una máquina con GPU y quiere CUDA, que cambie el índice de
torch en su copia — **no commitear ese cambio**, rompe el lock del resto.

## Los dos datasets (ninguno versionado, los dos se bajan a mano)

### 1. Mendeley — 72 bovinos, el "propio"

72 bovinos de Mongolia Interior, vista lateral y trasera, con keypoints, medidas
morfométricas en cm **y peso corporal**. Licencia CC BY 4.0.

1. https://data.mendeley.com/datasets/h2s22wr5py/2 → botón **Download All** (2,29 GB)
2. Guardarlo como `data/h2s22wr5py-2.zip`
3. `uv run python src/preparar.py`

Mendeley está detrás de Cloudflare: la descarga no se puede automatizar.

### 2. AcmeAI / BMGF — 4.728 fotos, el de validación

`www.acmeai.tech Dataset - BMGF-LivestockWeight-CV` — fotos reales de campo en
Bangladesh, con peso de báscula, 9 keypoints anatómicos etiquetados
profesionalmente, máscaras de segmentación **y una calcomanía de referencia ya
segmentada en las máscaras**. Es el dataset que permite probar hipótesis a escala
sin salir a fotografiar.

Se descarga del portal de Acme AI (`acmeai.tech`, sección de datasets públicos;
publicado bajo el proyecto BMGF-LivestockWeight-CV). **Va en la carpeta padre del
repo**, no adentro — ver `config.DATASET_ACMEAI`:

```
C:\claude_code\
├── estimador_peso_bovino\          <- este repo
└── www.acmeai.tech Dataset - BMGF-LivestockWeight-CV\
    ├── Vector\B3\Side\data\COCO_Side.json
    ├── Vector\B4\Side\data\coco_b4_side.json
    ├── Pixel\B3\annotations\*___fuse.png
    └── Pixel\B4\Side\annotations\*___fuse.png
```

Estructura relevante, por si hay que reubicar archivos: los keypoints vienen en
COCO (`Vector/`), las máscaras en PNG RGB (`Pixel/`), y el sticker está pintado
de un color fijo por batch (`STICKER_RGB` en `src/validar_metodo_acmeai.py`).
Los batches B3 y B4 comparten los mismos 9 landmarks pero **en orden distinto
dentro del json** — por eso todo se mapea por nombre, nunca por índice.

## Qué hace cada script

| Archivo | Qué hace |
|---|---|
| `src/preparar.py` | Descomprime el zip de Mendeley e inspecciona la estructura |
| `src/segmentar.py` | YOLO-seg sobre cada imagen → máscara binaria |
| `src/features.py` | 71 descriptores de forma (enfoque descartado, ver README) |
| `src/entrenar.py` | GroupKFold por animal + Ridge/Boosting + MAPE por franja |
| `src/exportar_etiquetado.py` | Prepara las 72 imágenes para etiquetar |
| `src/organizar_export_cvat.py` | Acomoda el export de CVAT al layout de YOLO |
| `src/entrenar_pose.py` | YOLO-pose sobre los landmarks propios |
| `src/evaluar_holdout.py` | MAPE real end-to-end, sin contaminar train/val |
| `src/medidas_desde_keypoints.py` | keypoints → distancias → ratios → peso |
| `src/validar_metodo_acmeai.py` | Valida el método **con sticker** sobre AcmeAI (15,8%) |
| `src/validar_profundidad_monocular.py` | Valida la alternativa **sin sticker**, con profundidad monocular métrica |

## Reglas que no se negocian

**Partición por animal, nunca por imagen.** `GroupKFold(groups=animal_id)`. Si
dos fotos del mismo animal caen en train y test, el modelo lo memoriza y el MAPE
reportado es mentira. Todos los scripts de evaluación ya lo hacen; cualquier
script nuevo también tiene que hacerlo.

**Contra el baseline, siempre.** Un MAPE suelto no dice nada en este rango de
pesos. Reportar el baseline de la mediana al lado, en la misma tabla.

**Control negativo cuando el resultado sorprende.** Mezclar las features al azar
y verificar que el MAPE se va al demonio. Ya salvó al proyecto una vez
(ver README, sección de ratios).

**Franjas de peso.** El promedio global esconde que el modelo anda bien en
100-200 kg y mal en los extremos. Reportar siempre desagregado.

**Nada de acentos en comentarios ni docstrings de `.py`.** Convención del repo,
por problemas de encoding en Windows. En los `.md` sí van.

## Dónde está el estado del proyecto

`README.md`, sección "Estado" (checklist) y las secciones fechadas. Cada tanda de
experimentos se documenta ahí con fecha, incluidas las hipótesis que se
descartaron y por qué — eso es la mitad del valor del informe final.
