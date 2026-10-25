# Reglas de dominio

Cada regla tiene un ID (`R-xx`) para citarla en commits, PR y nombres de test. La numeración se conserva de la
versión 1 para no romper referencias.

## Estimación (servicio)

- **R-20 — El modelo no extrapola.** Fuera del rango de `config.json`, `confiable = false` con motivo
  `fuera_de_rango`. El rango nunca se hardcodea.
- **R-21 — El "no sé" es un resultado válido:** el trabajo termina `completado`, no es un error.
- **R-22 — Varias fotos del mismo animal (1 a 4) se agregan** con la mediana de las confiables. Si ninguna es
  confiable, el agregado tampoco.
- **R-23 — Todo resultado registra la versión del modelo.**
- **R-24 — Imagen borrosa → no confiable** (`imagen_borrosa`). Umbral pendiente de calibrar (ver `errores.md`).
- **R-25 — Animal cortado por el borde, partes tapadas, detección dudosa o ausente → no confiable**, con los
  umbrales actuales del `EstimadorPeso` (`CONF_DETECCION_MIN = 0.35`, `CONF_KEYPOINT_MIN = 0.50`).
- **R-26 — Raza, edad y categoría son opcionales:** se reciben como `metadatos`, se devuelven sin modificar y
  **no** alteran la estimación. (Reemplaza a R-12 de la v1.)

## Dataset propio (herramienta CLI)

Fuente: [`RECOLECCION_DATASET.md`](https://github.com/luudipietro/Estimador_Bovino_V2/blob/main/RECOLECCION_DATASET.md).
Las referencias `§` apuntan a ese documento.

### Identidad y tiempo
- **R-01 — Toda foto pertenece a un animal identificado** por caravana. Sin ella no entra al dataset. *(§4, §5.1)*
- **R-02 — La identidad del animal es (establecimiento, caravana).** Un animal puede tener muchas sesiones. *(§2)*
- **R-03 — Foto y pesaje del mismo día** (fecha EXIF vs. fecha del pesaje, en la zona del establecimiento). Si no
  coinciden, el par se guarda pero **no es apto para entrenar**. *(§3.5, §5.2)*
- **R-04 — El peso de balanza es la verdad de terreno** y se registra con la precisión de la balanza. *(§4)*

### Integridad de la imagen
- **R-05 — El original es inmutable:** se guarda byte a byte y se calcula su SHA-256. *(§4, §5.5)*
- **R-06 — Se extrae y conserva el EXIF** y, en iPhone, el `AccelerationVector`. Hoy no se usa, pero sin ellos
  habría que repetir la recolección si hiciera falta recuperar la escala. *(§4, §4-bis)*
- **R-07 — Una foto sin metadata genera advertencia, nunca silencio** (WhatsApp, Telegram, PNG, capturas).
  Queda marcada `metadata_completa = false`. *(§4-bis)*
- **R-08 — Formatos:** JPEG preferido, HEIC aceptado (`pillow-heif`) y PNG con advertencia. *(§4-bis)*

### Datos de sesión y de animal
- **R-09 — Toda sesión registra:** establecimiento, fecha, altura aproximada de la cámara, operario, modelo de
  teléfono, precisión de la balanza y cambios de criterio. *(§4-bis)*
- **R-10 — El GPS solo se guarda con consentimiento** del establecimiento. *(§4-bis)*
- **R-11 — Obligatorios para el dataset:** `peso_kg`, `caravana`, `fecha_hora`, `raza`, `categoria` y `sexo`.
  Deseable: edad o dentición. Opcional: condición corporal. *(§4)*
- **R-13 — Catálogos cerrados:** `categoria ∈ {ternero, novillo, vaquillona, vaca, toro}` y
  `sexo ∈ {macho, hembra}`.

### Cobertura
- **R-14 — El avance se mide en animales distintos, no en fotos.** *(§2)*
- **R-15 — Bandas de peso:** `<150`, `150–250`, `250–350`, `350–450`, `450–550` y `>550` kg. La prioridad es la
  zona comercial (350–550). *(§2)*
- **R-16 — Mínimo de 40–50 animales por banda.** *(§2)*
- **R-17 — Metas:** mínimo viable 150–200, objetivo 300–400, ideal 500+. *(§2)*
- **R-18 — Entre 2 y 4 fotos laterales por animal y sesión.** *(§2, §5.4)*
- **R-19 — El split se hace siempre por animal**, nunca por foto. *(§5.1)*

## Referencia de cobertura

- Error esperable según animales distintos en la banda: ~20 → 27 % · ~50 → 22 % · ~215 → 15,6 % · ~340 → 11,8 %.
- Meta por defecto para 300 animales: `<150` 50 · `150–250` 60 · `250–350` 70 · `350–450` 70 · `450–550` 50 ·
  `>550` los que haya.
- Un animal que cruzó de banda entre sesiones cuenta en ambas, y se informa.

## Herramienta `bv-dataset`

- **Formato de entrada:** `AAAA-MM-DD_Establecimiento/fotos/<caravana>_<n>.<ext>` + `pesajes.csv`
  (`caravana, peso_kg, fecha, raza, categoria, sexo, edad, observaciones`). La caravana es **texto**
  (`0451` ≠ `451`). El CSV acepta separador `,` o `;` y decimal `.` o `,`.
- **Regex de nombre de foto:** `^(?P<caravana>[^_]+)_(?P<n>\d+)\.(jpe?g|heic|png)$` (sin distinguir mayúsculas).
- **`ingresar`:** hash → omitir duplicados → copiar el original intacto (R-05) → extraer metadata (R-06) → aplicar
  R-07, R-08 y R-10 → cruzar con el pesaje → evaluar R-03 → detectar huérfanos (R-01) → advertir según R-18.
  **Nada se descarta en silencio.**
- **`exportar`:** split por animal con semilla registrada y manifiesto con hashes. Ese manifiesto es la entrada de
  `scripts/evaluar_modelo.py` para RNF-02.
- Proteger la descompresión de ZIPs contra *zip slip* y bombas de descompresión.
