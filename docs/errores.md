# Catálogo de errores y motivos de "no sé" (RF-22)

Los códigos son **`snake_case` estables**. El backend C# los mapea a sus propios mensajes: no se renombran ni se
reutilizan sin versionar la API (`/v2`).

Hay tres clases de resultado y **no se mezclan**.

## 1. Entrada inválida: se rechaza al recibir (4xx)

| Código | HTTP | Cuándo |
|---|---|---|
| `formato_no_admitido` | 415 | El tipo real del archivo (por *magic bytes*) no es JPEG, PNG ni HEIC |
| `imagen_corrupta` | 422 | No se puede decodificar |
| `imagen_muy_grande` | 413 | Supera 25 MB (`MAX_MB_IMAGEN`) **o el límite de píxeles configurado** (`MAX_PIXELES`), incluida una bomba de descompresión (`DecompressionBombError` de Pillow). No se agrega un código nuevo: sería un cambio de contrato |
| `resolucion_insuficiente` | 422 | Menos de 0,75 MP (`RESOLUCION_MINIMA_MP`, valor propuesto a confirmar con el equipo; `verificar_fotos.py` usa 3 MP) |
| `cantidad_imagenes_invalida` | 422 | Menos de 1, más de 4, o imágenes y referencias en el mismo pedido |
| `id_evento_en_conflicto` | 409 | Mismo `id_evento` con otra imagen |
| `referencia_no_permitida` | 422 | URL de imagen o de callback fuera de la allowlist |
| `no_autenticado` | 401 | Token ausente o inválido |
| `trabajo_no_encontrado` | 404 | El `id` no existe o ya venció (`TTL_TRABAJOS_S`) |
| `cola_llena` | 503 | Hay `TAMANO_COLA` trabajos esperando. Lleva `Retry-After` (5 s). *(propuesto, a confirmar)* |
| `solicitud_invalida` | 400 | Pedido mal formado: falta `Idempotency-Key`, JSON roto, `metadatos` que no es un objeto, `sha256` o URL con mal formato, campo desconocido o cuerpo JSON de más de 64 KB. Incluye `campos` con los nombres (no los valores). Es el código también de una ruta inexistente (404) o un método no permitido (405). *(propuesto, a confirmar)* |

## 2. El modelo no puede estimar con confianza: **no es un error**

El trabajo termina `completado`, con `confiable: false` y la lista `motivos` (R-21).

| Motivo | Cuándo | Regla |
|---|---|---|
| `sin_animal` | El detector no encuentra ningún bovino | R-25 |
| `deteccion_dudosa` | Confianza de detección < `CONF_DETECCION_MIN` (0,35) | R-25 |
| `partes_tapadas` | Algún keypoint con confianza < `CONF_KEYPOINT_MIN` (0,50) | R-25 |
| `animal_cortado` | El animal toca el borde de la foto original | R-25 |
| `imagen_borrosa` | Varianza del Laplaciano sobre el recorte por debajo del umbral | R-24 |
| `fuera_de_rango` | El peso estimado cae fuera del rango de `config.json` | R-20 |

> **R-24:** el umbral de la varianza del Laplaciano **no está definido**. Se calibra con las imágenes válidas y
> borrosas del set de prueba y se deja configurable. No inventar un número. Sin umbral configurado
> (`UMBRAL_BORROSIDAD` vacío) la regla **no se aplica**: ninguna foto se marca `imagen_borrosa`. El adaptador mide
> y el dominio decide (ver `reglas-dominio.md`).

Un animal **descentrado pero completo** es válido.

## 3. Falla interna: el trabajo queda `fallido`

| Código | `reintentable` | Cuándo |
|---|---|---|
| `modelo_no_disponible` | sí | El modelo no cargó o no responde |
| `tiempo_excedido` | sí | Superó `PRESUPUESTO_ESTIMACION_S` |
| `referencia_inaccesible` | sí | No se pudo obtener la imagen: falló la descarga de la URL firmada, la imagen subida ya no estaba en el almacén temporal, o su contenido no coincide con el `sha256` declarado |
| `error_interno` | no | Cualquier otro caso |

## Reglas de salida
- Nunca se exponen trazas, rutas ni nombres de clases. El detalle va al log con `request_id` e `id_evento`.
- Los 5xx y los trabajos `fallido` devuelven un `detalle` genérico y legible.
- Todo error de la API sale como `application/problem+json` (RFC 9457) con `type`, `title`, `status`, `detail`,
  `codigo`, `regla` (si aplica) y `request_id`; el mismo `request_id` va en el encabezado `X-Request-ID`.
- Las fallas internas solo llegan a una respuesta HTTP desde un endpoint síncrono (`/v1/validaciones`):
  `modelo_no_disponible` 503, `tiempo_excedido` 504, `referencia_inaccesible` 502 y `error_interno` 500. En un
  trabajo asincrónico dejan el estado `fallido`, no un error HTTP.
- Agregar un código nuevo es un cambio de contrato: ver `contrato-backend.md`.
