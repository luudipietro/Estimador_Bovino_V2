# Contrato con el backend C# (adaptador único)

El backend C# es el **único cliente** de este servicio. Cualquier cambio en este contrato requiere `/v2` y aviso al
responsable del adaptador.

## Convenciones
- **HTTPS + JSON.** Prefijo **`/v1`**.
- Autenticación con `Authorization: Bearer <token de servicio>`, rotado por ambiente. `/salud` es la única ruta
  abierta.
- Errores con **RFC 9457** (`application/problem+json`): `type`, `title`, `status`, `detail`, `codigo` y, si
  aplica, `regla`. Los códigos están en `errores.md`.
- `Idempotency-Key` = `id_evento` (el ID del evento de pesaje que genera el backend).
- Fechas en ISO 8601 con zona horaria. Pesos en kg con un decimal.

## Endpoints

| Método | Ruta | Uso | Respuesta |
|---|---|---|---|
| `GET` | `/salud` | Liveness/readiness: modelo cargado, versión e instancias. Ver [Cuerpos de `/salud`, `/v1/modelo` y `/v1/validaciones`](#cuerpos-de-salud-v1modelo-y-v1validaciones) | 200 / 503 |
| `POST` | `/v1/estimaciones` | 1–4 imágenes (multipart) **o** `referencias_imagen` (URLs firmadas de la allowlist), más `metadatos?` y `url_notificacion?`. El `id_evento` va en `Idempotency-Key` | **202** + `Location` (nuevo) / **200** + `Location` (repetido) |
| `GET` | `/v1/estimaciones/{id}` | Estado: `pendiente` · `procesando` · `completado` · `fallido` | 200 / 404 |
| `POST` | `/v1/validaciones` | Validación rápida de calidad sin estimar (formato, resolución, animal presente). Apoya RF-15 desde el backend. Multipart con **un** archivo en el campo `imagen` (*propuesto*) | 200 |
| `GET` | `/v1/modelo` | Versión, rango de validez y fecha de entrenamiento | 200 |

## `POST /v1/estimaciones`: formato del pedido

Encabezados: `Authorization: Bearer <token>` e `Idempotency-Key: <id_evento>` (obligatorios).

`Idempotency-Key` es el **único origen del `id_evento`** (*propuesto*): 1 a 128 caracteres ASCII imprimibles, sin
espacios. Si falta o no cumple eso, `400 solicitud_invalida` con `campos: ["Idempotency-Key"]`. Un `id_evento` dentro
del cuerpo (o como campo multipart) **no se acepta**: da `400 solicitud_invalida`, porque habría dos fuentes que
podrían contradecirse.

Hay **dos formas** de mandar las imágenes, mutuamente excluyentes: si llegan las dos, `422
cantidad_imagenes_invalida`.

**A. `multipart/form-data`**

| Campo | Tipo | Obligatorio | Notas |
|---|---|---|---|
| `imagenes` | archivo, repetible (1–4) | sí | El tipo real se detecta por *magic bytes*, no por extensión |
| `metadatos` | texto con un objeto JSON | no | R-26: se devuelve sin modificar y no altera la estimación |
| `url_notificacion` | texto (URL HTTPS) | no | Host en `HOSTS_CALLBACK_PERMITIDOS` |

**B. `application/json`**
```json
{"referencias_imagen": [{"url": "https://almacenamiento.example/…?firma=…", "sha256": "9f86d081…0a08"}],
 "metadatos": {"raza": "Brangus"},
 "url_notificacion": "https://backend.example/api/estimaciones/callback"}
```
`referencias_imagen`: 1–4 objetos `{url, sha256}`.
- `url`: HTTPS, con el host en `HOSTS_IMAGEN_PERMITIDOS`.
- `sha256`: SHA-256 del contenido de la imagen, **64 caracteres hexadecimales**. Se aceptan mayúsculas y minúsculas
  y el servicio las convierte a minúsculas, así que `AB…` y `ab…` son el mismo hash. Un valor mal formado se rechaza
  al recibir el pedido.
- Al descargar, el servicio **verifica** que el contenido coincida con el `sha256` declarado. Si no coincide, el
  trabajo queda `fallido` con `referencia_inaccesible`.

Con multipart el servicio calcula el SHA-256 de cada archivo recibido.

Respuesta **202**: encabezado `Location: /v1/estimaciones/{id}` y cuerpo
`{"id": "trb_01J…", "id_evento": "c3b1…", "estado": "pendiente"}`.
Si el `id_evento` ya existía con las mismas imágenes (en cualquier orden), la respuesta es **200** (*propuesto*; un
trabajo nuevo es 202) con el mismo `Location` y el cuerpo `{id, id_evento, estado}` del trabajo existente, con su
estado actual. Así el backend distingue "lo creé ahora" de "ya existía" sin consultar.

Tope del cuerpo (*propuesto*), medido sin leer el pedido completo:
- **JSON:** 64 KB. Si lo supera: `413 imagen_muy_grande` (el mismo código y la misma forma de rechazo que el
  multipart; el `detail` dice que es el cuerpo). No se agrega un código nuevo.
- **Multipart:** `4 × MAX_MB_IMAGEN` más 1 MiB de margen para los demás campos. Si lo supera: `413 imagen_muy_grande`.
  Se rechaza **antes de leer el cuerpo** cuando el pedido declara `Content-Length` mayor que el tope, y además se
  corta durante la lectura apenas lo acumulado lo supera, lo que cubre `Transfer-Encoding: chunked` y un
  `Content-Length` que miente. Una sola imagen mayor que `MAX_MB_IMAGEN` también es `413 imagen_muy_grande`.

## Errores 4xx: formato

`application/problem+json` (RFC 9457). El `type` es `https://bovinovision/errores/<codigo>` (identificador, no
tiene por qué resolver). Ejemplo:
```json
{"type": "https://bovinovision/errores/formato_no_admitido",
 "title": "Formato de imagen no admitido",
 "status": 415,
 "detail": "El archivo no es JPEG, PNG ni HEIC.",
 "codigo": "formato_no_admitido",
 "request_id": "7f1c…"}
```

### `solicitud_invalida` (400, *propuesto*)

El pedido está mal formado y no llega a evaluarse el contenido: falta o es inválido `Idempotency-Key`; el `Content-Type`
no es multipart ni JSON; JSON roto o que no es un objeto; `metadatos` que no es un objeto JSON; un campo desconocido;
`sha256` que no son 64 hexadecimales; `url` vacía o de más de 2048 caracteres. Además lleva
un campo propio, **`campos`**: lista de strings con los **nombres** de lo que está mal (nunca los valores recibidos,
que pueden ser URLs firmadas o datos del cliente), cada uno acotado a 64 caracteres.
```json
{"type": "https://bovinovision/errores/solicitud_invalida",
 "title": "Solicitud inválida",
 "status": 400,
 "detail": "El pedido está mal formado.",
 "codigo": "solicitud_invalida",
 "campos": ["referencias_imagen.0.sha256"],
 "request_id": "7f1c…"}
```
En los errores de validación del JSON, los nombres usan puntos para anidar (`referencias_imagen.0.sha256`). Una URL
sintácticamente inválida, que no sea `https`, con credenciales o fuera de la allowlist **no** es `solicitud_invalida`:
es `422 referencia_no_permitida`.

### Otros errores de este endpoint
- `413 imagen_muy_grande`: ver el tope del cuerpo más arriba.
- `503 cola_llena`: lleva `Retry-After: 5` (segundos, *propuesto*).
- `401 no_autenticado`: lleva `WWW-Authenticate: Bearer`.

## Cuerpos de `/salud`, `/v1/modelo` y `/v1/validaciones`

**`GET /salud`** (sin autenticación, `Cache-Control: no-store`):
```json
{"estado": "ok", "modelo_cargado": true, "version_modelo": "2026-09-25 / …", "instancias": 4, "error": null}
```
`estado` es `ok` o `no_disponible` (con 503). `instancias` es la cantidad de copias del modelo cargadas (una por hilo
de inferencia). **`error` es `null` en esta versión** en el único caso en que el servicio responde: la app no arranca
si falla la verificación de hashes o la carga del modelo, así que no existe un estado "cargando" ni un 200 con error.
El 503 con `modelo_cargado: false` y un texto en `error` queda solo como defensa si el contenedor no está armado.

**`GET /v1/modelo`**:
```json
{"version": "2026-09-25 / …", "rango_validez_kg": [36.0, 621.0],
 "mape_test": 15.6, "ic95_test": [14.0, 16.3], "fecha_entrenamiento": null}
```
`version` y `rango_validez_kg` salen del `config.json` del modelo (R-20, R-23). `mape_test` e `ic95_test` son `null`
si ese archivo no los trae. `fecha_entrenamiento` se toma del `config.json` **solo si está**; si no, es `null` (el
`config.json` de v1 no la trae). No se inventa ni se deduce de la fecha del archivo.

**`POST /v1/validaciones`** (*propuesto*): `multipart/form-data` con **exactamente un** archivo en el campo `imagen`
y ningún otro campo; otra cosa es `400 solicitud_invalida`. No crea trabajo ni usa `Idempotency-Key`. Responde 200:
```json
{"apta": false, "motivos": ["sin_animal"], "advertencias": []}
```
Un archivo inválido (formato, corrupto, resolución) es un 4xx con su código; una foto válida pero de mala calidad es
`apta: false` con sus `motivos`. Si la validación supera su presupuesto de tiempo: `504 tiempo_excedido`.

## Arranque del servicio

El servicio **no arranca** si falla la verificación de hashes de los `.pt` contra `SHA256SUMS.txt` o la carga del
modelo: termina con error y `/salud` no llega a responder. Recién con las instancias cargadas empieza a aceptar
pedidos. El arranque con los pesos v1 en CPU tardó entre 14 y 16 s en la máquina de desarrollo; medirlo de nuevo en
homologación.

## Ciclo de vida del trabajo

Valores **propuestos**, a confirmar con el equipo (el TP8 no los define):

- **Retención:** el trabajo se conserva `TTL_TRABAJOS_S` (24 h) desde que termina. El backend C# es el dueño del
  evento de pesaje; este servicio no es un archivo.
- **`GET` sobre un trabajo vencido o inexistente:** `404 trabajo_no_encontrado`. No se distingue entre ambos
  casos.
- **Cola:** admite hasta `TAMANO_COLA` trabajos esperando. Si está llena, el `POST` responde `503 cola_llena` con
  `Retry-After: 5` (segundos, *propuesto*); el backend reintenta con el mismo `Idempotency-Key`.
- **Reinicio del servicio:** con el repositorio en memoria, los trabajos pendientes se pierden. El backend lo
  detecta por el `404` y reenvía con el mismo `id_evento`.

## Resultado `completado`
```json
{
  "id": "trb_01J…",
  "id_evento": "c3b1…",
  "estado": "completado",
  "peso_kg": 187.4,
  "confiable": true,
  "motivos": [],
  "version_modelo": "2026-09-25 / entrenado con AcmeAI-BMGF (Bangladesh)",
  "rango_validez_kg": [36.0, 621.0],
  "imagenes": [
    {"indice": 1, "peso_kg": 185.9, "confiable": true, "motivos": [],
     "confianza_deteccion": 0.94, "confianza_keypoint_minima": 0.88, "bovinos_en_la_escena": 1}
  ],
  "advertencias": ["la imagen 2 no trae metadata EXIF"],
  "metadatos": {"raza": "Brangus"},
  "duracion_ms": {"cola": 120, "inferencia": 1840, "total": 2050},
  "creado": "2026-10-25T12:00:10-03:00",
  "finalizado": "2026-10-25T12:00:12-03:00"
}
```

## Resultado `fallido`
```json
{"id": "trb_01J…", "id_evento": "c3b1…", "estado": "fallido",
 "codigo": "tiempo_excedido", "detalle": "La estimación superó el presupuesto de tiempo.", "reintentable": true}
```

## Idempotencia
- La **huella** del pedido se calcula sobre los SHA-256 de las imágenes **ordenados**. No depende ni de la URL ni
  de su firma: solo del contenido.
- Mismo `id_evento` + misma huella → devuelve el mismo trabajo, con **200** (un trabajo nuevo es 202). Un reintento con las mismas imágenes **en otro
  orden** también es el mismo pedido (RF-17: la sincronización offline puede reordenarlas), no un `409`.
- Mismo `id_evento` + otra huella → `409 id_evento_en_conflicto`.
- **Los índices siguen el orden del primer pedido.** "Imagen 1", "imagen 2" en `imagenes` y en `advertencias` se
  asignan según el orden en que llegaron la primera vez. Si un reintento trae otro orden, la respuesta es el
  trabajo original con sus índices originales; el segundo orden se ignora.

## Callback (si se envió `url_notificacion`)
- `POST` a un host de la allowlist (`https`, sin credenciales en la URL) con `Content-Type: application/json` y el
  mismo cuerpo que `GET /v1/estimaciones/{id}`.
- **Firma:** `X-BV-Firma: sha256=<hex>`, donde `<hex>` es el HMAC-SHA256, con el secreto compartido
  (`SECRETO_HMAC`), del texto `"<timestamp>" + "." + "<cuerpo exacto, en bytes>"`.
- **`X-BV-Timestamp`:** segundos Unix (UTC) en que se firmó, como texto. **Está dentro de lo firmado**: si solo se
  firmara el cuerpo, un callback capturado podría reenviarse con un timestamp nuevo y la firma seguiría valiendo.
  Se recalcula en cada reintento.
- **Lo que tiene que hacer el receptor (backend C#):** recalcular el HMAC sobre `<timestamp>.<cuerpo>` con los
  bytes recibidos (sin reserializar el JSON), compararlo **en tiempo constante** y rechazar timestamps fuera de una
  ventana (se sugiere ±5 minutos, a confirmar con quien implemente el adaptador). Como un reintento puede entregar
  dos veces el mismo resultado, además conviene deduplicar por `id_evento`.
- **Se considera entregado** con cualquier respuesta `2xx`. El servicio **no sigue redirecciones**: reenviaría el
  cuerpo firmado a otro host.
- Ante un fallo (no `2xx`, timeout o error de conexión) se reintenta con backoff exponencial: `INTENTOS_CALLBACK`
  intentos en total (3) con esperas de `ESPERA_BASE_CALLBACK_S` (1 s y 2 s). Si igual falla, **el resultado sigue
  disponible por `GET`**: el backend nunca depende solo del callback.
- El envío corre en una tarea aparte, después de guardar el estado final: no retrasa la cola ni cuenta dentro del
  presupuesto de tiempo.

## Presupuesto de tiempo (RNF-01)
Cada trabajo tiene un límite configurable (`PRESUPUESTO_ESTIMACION_S`, propuesto 45 s) para dejar margen al
backend dentro de los 60 s. Si se supera, el trabajo queda `fallido` con `tiempo_excedido`.

## Pendientes de confirmar con el equipo
- Si el estado de los trabajos va en Redis (no figura en el TP8) o en otro mecanismo.
- Si el backend envía la imagen por multipart o por URL firmada del almacenamiento privado.
