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
| `GET` | `/salud` | Liveness/readiness: modelo cargado, versión y dependencias | 200 / 503 |
| `POST` | `/v1/estimaciones` | 1–4 imágenes (multipart) **o** `referencias_imagen` (URLs firmadas de la allowlist), más `id_evento`, `metadatos?` y `url_notificacion?` | **202** + `Location` |
| `GET` | `/v1/estimaciones/{id}` | Estado: `pendiente` · `procesando` · `completado` · `fallido` | 200 / 404 |
| `POST` | `/v1/validaciones` | Validación rápida de calidad sin estimar (formato, resolución, animal presente). Apoya RF-15 desde el backend. | 200 |
| `GET` | `/v1/modelo` | Versión, rango de validez y fecha de entrenamiento | 200 |

## `POST /v1/estimaciones`: formato del pedido

Encabezados: `Authorization: Bearer <token>` e `Idempotency-Key: <id_evento>` (obligatorios).

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
Si el `id_evento` ya existía con la misma imagen, se devuelve el trabajo existente (con su estado actual).

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

## Ciclo de vida del trabajo

Valores **propuestos**, a confirmar con el equipo (el TP8 no los define):

- **Retención:** el trabajo se conserva `TTL_TRABAJOS_S` (24 h) desde que termina. El backend C# es el dueño del
  evento de pesaje; este servicio no es un archivo.
- **`GET` sobre un trabajo vencido o inexistente:** `404 trabajo_no_encontrado`. No se distingue entre ambos
  casos.
- **Cola:** admite hasta `TAMANO_COLA` trabajos esperando. Si está llena, el `POST` responde `503 cola_llena` con
  `Retry-After`; el backend reintenta con el mismo `Idempotency-Key`.
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
- Mismo `id_evento` + misma huella → devuelve el mismo trabajo. Un reintento con las mismas imágenes **en otro
  orden** también es el mismo pedido (RF-17: la sincronización offline puede reordenarlas), no un `409`.
- Mismo `id_evento` + otra huella → `409 id_evento_en_conflicto`.
- **Los índices siguen el orden del primer pedido.** "Imagen 1", "imagen 2" en `imagenes` y en `advertencias` se
  asignan según el orden en que llegaron la primera vez. Si un reintento trae otro orden, la respuesta es el
  trabajo original con sus índices originales; el segundo orden se ignora.

## Callback (si se envió `url_notificacion`)
- `POST` a un host de la allowlist con el mismo cuerpo que `GET /v1/estimaciones/{id}`.
- Firma en `X-BV-Firma: sha256=<HMAC del cuerpo>` con un secreto compartido, más `X-BV-Timestamp` para evitar
  replay.
- Ante un fallo se reintenta con backoff (3 intentos). Si igual falla, **el resultado sigue disponible por `GET`**:
  el backend nunca depende solo del callback.

## Presupuesto de tiempo (RNF-01)
Cada trabajo tiene un límite configurable (`PRESUPUESTO_ESTIMACION_S`, propuesto 45 s) para dejar margen al
backend dentro de los 60 s. Si se supera, el trabajo queda `fallido` con `tiempo_excedido`.

## Pendientes de confirmar con el equipo
- Si el estado de los trabajos va en Redis (no figura en el TP8) o en otro mecanismo.
- Si el backend envía la imagen por multipart o por URL firmada del almacenamiento privado.
