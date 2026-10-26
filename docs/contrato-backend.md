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
- Mismo `id_evento` + misma imagen (SHA-256) → devuelve el mismo trabajo.
- Mismo `id_evento` + otra imagen → `409`.

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
