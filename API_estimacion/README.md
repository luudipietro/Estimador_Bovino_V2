# API de estimación de peso bovino

Servicio HTTP, en su propio contenedor, que recibe **una foto lateral ya achicada** y devuelve el peso estimado.
Envuelve el módulo `estimador/` sin modificarlo: el build reutiliza `estimador/estimador.py` y su `uv.lock`.

```
app móvil ──(JPEG 1024 px, ~200 KB)──▶  POST /estimar  ──▶  detector → pose → CNN  ──▶  { peso_kg, confiable, ... }
```

## Levantarla

Requisito: que existan los modelos en `estimador/modelos/` (no viajan en git; correr `python descargar_modelos.py`
dentro de `estimador/` si faltan).

```bash
docker compose -f API_estimacion/docker-compose.yml up -d --build
curl http://localhost:8000/salud
```

La primera vez tarda unos minutos (descarga de torch/ultralytics). Los modelos **no están dentro de la imagen**: se
montan de solo lectura desde `estimador/modelos/`. Para reentrenar y publicar un modelo nuevo alcanza con reemplazar
esos archivos (y `config.json`) y reiniciar el contenedor, sin reconstruir la imagen:

```bash
docker compose -f API_estimacion/docker-compose.yml restart
```

Otro puerto: `PUERTO=9000 docker compose -f API_estimacion/docker-compose.yml up -d`.

## Contrato

### `POST /estimar`
`multipart/form-data` con la foto en el campo **`foto`** (JPEG o PNG).

**Qué debe hacer la app antes de enviar** (ejemplo ejecutable en [`cliente/enviar_foto.py`](cliente/enviar_foto.py)):

1. Aplicar la orientación EXIF a los píxeles. Al reducir se pierde el tag y el servidor ya no puede corregirla.
2. Reducir a **1024 px de lado mayor**, conservando la relación de aspecto (nunca agrandar).
3. Codificar JPEG calidad ~90. La foto pasa de varios MB a ~150–300 KB. (PNG también se acepta: ver la nota de abajo.)

> **¿Por qué 1024 y no menos?** El estimador reduce internamente toda foto a 1024 px de lado mayor, así que enviar
> 1024 es *equivalente* a enviar la original. Pero después recorta al animal y reescala (pose a 640 px, CNN a 384×384):
> si se achica a 640 o 384 *antes* de recortar, un animal que está lejos queda con muy pocos píxeles y se pierde
> precisión. 1024 es el mínimo seguro.

> **Decisión abierta: JPEG o PNG.** Achicar a 1024 px es *exactamente* equivalente a lo que hace el módulo (con PNG la
> API da el mismo peso que el módulo, diferencia 0,0 %, verificado). Lo que cambia el resultado es la **compresión
> JPEG**: sobre ~30 fotos reales movió el peso estimado ~2,5 % en promedio y hasta ~15 % en algunas (p. ej. 112 → 129 kg),
> y no mejora subiendo la calidad (q85 ≈ q95). Es sensibilidad del propio modelo a cambios imperceptibles (el recorte
> depende de los keypoints, que se mueven un poco), no un defecto del transporte; el entrenamiento usó JPEG q93, así que
> ninguna de las dos variantes es "la verdadera". Compromiso:
>
> | Formato | Peso por foto | Red vs. original (~4,5 MB) | Reproduce el módulo |
> |---|---|---|---|
> | JPEG q90 (default) | ~270 KB | −94 % | ±2,5 % típico, hasta ~15 % |
> | PNG | ~1,3 MB | −71 % | exacto |
>
> El cliente de ejemplo soporta ambos: `python cliente/enviar_foto.py foto.jpg --formato png`. Conviene que el equipo
> decida cuál usa la app; esta variabilidad es otro argumento a favor de reentrenar y de medir la repetibilidad del modelo.

Respuesta `200`:

```json
{
  "peso_kg": 187.4,
  "confiable": true,
  "motivo": null,
  "confianza_deteccion": 0.94,
  "confianza_keypoint_minima": 0.88,
  "bovinos_en_la_escena": 1,
  "keypoints": {"wither": [412.0, 233.5, 0.99], "...": []},
  "recorte": [120, 88, 940, 700],
  "resolucion": "1024x768",
  "modelo": "2026-09-25 / entrenado con AcmeAI-BMGF (Bangladesh)",
  "advertencias": [],
  "tiempo_ms": 1840
}
```

- Las claves están **siempre**, también cuando no se encuentra al animal (`peso_kg: null`, `confiable: false`, `motivo`).
- **`confiable: false` es una respuesta normal (200), no un error**: el modelo dice "no sé". Conviene pedir otra foto.
- `resolucion`, `keypoints` y `recorte` están en el espacio de la imagen **recibida** (reducida a 1024 px si hacía falta).
- `advertencias` incluye avisos de la API, por ejemplo que la foto llegó sin achicar, o que el peso está cerca del
  techo del modelo (≥ 240 kg).

> **`confiable: true` no garantiza un peso correcto.** El modelo se entrenó con bovinos de Bangladesh (78 % entre 100 y
> 200 kg) y **no extrapola**: un novillo de 450 kg sale cerca de 250 kg y aun así puede venir `confiable: true`. Esto
> está medido y es esperado hasta que se reentrene con el dataset argentino (ver `RECOLECCION_DATASET.md`).

Errores:

| Código | Cuándo |
|---|---|
| 400 | el archivo llegó vacío |
| 413 | pesa más de 5 MB (`MAX_MB`): probablemente no se achicó |
| 415 | no es una imagen válida, o no es JPEG/PNG (los JPEG "MPO" de algunos celulares se aceptan) |
| 422 | lado mayor menor a 640 px, o falta el campo `foto` |
| 500 | falla interna al procesar |

Una foto de más de 1024 px **se acepta** (el servidor la reduce) y devuelve un aviso en `advertencias`.

### `GET /salud`
`200` con `{"estado": "ok", ...}` solo cuando los modelos están cargados. Lo usa el healthcheck de Docker.

### `GET /info`
Contrato de imagen (lado mayor, formatos, tamaño máximo), versión del modelo, MAPE medido y limitaciones. El cliente
puede leerlo en vez de hardcodear los valores.

Documentación interactiva: `http://localhost:8000/docs`.

## Configuración (variables de entorno)

| Variable | Default | Qué hace |
|---|---|---|
| `MODELOS_DIR` | `/modelos` | carpeta con `config.json` y los `.pt` (en el contenedor es el volumen montado) |
| `MAX_MB` | `5` | tamaño máximo de la foto recibida |
| `PESO_SATURACION_KG` | `240` | desde qué peso se agrega la advertencia de saturación |
| `PUERTO` | `8000` | puerto publicado por docker compose |

## Pruebas

```bash
# validación de fotos (no necesita modelos ni contenedor)
python -m unittest discover -s API_estimacion/tests -t API_estimacion

# la API con foto achicada vs. el módulo con la foto original (necesita la API corriendo y el entorno de estimador/)
python API_estimacion/tests/comparar_con_modulo.py foto1.jpg foto2.jpg --url http://localhost:8000
python API_estimacion/tests/comparar_con_modulo.py foto1.jpg foto2.jpg --formato png   # debe dar diferencia 0,0 %
```

Tamaño de la imagen Docker: ~2,6 GB (torch + ultralytics + OpenCV; es el costo de este stack en CPU).

## Decisiones de diseño

- **Un solo worker y una consulta a la vez** (hay un lock): hay un único juego de modelos en CPU, y dos inferencias en
  paralelo solo se pelearían los núcleos y la RAM. Las consultas se encolan. Para más caudal, más réplicas del contenedor.
- **CPU**: las ruedas de torch son las de CPU, igual que el módulo. Una consulta tarda entre ~0,3 s (medido en una PC de escritorio) y ~3 s según la máquina.
- **Sin descargas en runtime**: `YOLO_AUTOINSTALL=False`, todo viene de la imagen y del volumen.
- **Licencia**: ultralytics es AGPL-3.0; tenerlo presente si se despliega como servicio de terceros.

## Pendientes

| Pendiente | Comentario |
|---|---|
| Autenticación | Hoy abierta. Agregar token (p. ej. header `X-API-Key`) antes de exponerla fuera de la red local |
| Registrar cada consulta | Guardar foto + resultado alimenta el dataset y la auditoría (Módulo 6) |
| Reentrenar con datos argentinos | Es lo único que bloquea el uso real; la API no cambia, solo los archivos de `modelos/` |
| Variante GPU | Imagen con torch CUDA para bajar la latencia |
| Release de modelos | `README` de `estimador/` omite `detector_coco.pt` entre los adjuntos de la release `modelos-v1`, pero `descargar_modelos.py` lo pide: confirmar que esté publicado |
