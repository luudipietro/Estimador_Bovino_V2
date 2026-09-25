# Módulo de estimación de peso — listo para integrar

Recibe una foto lateral de un bovino y devuelve un peso estimado. No necesita marcador, ni sensores, ni que el
animal pase por una manga.

## Lo que hay acá

```
estimador/
├── estimador.py      <- la clase EstimadorPeso (esto es lo que se integra)
├── api.py            <- API HTTP para que la app le pegue
├── pyproject.toml    <- dependencias
└── modelos/
    ├── detector_coco.pt   (5 MB)  encuentra al bovino en la escena
    ├── pose9.pt          (19 MB)  YOLO-pose: ubica 9 puntos anatómicos
    ├── peso_convnext.pt (106 MB)  CNN que estima el peso
    ├── config.json                versión y rango de validez del modelo
    └── SHA256SUMS.txt             para verificar las descargas
```

Son **130 MB en total**. Corre en CPU sin problema (1 a 3 segundos por foto); con GPU es más rápido.

> ### ⚠️ Los archivos de `modelos/` NO viajan en el repositorio
> `peso_convnext.pt` pesa 106 MB y **GitHub rechaza archivos de más de 100 MB**. Están en el `.gitignore`.
>
> **Si clonaste el repo y querés usar el módulo:**
> ```bash
> python descargar_modelos.py
> ```
> Baja los dos archivos y verifica que no se hayan corrompido. Si todavía no está publicada la release, el
> script te dice cómo conseguirlos a mano.
>
> `config.json` y los hashes sí se versionan, así que siempre se sabe qué versión de modelo corresponde a cada
> commit del código.

## Uso como librería

```python
from estimador import EstimadorPeso

est = EstimadorPeso("modelos/")          # carga los modelos una sola vez
resultado = est.estimar("vaca.jpg")      # acepta ruta, bytes o PIL.Image
```

Devuelve:

```python
{
  "peso_kg": 187.4,
  "confiable": True,
  "motivo": None,                         # por qué no es confiable, si no lo es
  "confianza_deteccion": 0.94,
  "confianza_keypoint_minima": 0.88,
  "keypoints": {"wither": [412.0, 233.5, 0.99], ...},
  "recorte": [120, 88, 940, 700],
  "resolucion": "4032x3024",
  "modelo": "2026-09-25 / entrenado con AcmeAI-BMGF (Bangladesh)"
}
```

## Uso como API

```bash
uv sync --extra api
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -F "foto=@vaca.jpg" http://localhost:8000/estimar
```

Hay documentación interactiva para probar desde el navegador en `http://localhost:8000/docs`, y un endpoint
`/salud` para monitoreo.

## Cómo funciona

1. La foto se reduce a 1024 px de lado mayor.
2. Un **detector genérico** busca todos los bovinos y se queda con el más grande (el de primer plano).
3. **YOLO-pose** ubica 9 puntos anatómicos sobre ese animal y se recorta dejando un margen.
4. Ese recorte va a 384×384 y una **CNN (ConvNeXt-Tiny)** predice el peso.
5. Se promedia la predicción con la de la imagen espejada, y se aplican las reglas de "no sé".

El paso 2 se agregó tras probar con fotos reales de corral argentino: sin él, el modelo de pose detectaba al
animal con confianza 0,05 (o no lo encontraba) porque fue entrenado con imágenes donde el bovino ocupa más de
la mitad del cuadro. Con el detector adelante, en esas mismas fotos la confianza sube a 0,87–0,91.

## Cuándo dice "no sé"

`confiable: False` cuando: no detecta ningún bovino, la detección es dudosa, hay partes del cuerpo tapadas o
fuera de cuadro, el animal aparece cortado por el borde, o el peso estimado cae fuera del rango con el que se
entrenó. En esos casos conviene pedirle otra foto al operario en vez de mostrar un número.

## ⚠️ Estado del modelo: sirve para integrar, NO para pesar de verdad todavía

Está entrenado con **4.540 fotos de bovinos de Bangladesh**, de 36 a 621 kg, con el **78% entre 100 y 200 kg**.

- Error medido en ese dominio: **15,6% de MAPE** (IC 95%: 14,0–16,3).
- **No extrapola**: nunca predice por encima de ~270 kg, aunque el animal pese 450.
- Probado sobre bovinos de otro país (Mongolia, 341–644 kg) **falla por completo** (61% de error).

En criollo: si le sacan una foto a un novillo argentino de 450 kg, va a decir alrededor de 250 kg. **Eso es
esperado y ya está medido**; no es un bug.

Sirve hoy para: integrar el flujo completo, probar la app, medir tiempos de respuesta, ajustar la interfaz,
definir qué hacer cuando el sistema dice "no sé".

Para que dé pesos reales hay que reentrenarlo con el **dataset argentino** (ver
[RECOLECCION_DATASET.md](../RECOLECCION_DATASET.md)). Cuando eso pase, **solo se reemplazan los archivos de
`modelos/` y se actualiza `config.json`**: la interfaz de este módulo no cambia, así que la app no se toca.

## Para el que publica los modelos (una sola vez)

Los `.pt` se distribuyen como adjuntos de una **release de GitHub**: admite hasta 2 GB por archivo, es gratis,
queda atado a una versión y no consume cuota de Git LFS.

### Opción A — desde la web (no requiere instalar nada)
1. Entrar a https://github.com/luudipietro/Estimador_Bovino_V2/releases/new
2. En *Choose a tag* escribir **`modelos-v1`** y elegir "Create new tag on publish".
   **El tag tiene que ser exactamente ese**, porque es el que usa `descargar_modelos.py`.
3. Título: `Modelos entrenados v1 (AcmeAI/Bangladesh)`
4. Arrastrar los tres archivos de `estimador/modelos/`: `pose9.pt`, `peso_convnext.pt` y `SHA256SUMS.txt`
5. Descripción sugerida: *MAPE 15,6% (IC 95%: 14,0–16,3). Sin marcador. No extrapola por encima de ~270 kg.*
6. **Publish release**

### Opción B — por línea de comandos
Requiere instalar GitHub CLI (`winget install GitHub.cli`, y después `gh auth login`):
```bash
gh release create modelos-v1 \
    estimador/modelos/pose9.pt \
    estimador/modelos/peso_convnext.pt \
    estimador/modelos/SHA256SUMS.txt \
    --title "Modelos entrenados v1 (AcmeAI/Bangladesh)" \
    --notes "MAPE 15,6% (IC 95%: 14,0-16,3). Sin marcador. No extrapola por encima de ~270 kg."
```

La URL ya está configurada en `descargar_modelos.py`, así que apenas se publique la release con el tag
`modelos-v1`, cualquiera del equipo corre `python descargar_modelos.py` y listo.

**Cuando se reentrene con el dataset argentino**: crear `modelos-v2` con los archivos nuevos, actualizar los
hashes y la `URL_BASE`, y subir la versión en `config.json`. Los modelos viejos quedan disponibles, así que
siempre se puede volver atrás y comparar.

### Si el repo es privado y alguien no tiene acceso
Alternativa rápida: subir los dos archivos a Drive/OneDrive compartido y pasar el link. Funciona, pero sin
versionado ni verificación automática. Por eso conviene la release.

## Qué falta para producción

| Pendiente | Comentario |
|---|---|
| Reentrenar con datos argentinos | Es lo único que bloquea el uso real |
| Autenticación de la API | Hoy está abierta; agregar token si se expone fuera de la red local |
| Registrar cada consulta | Guardar foto + resultado alimenta el dataset y permite auditar (Módulo 6) |
| Versionado de modelos | `config.json` ya lleva versión; falta decidir cómo se actualiza en el servidor |
| Contenedor Docker | Facilita el despliegue en la nube |
