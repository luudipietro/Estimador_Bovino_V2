# BovinoVision · Servicio de estimación de peso (API Python)

Sus reglas aplican solo al servicio Python de API REST.

BovinoVision estima el **peso de bovinos a partir de una foto lateral tomada con un smartphone**, sin balanza, y
lleva el historial y la analítica del rodeo, para **productores ganaderos y operarios de campo en Argentina**.
Este archivo guía a cualquier agente de IA que escriba o revise el **servicio de estimación**: la API en Python que
recibe una imagen, localiza los puntos anatómicos del animal y devuelve el peso estimado en kg (o un "no sé" con
motivo). También cubre la CLI de curación del dataset propio.

> Proyecto Final de Ingeniería en Sistemas de Información, UTN-FRT, comisión 5K1 (13/05/2026 → 18/11/2026).
> Responder, comentar y documentar siempre en **español**. Versión 3.0, alineada con el TP8 (Plan de Testing).

## Lugar del servicio en el sistema

```
Frontend web (React + Vite, offline) ──► Backend de negocio (C#) ──adaptador único──► Servicio de estimación (Python) ◄── ESTE REPO
                                                  │
                                                  ▼
                                    PostgreSQL (Neon) + almacenamiento privado de imágenes
```

- Este servicio es **stateless respecto del negocio**: no conoce animales, lotes ni usuarios. Solo guarda el estado
  temporal de los trabajos de estimación.
- Lo consume **un único cliente**: el adaptador del backend C#. La autenticación es de servicio a servicio.
- Animales, lotes, eventos de pesaje inmutables, usuarios, auditoría, GMD, alertas y notificaciones al usuario
  **son del backend C#**. Si una tarea los requiere, se pregunta antes.

## Stack

- **Lenguaje:** Python **3.13** (`requires-python = ">=3.13,<3.14"` y `.python-version` con `3.13`). Es la versión
  con la que se entrenó el modelo y se verificaron las wheels de ROCm. No pasar a 3.14 durante el testing.
- **Framework:** **FastAPI** + **Uvicorn** · **Pydantic v2** (solo en `api/` y en la configuración).
- **Modelo:** `estimador/` (`EstimadorPeso`), con versiones fijas: `torch==2.14.0` (CPU), `torchvision==0.29.0`,
  `ultralytics==8.4.144`, `timm==1.0.29`. Cadena: detector COCO → YOLO-pose de 9 keypoints → ConvNeXt-Tiny.
- **Trabajos:** detrás del puerto `RepositorioTrabajos` (memoria en dev y tests; Redis con TTL en homologación,
  a confirmar con el equipo).
- **Imágenes:** el servicio no es dueño del almacenamiento: recibe, procesa y descarta.
- **Metadata:** Pillow, pillow-heif y exiftool opcional (`AccelerationVector`).
- **Tests:** pytest, pytest-asyncio, pytest-cov, httpx, respx. **Calidad:** ruff y mypy. **Entorno:** uv (el
  `uv.lock` se versiona).
- **Versionado:** el código va en **Git (GitHub)**. Los documentos académicos (TP1–TP8 y los xlsx de pruebas) se
  versionan en **SVN** (TP8 §5.5) y no se mueven al repo.

## Comandos

> El servicio vive en `servicio-estimacion/` y la CLI del dataset en `herramientas/dataset/`. Si una carpeta
> propuesta todavía no existe, crearla siguiendo `docs/arquitectura.md`.

### Servicio (`servicio-estimacion/`)
- `uv sync` — instala las dependencias.
- `python ../estimador/descargar_modelos.py` — descarga y verifica los pesos. **Obligatorio la primera vez y tras
  cada cambio de versión del modelo.**
- `uv run uvicorn bv_estimacion.api.main:create_app --factory --reload --port 8001` — **arranca el servidor en
  local** (docs en `http://localhost:8001/docs`).
- `uv run pytest` — **tests unitarios y de API. Deben pasar antes de cada commit.** Excluye las marcas `modelo` y
  `rendimiento`.
- `uv run pytest -m modelo` — tests con los pesos reales.
- `uv run pytest --cov=bv_estimacion --cov-fail-under=90` — cobertura, antes del PR.
- `uv run ruff check . && uv run ruff format --check . && uv run mypy src/bv_estimacion/dominio src/bv_estimacion/aplicacion`
  — **estilo y tipos, antes de cada PR.**
- `uv run python scripts/exportar_openapi.py` — regenera `openapi.json`, el contrato que lee el equipo del adaptador
  C#. Con `--comprobar` falla si el archivo está desactualizado (un test también lo verifica).
- `uv run python scripts/carga_concurrente.py --url <url> --n 50` — rendimiento (RNF-01, RNF-03).
- `uv run python scripts/evaluar_modelo.py --registros <csv> --fotos <carpeta>` — MAPE por animal y por banda
  contra el baseline de la mediana (RNF-02).
- `docker build -t bv-estimacion:$(git rev-parse --short HEAD) .` — **compila la imagen** (multi-stage, PyTorch CPU,
  sin los `.pt` adentro).

### CLI del dataset (`herramientas/dataset/`)
- `uv run bv-dataset verificar <carpeta>` · `ingresar <sesion>` · `cobertura` · `exportar --semilla 42`.

### Módulos existentes del repo
- `python verificar_fotos.py <carpeta>` — verificador original de metadata (su lógica migra a la CLI).
- `cd estimador && uv run uvicorn api:app --port 8000` — la API mínima actual, solo como referencia.
- `cd bovinovision && uv sync --extra cpu|rocm` — entorno de investigación. **Siempre `uv run --extra cpu|rocm`**:
  sin el extra se borra torch.

## Estructura del proyecto

- `servicio-estimacion/` — **la API de estimación**, en capas (`dominio/`, `aplicacion/`, `infraestructura/`,
  `api/`, `tests/`, `scripts/`). Es donde trabaja este agente. Detalle en `docs/arquitectura.md`.
- `herramientas/dataset/` — CLI de curación del dataset propio.
- `estimador/` — módulo de inferencia (`EstimadorPeso`) y `modelos/config.json`. Se **consume**, no se reescribe.
- `bovinovision/` — investigación vigente del modelo (scripts por etapa, `runs/*.csv`). **Evidencia: no se toca.**
- `modelo-peso/` — investigación anterior. **Congelada.**
- `docs/` — documentación para agentes (ver [Documentación](#documentación)).

**Regla de dependencias:** `api → aplicacion → dominio ← infraestructura`. El dominio no importa nada de las otras
capas, ni FastAPI, ni PyTorch, ni Pillow.

## Convenciones

- **Nombres en español**, coherentes con el repo: `peso_kg`, `confiable`, `motivo`, `id_evento`. `snake_case` para
  variables, funciones y módulos; `PascalCase` para clases. Casos de uso con verbo en infinitivo y método
  `ejecutar()`. Recursos REST en plural y sustantivos.
- **Tests** en `tests/`, espejando `src/`. Cada test de un requisito lleva `@pytest.mark.requisito("RF-20")`. Las
  reglas se nombran con su ID: `test_r20_peso_fuera_de_rango_no_es_confiable`. No inventar IDs de casos del Anexo B.
  Los tests unitarios y de API no usan red, GPU ni los `.pt`.
- **Errores (RF-22):** tres clases que no se mezclan. *Entrada inválida* → 4xx inmediato. *El modelo no puede
  estimar con confianza* → trabajo `completado` con `confiable: false` (**el "no sé" no es un error**). *Falla
  interna* → trabajo `fallido`. Códigos `snake_case` estables en `docs/errores.md`. Nunca se exponen trazas ni
  nombres de clases.
- **Validar toda entrada en el borde:** tamaño, tipo real por *magic bytes* y decodificabilidad, antes de encolar.
- **Asíncrono (RF-19):** el `POST` responde 202. Idempotencia por `id_evento`. Presupuesto de tiempo por trabajo
  para cumplir RNF-01.
- **Nada CPU-bound en el event loop:** la inferencia corre en un hilo con `CapacityLimiter`, dentro del adaptador.
- Value objects con `@dataclass(frozen=True, slots=True)`. Logs JSON con `request_id`, `id_evento` y
  `version_modelo`. En Windows, evitar caracteres fuera de cp1252 (`σ`, `±`) en consola.

## No hagas

### Límites duros
- No instales dependencias sin avisar y justificar. No cambies las versiones fijadas de torch/ultralytics/timm.
- No implementes lógica del backend de negocio (animales, lotes, eventos de pesaje, GMD, alertas, usuarios,
  notificaciones al usuario final).
- No te conectes a la base PostgreSQL de negocio (Neon), tampoco a `bovinovision_test`.
- No reintroduzcas marcadores, stickers ni ArUco.
- No devuelvas un peso como `confiable: true` fuera del rango de entrenamiento (R-20), ni conviertas el "no sé" en
  un error.
- No ajustes umbrales, rangos ni métricas para que un test de precisión "pase". Si RNF-02 falla, se informa el
  número real.

### Zonas prohibidas
- No toques `bovinovision/` ni `modelo-peso/`.
- No modifiques `estimador/estimador.py` ni el contrato de `EstimadorPeso.estimar()`: los ajustes van en el
  adaptador.
- No edites los entregables académicos (TP1–TP8 ni los xlsx de *Casos de Prueba*, *Defectos-Mejoras* y
  *Observaciones*) sin pedido explícito.

### Seguridad
- No subas `.env*`, tokens, URLs firmadas, `*.pt`, datasets ni fotos de campo.
- No aceptes URLs de imagen o de callback fuera de la allowlist (SSRF).
- No loguees imágenes, URLs firmadas completas, tokens ni coordenadas GPS.
- No dejes endpoints sin autenticación, salvo `/salud`.
- No persistas las imágenes más allá del procesamiento.

### Antipatrones
- No pongas lógica de negocio en routers ni en esquemas Pydantic.
- No uses `Any` ni `# type: ignore` sin un comentario que lo justifique.
- No uses `@app.on_event` (deprecado): usá `lifespan`.
- No guardes estado global de módulo (como el `_estimador` de `estimador/api.py`): usá `app.state` + `Depends`.
- No hardcodees el rango del modelo (`36–621`, `270`): se lee de `config.json`.

## Flujo de trabajo

- **Antes de una tarea no trivial, proponé un plan y esperá mi OK.** Es no trivial si toca más de una capa, el
  contrato con el backend C#, un código de error, una regla `R-xx` o los umbrales del modelo.
- **Una tarea a la vez.** Al terminar, decime qué cambiaste: archivos, requisitos (`RF/RNF`), reglas (`R-xx`) y
  tests agregados.
- **Si no estás seguro al 80 %, preguntá. No inventes** requisitos, umbrales, códigos ni IDs de casos de prueba.
- **Orden de implementación:** dominio (con tests) → caso de uso (con fakes) → adaptadores → router → test de API.
- **Una rama por tarea.** Commits en español con prefijo convencional y el requisito o regla:
  `feat(estimacion): marcar imágenes borrosas como no confiables (RF-15, R-24)`.
- **Revisión cruzada (TP8 §5.2):** todo PR lo aprueba un integrante distinto del autor.
- **Antes de commitear:** `uv run pytest`. **Antes del PR:** lint + mypy + cobertura y el checklist de abajo.
- **Defectos:** las severidades 1 y 2 suspenden las pruebas; el arreglo lleva un test que reproduce la falla y cita
  el ID del registro de defectos. Detalle en `docs/testing.md`.
- **Cambios de contrato o de modelo:** si cambia la respuesta de `/v1/estimaciones` o un código de error, se crea
  `/v2` y se avisa al responsable del adaptador C#. Los pesos nuevos se publican como release `modelos-vN`
  actualizando hashes y `config.json`; el servicio no debería cambiar de código.
- **Calendario:** hasta el 29/10, priorizar tests unitarios de dominio y aplicación sobre funcionalidades nuevas.

### Checklist de PR
- [ ] Se respeta la regla de dependencias entre capas y no hay lógica del backend C#.
- [ ] Cada requisito tocado tiene su test con `@pytest.mark.requisito(...)`, y cada regla, su test con `R-xx`.
- [ ] Los fixtures de imágenes cubren el caso nuevo, si aplica.
- [ ] Los errores salen en `problem+json` con un `codigo` del catálogo, sin detalles internos.
- [ ] Nada CPU-bound corre en el event loop y se respeta el presupuesto de tiempo.
- [ ] `ruff`, `mypy` y `pytest` pasan, con cobertura ≥ 90 % en dominio y aplicación.
- [ ] Si se arregló un defecto: test de regresión y cita del ID del registro.
- [ ] No se agregaron dependencias ni se cambiaron versiones del stack de ML sin avisar.
- [ ] Si cambió el contrato o el catálogo de errores: `/v2` y aviso al responsable del adaptador C#.

## Documentación

### Leer según la tarea (en `docs/`)
- [`docs/requisitos.md`](../docs/requisitos.md) — RF-19 a RF-22 y RNF-01 a RNF-03, y **el estado del modelo** (no
  extrapola; RNF-02 no pasa con el modelo actual sobre datos de otro rango). Leer antes de tocar precisión.
- [`docs/contrato-backend.md`](../docs/contrato-backend.md) — endpoints, resultados, idempotencia y callback. Leer antes
  de tocar cualquier ruta o respuesta.
- [`docs/errores.md`](../docs/errores.md) — catálogo de códigos y motivos de "no sé". Leer antes de agregar o cambiar
  un error.
- [`docs/reglas-dominio.md`](../docs/reglas-dominio.md) — reglas R-01 a R-26 y la CLI del dataset. Leer antes de tocar
  dominio o la herramienta de ingesta.
- [`docs/testing.md`](../docs/testing.md) — niveles, imágenes de prueba obligatorias, severidades, criterios y
  cronograma del TP8. Leer antes de escribir tests.
- [`docs/arquitectura.md`](../docs/arquitectura.md) — capas, modelo de dominio y ejemplos de código por capa. Leer
  antes de crear módulos.

### Del repositorio
- [`RECOLECCION_DATASET.md`](https://github.com/luudipietro/Estimador_Bovino_V2/blob/maximo/RECOLECCION_DATASET.md) —
  fuente de las reglas del dataset.
- [`estimador/README.md`](https://github.com/luudipietro/Estimador_Bovino_V2/blob/maximo/estimador/README.md) —
  contrato de `EstimadorPeso`, el "no sé" y la publicación de pesos.
- [`bovinovision/CLAUDE.md`](https://github.com/luudipietro/Estimador_Bovino_V2/blob/maximo/bovinovision/CLAUDE.md) —
  experimentos y trampas de los datos.
- [`COMO_LLEGAMOS.md`](https://github.com/luudipietro/Estimador_Bovino_V2/blob/maximo/COMO_LLEGAMOS.md) — historia de
  las decisiones.

Si dos fuentes se contradicen, manda el orden de esta lista: `docs/` → repositorio → el resto.

### Del proyecto (en SVN; el agente no las lee ni las edita)
TP1 Acta · TP2 Alcance y matriz de trazabilidad · TP5 Riesgos · TP6 Plan de gestión · **TP7 ERS (IEEE 830)** y casos
de uso · **TP8 Plan de Testing** · *Casos de Prueba – BovinoVision.xlsx*, *Defectos-Mejoras* y *Observaciones y
sugerencias*. Si una tarea depende de su contenido y no está en `docs/`, pedirlo.

### Datos de prueba
- Dataset público de Mendeley (DOI [10.17632/h2s22wr5py.2](https://doi.org/10.17632/h2s22wr5py.2)), como
  complemento durante el desarrollo (TP8 §5.1).
- Asesoramiento del Ing. zootecnista Cristian Ávila y del establecimiento colaborador (Cabaña Budeguer, Cachi Yaco).

### Licencias
Ultralytics es **AGPL-3.0**. Depth Pro (solo investigación) tiene licencia no comercial.
