# Testing del servicio de estimación

Alineado con el **TP8 — Plan de Testing (PT-BovinoVision-2026-01)**.

## Niveles (TP8 §5)

| Nivel | En este servicio | Herramientas | Marca |
|---|---|---|---|
| **Unitarias** (caja blanca) | Dominio (agregación, R-20…R-26, transiciones de `Trabajo`) y casos de uso con fakes | `pytest`, `pytest-asyncio` | — |
| **Integración** | API ↔ casos de uso ↔ adaptadores; callback con `respx`; adaptador del modelo con pesos reales | `httpx.AsyncClient`, `respx` | `modelo` (pesos reales) |
| **Modelo** | MAPE por animal, por banda y contra el baseline de la mediana, sobre ≥ 100 registros de báscula | `scripts/evaluar_modelo.py` | — |
| **Rendimiento** | 50 trabajos concurrentes; p50/p95; % ≤ 60 s | `scripts/carga_concurrente.py` | `rendimiento` |
| **Seguridad** | Token ausente o inválido, allowlist de URLs, firma del callback, tamaño y tipo MIME | `pytest` | — |

La integración **backend C# → adaptador → API Python** la prueban en homologación los responsables del adaptador.
Este repo provee el contrato (`contrato-backend.md`) y los ejemplos que esa prueba usa.

## Trazabilidad

- Cada test que cubre un requisito lleva `@pytest.mark.requisito("RF-20")`.
- Si corresponde a un caso del Anexo B (*Casos de Prueba – BovinoVision.xlsx*), lleva también
  `@pytest.mark.caso("<ID>")`. **No inventar IDs de caso**: si no están, dejar solo `requisito`.
- Las reglas se nombran con su ID: `test_r20_peso_fuera_de_rango_no_es_confiable`.
- Las marcas se registran en `pyproject.toml`.

## Imágenes de prueba obligatorias (TP8 §5.1)

`tests/fixtures/imagenes/` debe tener al menos una de cada:

| Tipo | Resultado esperado |
|---|---|
| Lateral **válida** | `completado`, `confiable: true` |
| **Borrosa** | `completado`, `confiable: false`, `imagen_borrosa` (R-24) |
| **Sin animal** | `completado`, `confiable: false`, `sin_animal` |
| Animal **descentrado** pero completo | `completado`, `confiable: true` (R-25) |
| Animal **cortado** por el borde | `completado`, `confiable: false`, `animal_cortado` |
| **Formato no admitido** (`.gif`, `.bmp`) | `415`, `formato_no_admitido` |
| Archivo **corrupto** | `422`, `imagen_corrupta` |
| JPEG **sin EXIF** | `completado` con advertencia |

En los tests unitarios, el `ModeloFalso` devuelve lo esperado según el nombre del fixture. Con `-m modelo` se
verifica que el modelo real coincide.

## Casos obligatorios adicionales

- Mismo `id_evento` + misma imagen → mismo trabajo. Mismo `id_evento` + otra imagen → `409`.
- Trabajo que supera el presupuesto → `fallido`, `tiempo_excedido`, `reintentable: true`.
- Callback caído → el resultado sigue disponible por `GET`.
- Estimación fuera de rango → `confiable: false` (R-20), nunca un error.
- Ninguna respuesta de error contiene trazas ni nombres de clases.

## Severidad de defectos (TP8 §6 y §7)

| Nivel | Nombre | Ejemplo | Qué hacer |
|---|---|---|---|
| **1** | Crítica | Se pierde un evento de pesaje | Se **suspenden** las pruebas. Prioridad absoluta, con test de regresión. |
| **2** | Alta | La estimación nunca finaliza | Se **suspenden** las pruebas. Prioridad alta, con test de regresión. |
| **3** | Media | Resultado incorrecto no crítico | La prueba continúa. Se registra y se prioriza. |
| **4** | Baja | Texto desalineado o mensaje poco claro | La prueba continúa. Se registra. |

- Todo arreglo incluye **un test que reproduce la falla antes del fix** y cita el ID del registro
  *Defectos-Mejoras* en el commit.
- Tras reparar un defecto se corre la **regresión completa** (`uv run pytest` + `-m modelo`). Las pruebas del caso
  afectado se reanudan desde su inicio.
- Se suspende también ante una falla de entorno (servidor de pruebas, servicio de estimación o proveedor de nube
  no disponibles) o de hardware o red. Se reanuda cuando se repara.

## Criterios de terminación que dependen del servicio

- RF-19 a RF-22, RNF-01 y RNF-02 tienen **al menos un test** con su marca `requisito`.
- **MAPE < 20 %** sobre ≥ 100 registros con peso de báscula, particionando por animal y con mejora sobre el
  baseline de la mediana. Ver el estado del modelo en `requisitos.md`.
- 50 estimaciones concurrentes dentro de los **60 s**.
- Cobertura ≥ 90 % en `dominio/` y `aplicacion/`.

## Calendario (TP8 §11)

| Actividad | Fechas |
|---|---|
| Pruebas unitarias | 23/10 → 29/10 |
| Pruebas de integración incrementales | 29/10 → 31/10 |
| Evaluación de usabilidad (SUS) | 31/10 → 01/11 |
| Reporte general de pruebas | 02/11 |

## Entorno de homologación (TP8 §10)

El servicio corre como **servicio independiente con el modelo versionado**, separado del de desarrollo y por
HTTPS. Token, secreto HMAC, allowlist, presupuesto y versión del modelo llegan por variables de entorno. **Nunca**
apuntar el ambiente de desarrollo al de homologación.

## Rendimiento

- La inferencia tarda 1–3 s por foto en CPU. Con 4 fotos y 50 animales concurrentes, **medir** antes de afirmar
  que se cumple RNF-01. Si no alcanza, se escala (más réplicas), no se reducen las validaciones.
- Cada resultado registra la duración por etapa (`cola`, `descarga`, `inferencia`, `total`) para el indicador
  "% de estimaciones ≤ 60 s".
