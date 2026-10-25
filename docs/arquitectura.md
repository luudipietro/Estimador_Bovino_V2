# Arquitectura y ejemplos de referencia

## Rol y principios

Se trabaja como desarrollador backend sénior en **FastAPI**, **APIs RESTful** y **Arquitectura en Capas**
(DDD / Clean Architecture simplificada). El código debe ser:

- **Desacoplado:** el dominio no conoce frameworks, PyTorch ni Pillow.
- **Testeable:** los casos de uso se prueban con fakes de sus puertos.
- **Mantenible:** una responsabilidad por módulo y routers delgados.
- **De alto rendimiento:** modelo cargado una vez, inferencia fuera del event loop, uploads en streaming.

## Capas

```
api  ──►  aplicacion  ──►  dominio  ◄──  infraestructura
                (puertos)              (adaptadores)
```

- `dominio` no importa nada de las otras capas ni de librerías de frameworks (solo stdlib).
- `aplicacion` define los **puertos** (`typing.Protocol`).
- `infraestructura` los implementa (modelo, trabajos, notificación, imágenes).
- `api` traduce HTTP ↔ casos de uso. Sin lógica de negocio. El cableado vive solo en `api/dependencias.py`.

## Modelo de dominio del servicio

| Concepto | Tipo | Atributos clave | Reglas |
|---|---|---|---|
| `Trabajo` | Agregado raíz | id, `IdEvento`, estado, imágenes (refs temporales), metadatos, tiempos | RF-19, idempotencia |
| `EstadoTrabajo` | Enum | `PENDIENTE → PROCESANDO → COMPLETADO \| FALLIDO` (sin vuelta atrás) | RF-22 |
| `ResultadoFoto` | Value object | peso_kg, confiable, motivos, confianzas, n_bovinos | R-24, R-25 |
| `ResultadoEstimacion` | Value object | agregado, motivos, versión, advertencias | R-20…R-23 |
| `MotivoNoSe` / `CodigoError` | Enum | catálogo cerrado (`errores.md`) | RF-22 |
| `VersionModelo` | Value object | versión, rango de validez | R-20, R-23 |

## Ejemplos por capa

**Dominio:** agregación y "no sé" sin frameworks.
```python
# dominio/estimacion.py
@dataclass(frozen=True, slots=True)
class ResultadoFoto:
    peso_kg: float | None
    confiable: bool
    motivos: tuple[MotivoNoSe, ...]

def agregar_resultados(fotos: Sequence[ResultadoFoto], version: VersionModelo) -> ResultadoEstimacion:
    """R-22: mediana de las confiables; R-20: fuera de rango no es confiable; R-23: versión."""
    confiables = [f.peso_kg for f in fotos if f.confiable and f.peso_kg is not None]
    if not confiables:
        motivos = tuple(sorted({m for f in fotos for m in f.motivos}))
        return ResultadoEstimacion(peso_kg=_mediana_o_none(fotos), confiable=False, motivos=motivos, version=version)
    peso = statistics.median(confiables)
    if not version.en_rango(peso):
        return ResultadoEstimacion(peso, False, (MotivoNoSe.FUERA_DE_RANGO,), version)
    return ResultadoEstimacion(peso, True, (), version)
```

**Aplicación:** caso de uso con presupuesto de tiempo.
```python
# aplicacion/casos_uso/procesar_estimacion.py
class ProcesarEstimacion:
    def __init__(self, modelo: ModeloPeso, trabajos: RepositorioTrabajos,
                 notificador: Notificador, imagenes: ObtenedorImagen, presupuesto_s: float) -> None: ...

    async def ejecutar(self, id_trabajo: IdTrabajo) -> None:
        trabajo = await self._trabajos.obtener(id_trabajo)
        await self._trabajos.guardar(trabajo.iniciar())
        try:
            with anyio.fail_after(self._presupuesto_s):                    # RNF-01
                imagenes = [await self._imagenes.obtener(r) for r in trabajo.imagenes]
                fotos = [await self._modelo.estimar(i) for i in imagenes]  # el adaptador usa to_thread
            trabajo = trabajo.completar(agregar_resultados(fotos, self._modelo.version))
        except TimeoutError:
            trabajo = trabajo.fallar(CodigoError.TIEMPO_EXCEDIDO, reintentable=True)
        except ModeloNoDisponible:
            trabajo = trabajo.fallar(CodigoError.MODELO_NO_DISPONIBLE, reintentable=True)
        await self._trabajos.guardar(trabajo)
        await self._notificador.notificar(trabajo)                         # RF-21 (soporte)
```

**API:** router delgado, 202 y `Location`.
```python
# api/v1/routers/estimaciones.py
@router.post("", status_code=202, responses={202: {"model": TrabajoAceptado}, 409: {}, 415: {}, 422: {}})
async def solicitar_estimacion(
    respuesta: Response,
    id_evento: Annotated[str, Header(alias="Idempotency-Key")],
    imagenes: Annotated[list[UploadFile], File(min_length=1, max_length=4)],
    caso: Annotated[SolicitarEstimacion, Depends(obtener_solicitar_estimacion)],
    _: Annotated[None, Depends(exigir_token_servicio)],
) -> TrabajoAceptado:
    trabajo = await caso.ejecutar(SolicitarEstimacionCmd(id_evento, [await leer_imagen(i) for i in imagenes]))
    respuesta.headers["Location"] = f"/v1/estimaciones/{trabajo.id}"
    return TrabajoAceptado.model_validate(trabajo)
```

**Arranque:** `lifespan`, hashes verificados y carga única.
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Configuracion()
    verificar_hashes_modelos(cfg.carpeta_modelos)          # si no coinciden, no arranca
    app.state.modelo = AdaptadorModelo(cfg.carpeta_modelos, limite=cfg.hilos_inferencia)
    yield
```

## Integración del modelo

- `infraestructura/ml/` envuelve `estimador.EstimadorPeso` e implementa el puerto `ModeloPeso`.
- Se carga **una sola vez** en el `lifespan` y se guarda en `app.state`. `/salud` responde 503 hasta que esté listo.
- La inferencia corre con `anyio.to_thread.run_sync` y un `CapacityLimiter` igual a los núcleos disponibles.
- Al arrancar se verifican los SHA-256 de los `.pt` (los de `descargar_modelos.py` / `SHA256SUMS.txt`). Si no
  coinciden, la app **no arranca**.
- El rango de validez y la versión se leen de `config.json`.
- Cambiar de modelo (por ejemplo `modelos-v2` con datos argentinos) debe requerir **solo** archivos y
  configuración.

### Qué cubre el adaptador (sin tocar `estimador/estimador.py`)

`EstimadorPeso.estimar()` devuelve un `dict` pensado para consumo humano. El adaptador
(`infraestructura/ml/adaptador_modelo.py`) lo convierte a `ResultadoFoto` y cubre estas brechas:

1. **Motivos en texto libre → códigos estables.** El módulo devuelve `motivo` como un string con los motivos
   unidos por `"; "`. El adaptador lo separa y lo traduce con una tabla **exacta**:

   | Texto del módulo | Código (`errores.md`) |
   |---|---|
   | `no se detectó ningún bovino en la foto` (o `peso_kg is None`) | `sin_animal` |
   | `el bovino no se distingue con claridad` | `deteccion_dudosa` |
   | `hay partes del cuerpo tapadas o fuera de cuadro` | `partes_tapadas` |
   | `el animal aparece cortado por el borde de la foto` | `animal_cortado` |
   | `el peso estimado cae fuera del rango con el que se entrenó el modelo` | `fuera_de_rango` |

   Un texto que no esté en la tabla **no se descarta en silencio**: el adaptador lo registra en el log y falla el
   test de contrato del adaptador, para detectar cambios futuros del módulo.
2. **Rango de validez.** `estimador.py` tiene el rango escrito en el código (`RANGO_ENTRENAMIENTO`). La fuente de
   verdad es `rango_entrenamiento_kg` de `config.json`: el dominio recalcula R-20 con ese rango
   (`VersionModelo.en_rango()`), sin depender del motivo que haya informado el módulo.
3. **Imagen borrosa (R-24).** El módulo no la calcula. El adaptador mide la varianza del Laplaciano sobre el
   `recorte` que devuelve el módulo; el umbral viene de configuración y no tiene valor por defecto hasta
   calibrarlo (ver `errores.md`).
4. **Concurrencia.** Los modelos de Ultralytics no son seguros si varios hilos comparten una misma instancia. El
   adaptador mantiene un **pool de N instancias** de `EstimadorPeso` (`HILOS_INFERENCIA`; unos 130 MB de modelos
   cada una) y un `CapacityLimiter(N)`. Una instancia se toma del pool para cada inferencia y se devuelve al
   terminar.
5. **Umbrales de R-25.** `CONF_DETECCION_MIN` y `CONF_KEYPOINT_MIN` se importan de `estimador`, no se copian.

### Cómo se instala `estimador/`
`estimador/pyproject.toml` declara `[build-system]` (hatchling, solo `estimador.py`), así que el servicio lo toma
como dependencia por ruta:

```toml
# servicio-estimacion/pyproject.toml
dependencies = ["estimador-peso-bovino"]
[tool.uv.sources]
estimador-peso-bovino = { path = "../estimador" }
```

Los `.pt` no viajan en el paquete: se descargan con `python ../estimador/descargar_modelos.py` y se apunta a la
carpeta con `CARPETA_MODELOS`.

## Variables de entorno

Las lee `infraestructura/config.py` (`pydantic-settings`). Nunca se versionan: el repo solo trae
`servicio-estimacion/.env.example`, sin secretos.

| Variable | Obligatoria | Por defecto | Uso |
|---|---|---|---|
| `TOKEN_SERVICIO` | sí | — | Token Bearer que usa el adaptador C# (rotado por ambiente) |
| `SECRETO_HMAC` | sí | — | Firma del callback (`X-BV-Firma`) |
| `HOSTS_IMAGEN_PERMITIDOS` | si se usan URLs firmadas | vacío | Allowlist de hosts para `referencias_imagen` (SSRF) |
| `HOSTS_CALLBACK_PERMITIDOS` | si se usa callback | vacío | Allowlist de hosts para `url_notificacion` (SSRF) |
| `CARPETA_MODELOS` | no | `../estimador/modelos` | Dónde están los `.pt`, `config.json` y `SHA256SUMS.txt` |
| `PRESUPUESTO_ESTIMACION_S` | no | `45` (propuesto) | Límite por trabajo (RNF-01) |
| `HILOS_INFERENCIA` | no | núcleos disponibles | Tamaño del pool y del `CapacityLimiter` |
| `MAX_MB_IMAGEN` | no | `25` | Límite por imagen |
| `RESOLUCION_MINIMA_MP` | no | `3` (a confirmar) | Umbral de `resolucion_insuficiente` |
| `UMBRAL_BORROSIDAD` | no | sin valor | Umbral de R-24. Sin valor, la regla queda **desactivada** |
| `TTL_TRABAJOS_S` | no | `86400` (propuesto) | Cuánto se conserva un trabajo (ver `contrato-backend.md`) |
| `TAMANO_COLA` | no | `200` (propuesto) | Trabajos que pueden esperar (ver `contrato-backend.md`) |

## Logs

JSON, una línea por evento. Campos fijos: `ts`, `nivel`, `evento`, `request_id`, `id_evento`, `id_trabajo`,
`version_modelo` y `duracion_ms` por etapa cuando aplica. **Nunca** imágenes, URLs firmadas completas, tokens ni
coordenadas GPS.

## Rendimiento

- Los uploads se leen por chunks y el límite de tamaño (25 MB por imagen) se valida **antes** de decodificar.
- El `POST` encola de inmediato y no espera la inferencia.
- Los hilos de inferencia se limitan a los núcleos disponibles.
