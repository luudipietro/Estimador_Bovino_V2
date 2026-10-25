"""API de estimación de peso bovino. Recibe una foto lateral ya achicada y devuelve el peso estimado.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Documentación interactiva en /docs. El detalle del contrato está en README.md.
"""
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from estimador import EstimadorPeso

from . import config
from .imagenes import ErrorImagen, preparar

# Claves que siempre están en la respuesta; el módulo devuelve un esquema reducido cuando no encuentra al animal.
CLAVES_RESPUESTA = ("peso_kg", "confiable", "motivo", "confianza_deteccion", "confianza_keypoint_minima",
                    "bovinos_en_la_escena", "keypoints", "recorte", "resolucion", "modelo")


def cargar_estimador() -> EstimadorPeso:
    cfg = config.MODELOS_DIR / "config.json"
    if not cfg.exists():
        raise RuntimeError(
            f"no se encontró {cfg}. Los modelos no viajan en el repositorio: correr "
            f"`python descargar_modelos.py` en estimador/ y montar esa carpeta modelos/ en {config.MODELOS_DIR}")
    return EstimadorPeso(config.MODELOS_DIR)


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    # Carga los modelos antes de aceptar tráfico, para que la primera foto no espere.
    app.state.estimador = cargar_estimador()
    app.state.candado = threading.Lock()  # un solo modelo en CPU: las consultas se encolan en vez de competir
    yield


app = FastAPI(title="BovinoVision — API de estimación de peso",
              description="Recibe una foto lateral de un bovino, achicada a 1024 px de lado mayor, y devuelve el peso estimado.",
              version="0.1.0", lifespan=ciclo_de_vida)


@app.exception_handler(ErrorImagen)
def _error_imagen(_, e: ErrorImagen):
    return JSONResponse(status_code=e.status, content={"detail": e.detalle})


@app.get("/salud")
def salud():
    """Para monitoreo y healthcheck de Docker."""
    return {"estado": "ok", "modelos_cargados": True, "modelo": app.state.estimador.cfg.get("version")}


@app.get("/info")
def info():
    """Contrato de imagen y datos del modelo, para que el cliente no los hardcodee."""
    cfg = app.state.estimador.cfg
    return {
        "imagen": {"lado_mayor_px": config.LADO_MAX_PX, "lado_mayor_minimo_px": config.LADO_MIN_PX,
                   "formatos": list(config.FORMATOS_PUBLICOS), "tamano_maximo_bytes": config.MAX_BYTES,
                   "campo_multipart": "foto"},
        "modelo": {"version": cfg.get("version"), "mape_test": cfg.get("mape_test"), "ic95_test": cfg.get("ic95_test"),
                   "rango_entrenamiento_kg": cfg.get("rango_entrenamiento_kg"), "nota": cfg.get("nota")},
        "limitaciones": [
            "Entrenado con bovinos de Bangladesh: no extrapola por encima de ~270 kg ni transfiere a otros rangos o razas.",
            "'confiable: true' no garantiza un peso correcto; solo indica que la foto y la detección son buenas.",
        ],
    }


@app.post("/estimar")
def estimar(foto: UploadFile = File(..., description="Foto lateral del animal (JPEG o PNG), lado mayor 1024 px")):
    # Se lee un byte de más para detectar el exceso sin cargar archivos enormes completos.
    datos = foto.file.read(config.MAX_BYTES + 1)
    im, aviso = preparar(datos)

    t0 = time.perf_counter()
    try:
        with app.state.candado:
            crudo = app.state.estimador.estimar(im)
    except Exception as e:  # no filtrar detalles internos al cliente
        raise HTTPException(500, f"no se pudo procesar la imagen: {e.__class__.__name__}")

    r = {k: crudo.get(k) for k in CLAVES_RESPUESTA}
    advertencias = []
    if aviso:
        advertencias.append(aviso)
    if r["peso_kg"] is not None and r["peso_kg"] >= config.PESO_SATURACION_KG:
        advertencias.append("el modelo no extrapola por encima de ~270 kg: este valor está cerca de su techo y el "
                            "animal podría pesar bastante más")
    r["advertencias"] = advertencias
    r["tiempo_ms"] = round((time.perf_counter() - t0) * 1000)
    return r
