"""API HTTP mínima para que la app móvil pida una estimación de peso.

    uv run --extra api uvicorn api:app --host 0.0.0.0 --port 8000
    (o:  python -m uvicorn api:app --reload)

Probar desde la consola:
    curl -F "foto=@vaca.jpg" http://localhost:8000/estimar

Respuesta:
    {"peso_kg": 187.4, "confiable": true, "motivo": null, "confianza_deteccion": 0.94, ...}

Documentación interactiva para probar desde el navegador: http://localhost:8000/docs
"""
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from estimador import EstimadorPeso

MODELOS = Path(__file__).parent / "modelos"
MAX_MB = 25

app = FastAPI(title="BovinoVision — estimación de peso",
              description="Recibe una foto lateral de un bovino y devuelve el peso estimado.",
              version="0.1.0")
_estimador: EstimadorPeso | None = None


def obtener_estimador() -> EstimadorPeso:
    """Carga los modelos una sola vez, en la primera llamada (tarda unos segundos)."""
    global _estimador
    if _estimador is None:
        _estimador = EstimadorPeso(MODELOS)
    return _estimador


@app.get("/salud")
def salud():
    """Para monitoreo: indica si el servicio está arriba y con los modelos cargados."""
    return {"estado": "ok", "modelos_cargados": _estimador is not None}


@app.post("/estimar")
async def estimar(foto: UploadFile = File(..., description="Foto lateral del animal (JPEG o PNG)")):
    datos = await foto.read()
    if not datos:
        raise HTTPException(400, "el archivo llegó vacío")
    if len(datos) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"la foto supera los {MAX_MB} MB")
    try:
        return obtener_estimador().estimar(datos)
    except Exception as e:  # no filtrar detalles internos al cliente
        raise HTTPException(500, f"no se pudo procesar la imagen: {e.__class__.__name__}")


@app.on_event("startup")
def precargar():
    """Carga los modelos al arrancar, para que la primera foto no espere."""
    obtener_estimador()
