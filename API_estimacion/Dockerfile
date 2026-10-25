# API de estimación de peso bovino (CPU).
#
# El contexto de build es la RAÍZ del repo (ver docker-compose.yml), porque reusa estimador/estimador.py y el
# uv.lock de estimador/ en vez de duplicarlos. Los modelos (.pt) NO van en la imagen: se montan en /modelos.
FROM python:3.12-slim

# OpenCV (dependencia de ultralytics) necesita estas dos librerías del sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.19

ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=False \
    MODELOS_DIR=/modelos

WORKDIR /app

# Dependencias primero: esta capa solo se rehace si cambian las versiones del lock
COPY estimador/pyproject.toml estimador/uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Ultralytics necesita un directorio escribible para su archivo de configuración
ENV YOLO_CONFIG_DIR=/tmp

COPY estimador/estimador.py ./
COPY API_estimacion/app ./app

RUN useradd --create-home --uid 1000 api
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/salud', timeout=4)"

# Un solo worker: hay un modelo en memoria y la inferencia usa todos los núcleos
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
