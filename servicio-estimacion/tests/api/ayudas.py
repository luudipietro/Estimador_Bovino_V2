"""Para los tests de la API: una app real con modelo falso, atendida por `httpx` sin red ni uvicorn."""

import hashlib
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx
from fastapi import FastAPI

from bv_estimacion.api.main import create_app
from tests import imagenes_sinteticas as img
from tests.modelo_falso import TOKEN, FabricaFalsa, configuracion_de_prueba

AUTORIZACION = {"Authorization": f"Bearer {TOKEN}"}
RUTA = "/v1/estimaciones"


@asynccontextmanager
async def servicio(
    tmp_path: Path,
    *,
    fabrica: FabricaFalsa | None = None,
    rutas: Callable[[FastAPI], None] | None = None,
    **cambios: Any,
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    """Arranca la app (con su `lifespan` completo) y da un cliente HTTP que la atiende en memoria.

    `raise_app_exceptions=False`: una excepción sin controlar tiene que verse como la respuesta 500 que recibiría el
    cliente real, no romper el test.
    """
    app = create_app(
        configuracion_de_prueba(tmp_path, **cambios),
        fabrica_modelo=(fabrica or FabricaFalsa()).configurada(),
        configurar_logs=False,
    )
    if rutas is not None:
        rutas(app)
    async with app.router.lifespan_context(app):
        transporte = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as cliente:
            yield app, cliente


def foto(numero: int = 0) -> bytes:
    """Un JPEG válido (más de 0,75 MP). Cada `numero` da una imagen distinta, o sea un SHA-256 distinto."""
    return img.jpeg(1024 + 8 * numero, 768)


def sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def archivos(*fotos: bytes, campo: str = "imagenes") -> list[tuple[str, tuple[str, bytes, str]]]:
    return [(campo, (f"foto{i}.jpg", datos, "image/jpeg")) for i, datos in enumerate(fotos, start=1)]


def encabezados(id_evento: str | None = "evt-1", **otros: str) -> dict[str, str]:
    base = dict(AUTORIZACION)
    if id_evento is not None:
        base["Idempotency-Key"] = id_evento
    return base | otros


async def pedir(
    cliente: httpx.AsyncClient,
    *fotos: bytes,
    id_evento: str | None = "evt-1",
    metadatos: object | None = None,
    url_notificacion: str | None = None,
) -> httpx.Response:
    """`POST /v1/estimaciones` por multipart."""
    datos: dict[str, str] = {}
    if metadatos is not None:
        datos["metadatos"] = json.dumps(metadatos)
    if url_notificacion is not None:
        datos["url_notificacion"] = url_notificacion
    return await cliente.post(RUTA, headers=encabezados(id_evento), files=archivos(*fotos), data=datos)


async def esperar_estado(cliente: httpx.AsyncClient, id_trabajo: str, estado: str = "completado") -> httpx.Response:
    with anyio.fail_after(5):
        while True:
            respuesta = await cliente.get(f"{RUTA}/{id_trabajo}", headers=AUTORIZACION)
            if respuesta.json()["estado"] == estado:
                return respuesta
            await anyio.sleep(0.01)
