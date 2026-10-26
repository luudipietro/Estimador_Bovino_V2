"""Para los tests de la API: una app real con modelo falso, atendida por `httpx` sin red ni uvicorn."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from bv_estimacion.api.main import create_app
from tests.modelo_falso import TOKEN, FabricaFalsa, configuracion_de_prueba

AUTORIZACION = {"Authorization": f"Bearer {TOKEN}"}


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
