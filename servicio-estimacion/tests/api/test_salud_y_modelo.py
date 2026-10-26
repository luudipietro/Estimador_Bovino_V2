from pathlib import Path

import httpx
import pytest

from bv_estimacion.api.main import create_app
from tests.api.ayudas import AUTORIZACION, servicio
from tests.modelo_falso import FabricaFalsa, configuracion_de_prueba, crear_carpeta_modelo

VERSION = "v-prueba / datos de prueba"


# --- GET /salud ---------------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_salud_responde_200_sin_token_con_el_modelo_cargado(tmp_path: Path) -> None:
    async with servicio(tmp_path, hilos_inferencia=2) as (_, cliente):
        respuesta = await cliente.get("/salud")  # sin Authorization

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "estado": "ok",
        "modelo_cargado": True,
        "version_modelo": VERSION,
        "instancias": 2,
        "error": None,
    }


@pytest.mark.requisito("RF-19")
async def test_salud_no_se_cachea(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.get("/salud")
    assert respuesta.headers["cache-control"] == "no-store"


@pytest.mark.requisito("RF-19")
async def test_si_el_modelo_no_esta_listo_salud_responde_503_con_el_motivo(tmp_path: Path) -> None:
    """No pasa con el arranque normal (la app no arranca sin modelo), pero `/salud` no tiene que mentir si pasa."""
    app = create_app(
        configuracion_de_prueba(tmp_path), fabrica_modelo=FabricaFalsa().configurada(), configurar_logs=False
    )
    transporte = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as cliente:
        respuesta = await cliente.get("/salud")  # sin `lifespan`: nada cargado

    assert respuesta.status_code == 503
    assert respuesta.json() == {
        "estado": "no_disponible",
        "modelo_cargado": False,
        "version_modelo": None,
        "instancias": 0,
        "error": "El modelo no está cargado.",
    }


# --- GET /v1/modelo -----------------------------------------------------------------------------


@pytest.mark.requisito("R-23")
async def test_modelo_necesita_token(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.get("/v1/modelo")
    assert respuesta.status_code == 401


@pytest.mark.requisito("R-23")
async def test_modelo_cuenta_version_rango_y_lo_opcional_que_tenga_el_config_json(tmp_path: Path) -> None:
    carpeta = crear_carpeta_modelo(
        tmp_path / "otros-modelos",
        mape_test=15.6,
        ic95_test=[14.0, 16.3],
        fecha_entrenamiento="2026-09-25",
    )
    async with servicio(tmp_path, carpeta_modelos=carpeta) as (_, cliente):
        respuesta = await cliente.get("/v1/modelo", headers=AUTORIZACION)

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "version": VERSION,
        "rango_validez_kg": [36.0, 621.0],
        "mape_test": 15.6,
        "ic95_test": [14.0, 16.3],
        "fecha_entrenamiento": "2026-09-25",
    }


@pytest.mark.requisito("R-23")
async def test_lo_que_el_config_json_no_trae_sale_en_null_y_no_se_saca_del_texto_de_la_version(tmp_path: Path) -> None:
    """El `config.json` de v1 no tiene `fecha_entrenamiento`, aunque la fecha figure dentro del texto de `version`."""
    carpeta = crear_carpeta_modelo(tmp_path / "v1", version="2026-09-25 / entrenado con AcmeAI-BMGF")
    async with servicio(tmp_path, carpeta_modelos=carpeta) as (_, cliente):
        cuerpo = (await cliente.get("/v1/modelo", headers=AUTORIZACION)).json()

    assert cuerpo["version"] == "2026-09-25 / entrenado con AcmeAI-BMGF"
    assert cuerpo["fecha_entrenamiento"] is None
    assert cuerpo["mape_test"] is None
    assert cuerpo["ic95_test"] is None


@pytest.mark.requisito("R-20")
async def test_el_rango_que_se_publica_es_el_del_config_json_no_uno_escrito_en_el_codigo(tmp_path: Path) -> None:
    carpeta = crear_carpeta_modelo(tmp_path / "otro-rango", rango_entrenamiento_kg=[80, 700])
    async with servicio(tmp_path, carpeta_modelos=carpeta) as (_, cliente):
        cuerpo = (await cliente.get("/v1/modelo", headers=AUTORIZACION)).json()
    assert cuerpo["rango_validez_kg"] == [80.0, 700.0]


# --- Autenticación en todas las rutas -------------------------------------------------------------


@pytest.mark.requisito("RF-22")
async def test_toda_ruta_de_la_api_exige_token_salvo_salud(tmp_path: Path) -> None:
    """Recorre las rutas del esquema OpenAPI: un router nuevo que olvide `Depends(exigir_token)` hace fallar este
    test. (FastAPI 0.162 registra los routers incluidos como `_IncludedRouter`, no como `APIRoute`, así que se mira
    el esquema, que lista las rutas efectivas.)"""
    async with servicio(tmp_path) as (app, cliente):
        rutas = app.openapi()["paths"]
        assert set(rutas) >= {"/salud", "/v1/modelo"}

        for ruta, operaciones in rutas.items():
            if ruta == "/salud":
                continue
            camino = ruta.replace("{id_trabajo}", "trb_inexistente")
            for metodo in operaciones:
                respuesta = await cliente.request(metodo.upper(), camino)  # sin Authorization
                assert respuesta.status_code == 401, f"{metodo.upper()} {ruta} quedó abierta"
                assert respuesta.json()["codigo"] == "no_autenticado"
