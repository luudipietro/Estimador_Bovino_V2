import json
import logging
from pathlib import Path

import pytest
from fastapi import FastAPI, Query

from bv_estimacion.api.errores import ESTADO_HTTP, TITULO
from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio, TransicionInvalida
from tests.api.ayudas import servicio

SECRETO_INTERNO = "clave-interna-no-se-ve /srv/app/bv_estimacion/modulo.py"


def rutas(app: FastAPI) -> None:
    @app.get("/_dominio/{codigo}")
    async def lanzar(codigo: str) -> None:
        raise ErrorDeDominio(CodigoError(codigo), "Detalle seguro de prueba.", regla="R-99")

    @app.get("/_dominio_con_secreto/{codigo}")
    async def lanzar_con_secreto(codigo: str) -> None:
        raise ErrorDeDominio(CodigoError(codigo), SECRETO_INTERNO)

    @app.get("/_transicion")
    async def transicion() -> None:
        raise TransicionInvalida(SECRETO_INTERNO)

    @app.get("/_boom")
    async def boom() -> None:
        raise RuntimeError(SECRETO_INTERNO)

    @app.get("/_numero")
    async def numero(n: int = Query()) -> dict[str, int]:
        return {"n": n}


# Los estados HTTP del catálogo de `docs/errores.md` (sección 1) más los de las fallas internas.
ESTADOS_ESPERADOS = {
    "formato_no_admitido": 415,
    "imagen_corrupta": 422,
    "imagen_muy_grande": 413,
    "resolucion_insuficiente": 422,
    "cantidad_imagenes_invalida": 422,
    "id_evento_en_conflicto": 409,
    "referencia_no_permitida": 422,
    "no_autenticado": 401,
    "trabajo_no_encontrado": 404,
    "cola_llena": 503,
    "solicitud_invalida": 400,
}


@pytest.mark.requisito("RF-22")
def test_todo_codigo_del_catalogo_tiene_estado_http_y_titulo() -> None:
    """Si alguien agrega un `CodigoError` y no le asigna un estado, este test falla en vez de salir un 500."""
    assert set(ESTADO_HTTP) == set(CodigoError)
    assert set(TITULO) == set(CodigoError)


@pytest.mark.requisito("RF-22")
@pytest.mark.parametrize(("codigo", "estado"), sorted(ESTADOS_ESPERADOS.items()))
async def test_cada_error_de_entrada_sale_con_el_estado_del_catalogo(tmp_path: Path, codigo: str, estado: int) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get(f"/_dominio/{codigo}")

    assert respuesta.status_code == estado
    assert respuesta.headers["content-type"] == "application/problem+json"
    assert ESTADO_HTTP[CodigoError(codigo)] == estado


@pytest.mark.requisito("RF-22")
async def test_el_cuerpo_es_rfc_9457_con_codigo_regla_y_request_id(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_dominio/formato_no_admitido")

    cuerpo = respuesta.json()
    assert cuerpo == {
        "type": "https://bovinovision/errores/formato_no_admitido",
        "title": TITULO[CodigoError.FORMATO_NO_ADMITIDO],
        "status": 415,
        "detail": "Detalle seguro de prueba.",
        "codigo": "formato_no_admitido",
        "regla": "R-99",
        "request_id": respuesta.headers["x-request-id"],
    }


@pytest.mark.requisito("RF-22")
async def test_sin_regla_el_cuerpo_no_trae_el_campo_regla(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_dominio_con_secreto/imagen_corrupta")
    assert "regla" not in respuesta.json()


@pytest.mark.requisito("RF-22")
async def test_cola_llena_avisa_cuando_reintentar(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_dominio/cola_llena")
    assert respuesta.status_code == 503
    assert int(respuesta.headers["retry-after"]) > 0


@pytest.mark.requisito("RF-22")
@pytest.mark.parametrize(
    ("codigo", "estado"),
    [("modelo_no_disponible", 503), ("tiempo_excedido", 504), ("referencia_inaccesible", 502), ("error_interno", 500)],
)
async def test_una_falla_interna_nunca_expone_el_detalle_con_que_se_lanzo(
    tmp_path: Path, codigo: str, estado: int
) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get(f"/_dominio_con_secreto/{codigo}")

    assert respuesta.status_code == estado
    assert SECRETO_INTERNO not in respuesta.text
    assert respuesta.json()["detail"]  # y hay un texto genérico, no vacío


@pytest.mark.requisito("RF-22")
async def test_una_transicion_invalida_es_un_500_generico(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_transicion")
    assert respuesta.status_code == 500
    assert respuesta.json()["codigo"] == "error_interno"
    assert SECRETO_INTERNO not in respuesta.text


@pytest.mark.requisito("RF-22")
async def test_una_excepcion_sin_controlar_es_un_500_generico_sin_trazas_ni_clases(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        with caplog.at_level(logging.ERROR):
            respuesta = await cliente.get("/_boom")

    assert respuesta.status_code == 500
    assert respuesta.headers["content-type"] == "application/problem+json"
    cuerpo = respuesta.json()
    assert cuerpo["codigo"] == "error_interno"
    assert cuerpo["request_id"] == respuesta.headers["x-request-id"]
    for filtrado in (SECRETO_INTERNO, "RuntimeError", "Traceback", ".py", "bv_estimacion", "modulo"):
        assert filtrado not in respuesta.text
    # El detalle va al log, con el mismo request_id para poder encontrarlo.
    registros = [r for r in caplog.records if r.getMessage() == "error_no_controlado"]
    assert len(registros) == 1
    assert registros[0].exc_info is not None
    assert getattr(registros[0], "request_id", None) == cuerpo["request_id"]


@pytest.mark.requisito("RF-22")
async def test_un_parametro_invalido_es_solicitud_invalida_y_no_repite_lo_que_mando_el_cliente(
    tmp_path: Path,
) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_numero", params={"n": "valor-que-no-se-repite"})

    assert respuesta.status_code == 400
    cuerpo = respuesta.json()
    assert cuerpo["codigo"] == "solicitud_invalida"
    assert "n" in cuerpo["campos"]
    assert "valor-que-no-se-repite" not in respuesta.text


@pytest.mark.requisito("RF-22")
async def test_un_parametro_que_falta_tambien_es_solicitud_invalida(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_numero")
    assert respuesta.status_code == 400
    assert respuesta.json()["codigo"] == "solicitud_invalida"


@pytest.mark.requisito("RF-22")
async def test_una_ruta_que_no_existe_y_un_metodo_no_permitido_tambien_salen_como_problem_json(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        no_existe = await cliente.get("/ruta/inexistente")
        metodo = await cliente.post("/_boom")

    assert no_existe.status_code == 404
    assert metodo.status_code == 405
    for respuesta in (no_existe, metodo):
        assert respuesta.headers["content-type"] == "application/problem+json"
        assert respuesta.json()["status"] == respuesta.status_code
        assert respuesta.json()["codigo"] == "solicitud_invalida"
        assert "request_id" in respuesta.json()


@pytest.mark.requisito("RF-22")
async def test_ninguna_respuesta_de_error_filtra_trazas(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuestas = [
            await cliente.get("/_boom"),
            await cliente.get("/_transicion"),
            await cliente.get("/_dominio_con_secreto/error_interno"),
            await cliente.get("/_numero", params={"n": "x"}),
            await cliente.get("/no-existe"),
        ]
    for respuesta in respuestas:
        cuerpo = json.dumps(respuesta.json())
        for pista in ("Traceback", 'File "', "Exception", "Error(", ".py", "site-packages", "/srv/"):
            assert pista not in cuerpo


@pytest.mark.requisito("RF-22")
async def test_la_documentacion_interactiva_esta_apagada_por_defecto_porque_no_tiene_autenticacion(
    tmp_path: Path,
) -> None:
    async with servicio(tmp_path) as (_, cliente):
        for ruta in ("/docs", "/openapi.json", "/redoc"):
            assert (await cliente.get(ruta)).status_code == 404


@pytest.mark.requisito("RF-22")
async def test_con_docs_habilitadas_se_publican_docs_y_openapi(tmp_path: Path) -> None:
    async with servicio(tmp_path, docs_habilitadas=True) as (_, cliente):
        assert (await cliente.get("/docs")).status_code == 200
        esquema = await cliente.get("/openapi.json")
    assert esquema.status_code == 200
    assert esquema.json()["info"]["title"].startswith("BovinoVision")
