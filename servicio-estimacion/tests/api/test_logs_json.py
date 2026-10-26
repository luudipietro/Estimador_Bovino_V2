import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from bv_estimacion.api.logs import CONTEXTO, JsonFormatter, configurar_logs, request_id_var
from bv_estimacion.api.middleware import ContextoDePeticion
from tests.api.ayudas import AUTORIZACION, servicio


@pytest.fixture
def salida() -> Iterator[io.StringIO]:
    """Un logger propio con el formateador JSON, para no depender del root logger de pytest."""
    buffer = io.StringIO()
    manejador = logging.StreamHandler(buffer)
    manejador.setFormatter(JsonFormatter())
    logger = logging.getLogger("prueba.logs_json")
    logger.handlers = [manejador]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield buffer
    logger.handlers = []


def ultimo(salida: io.StringIO) -> dict[str, Any]:
    lineas = [linea for linea in salida.getvalue().splitlines() if linea]
    cuerpo = json.loads(lineas[-1])
    assert isinstance(cuerpo, dict)
    return cuerpo


@pytest.mark.requisito("RF-22")
def test_cada_evento_es_una_linea_json_con_los_campos_fijos(salida: io.StringIO) -> None:
    logging.getLogger("prueba.logs_json").info("trabajo_finalizado")

    lineas = salida.getvalue().splitlines()
    assert len(lineas) == 1
    registro = json.loads(lineas[0])
    assert set(registro) >= {"ts", "nivel", "evento", "request_id", "id_evento", "id_trabajo", "version_modelo"}
    assert registro["nivel"] == "INFO"
    assert registro["evento"] == "trabajo_finalizado"
    assert registro["ts"].endswith("+00:00") or registro["ts"].endswith("Z")  # con zona horaria


@pytest.mark.requisito("RF-22")
def test_los_campos_que_no_aplican_van_en_null(salida: io.StringIO) -> None:
    logging.getLogger("prueba.logs_json").info("evento_suelto")
    registro = ultimo(salida)
    assert registro["request_id"] is None
    assert registro["id_evento"] is None
    assert registro["id_trabajo"] is None


@pytest.mark.requisito("RF-22")
def test_el_id_evento_y_el_id_trabajo_salen_de_los_extras(salida: io.StringIO) -> None:
    logging.getLogger("prueba.logs_json").info(
        "trabajo_finalizado", extra={"id_evento": "evt-1", "id_trabajo": "trb_01", "estado": "completado"}
    )
    registro = ultimo(salida)
    assert (registro["id_evento"], registro["id_trabajo"], registro["estado"]) == ("evt-1", "trb_01", "completado")


@pytest.mark.requisito("RF-22")
def test_el_request_id_sale_del_contexto_de_la_peticion(salida: io.StringIO) -> None:
    token = request_id_var.set("req-123")
    try:
        logging.getLogger("prueba.logs_json").info("algo")
    finally:
        request_id_var.reset(token)
    assert ultimo(salida)["request_id"] == "req-123"


@pytest.mark.requisito("R-23")
def test_la_version_del_modelo_sale_en_todas_las_lineas_una_vez_cargado(salida: io.StringIO) -> None:
    anterior = CONTEXTO.version_modelo
    CONTEXTO.version_modelo = "v-de-prueba"
    try:
        logging.getLogger("prueba.logs_json").info("algo")
    finally:
        CONTEXTO.version_modelo = anterior
    assert ultimo(salida)["version_modelo"] == "v-de-prueba"


@pytest.mark.requisito("RF-22")
def test_lo_que_no_se_puede_serializar_se_convierte_en_texto_y_no_rompe_el_log(salida: io.StringIO) -> None:
    logging.getLogger("prueba.logs_json").info("algo", extra={"ruta": Path("a/b"), "cosas": {1, 2}})
    registro = ultimo(salida)
    assert registro["ruta"] in ("a/b", "a\\b")
    assert isinstance(registro["cosas"], str)


@pytest.mark.requisito("RF-22")
def test_nunca_se_loguea_una_url_ni_un_token_aunque_vengan_en_un_mensaje_o_en_la_excepcion(
    salida: io.StringIO,
) -> None:
    url = "https://almacenamiento.example/privado/foto.jpg?firma=ABC123SECRETA"
    logger = logging.getLogger("prueba.logs_json")
    try:
        raise RuntimeError(f"falló la descarga de {url} con Authorization: Bearer token-super-secreto")
    except RuntimeError:
        logger.exception("descarga_fallida", extra={"motivo": f"no se pudo bajar {url}"})

    texto = salida.getvalue()
    for secreto in ("ABC123SECRETA", "almacenamiento.example", "token-super-secreto", "firma="):
        assert secreto not in texto
    registro = ultimo(salida)
    assert registro["error"] == "RuntimeError"  # la clase sí sirve para diagnosticar
    assert "Traceback" in registro["traza"]  # y la traza queda en el log (no en la respuesta), ya saneada


@pytest.mark.requisito("RF-22")
def test_una_linea_con_excepcion_sigue_siendo_una_sola_linea_json(salida: io.StringIO) -> None:
    try:
        raise ValueError("uno\ndos")
    except ValueError:
        logging.getLogger("prueba.logs_json").exception("fallo")
    assert len(salida.getvalue().splitlines()) == 1


@pytest.mark.requisito("RF-22")
def test_configurar_logs_es_idempotente_y_deja_un_solo_manejador_json() -> None:
    raiz = logging.getLogger()
    antes = list(raiz.handlers)
    try:
        configurar_logs()
        configurar_logs()
        propios = [h for h in raiz.handlers if isinstance(h.formatter, JsonFormatter)]
        assert len(propios) == 1
    finally:
        raiz.handlers = antes


def rutas(app: FastAPI) -> None:
    @app.get("/_ok")
    async def ok() -> dict[str, bool]:
        return {"ok": True}


@pytest.mark.requisito("RF-22")
async def test_cada_peticion_deja_una_linea_de_acceso_sin_query_string_ni_encabezados(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        with caplog.at_level(logging.INFO):
            respuesta = await cliente.get("/_ok", params={"firma": "SECRETA"}, headers=AUTORIZACION)

    registros = [r for r in caplog.records if r.getMessage() == "peticion"]
    assert len(registros) == 1
    r = registros[0]
    assert (r.metodo, r.ruta, r.estado) == ("GET", "/_ok", 200)  # type: ignore[attr-defined]
    assert r.duracion_ms >= 0  # type: ignore[attr-defined]
    assert r.request_id == respuesta.headers["x-request-id"]  # type: ignore[attr-defined]
    volcado = json.dumps({k: str(v) for k, v in vars(r).items()})
    assert "SECRETA" not in volcado
    assert AUTORIZACION["Authorization"] not in volcado


@pytest.mark.requisito("RF-22")
async def test_el_request_id_es_nuevo_en_cada_peticion_y_se_respeta_el_que_manda_el_cliente(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        a = await cliente.get("/_ok")
        b = await cliente.get("/_ok")
        propio = await cliente.get("/_ok", headers={"X-Request-ID": "mi-id-de-trazabilidad_01"})

    assert a.headers["x-request-id"] != b.headers["x-request-id"]
    assert len(a.headers["x-request-id"]) == 32
    assert propio.headers["x-request-id"] == "mi-id-de-trazabilidad_01"


@pytest.mark.requisito("RF-22")
@pytest.mark.parametrize("invalido", ["con espacios", "x" * 200, "inyeccion;<script>", ""])
async def test_un_request_id_raro_del_cliente_se_reemplaza_por_uno_propio(tmp_path: Path, invalido: str) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_ok", headers={"X-Request-ID": invalido})
    assert respuesta.headers["x-request-id"] != invalido
    assert len(respuesta.headers["x-request-id"]) == 32


@pytest.mark.requisito("R-23")
async def test_al_arrancar_el_log_lleva_la_version_del_modelo_de_la_carpeta_de_modelos(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas):
        assert CONTEXTO.version_modelo == "v-prueba / datos de prueba"


@pytest.mark.requisito("RF-22")
async def test_el_middleware_deja_pasar_sin_tocar_lo_que_no_es_http() -> None:
    """El `lifespan` y los websockets pasan por el mismo middleware: no tienen `request_id` ni línea de acceso."""
    vistos: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        vistos.append(scope["type"])

    await ContextoDePeticion(app)({"type": "lifespan"}, None, None)  # type: ignore[arg-type]

    assert vistos == ["lifespan"]
