from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from bv_estimacion.aplicacion.puertos import ImagenNoDisponible
from bv_estimacion.infraestructura.imagenes.descarga_http import ObtenedorImagenHttp

URL = "https://almacen.example/fotos/a.jpg?firma=SECRETO"
MAX_BYTES = 1000


async def con_obtenedor(router: respx.MockRouter | None = None, hosts: list[str] | None = None) -> ObtenedorImagenHttp:
    cliente = httpx.AsyncClient(timeout=5.0)
    return ObtenedorImagenHttp(cliente, ["almacen.example"] if hosts is None else hosts, MAX_BYTES)


async def trozos(*partes: bytes) -> AsyncIterator[bytes]:
    for parte in partes:
        yield parte


@pytest.mark.requisito("RF-19")
async def test_descarga_el_contenido_de_una_url_permitida() -> None:
    with respx.mock() as router:
        ruta = router.get("https://almacen.example/fotos/a.jpg").mock(
            return_value=httpx.Response(200, content=b"imagen")
        )
        assert await (await con_obtenedor(router)).obtener(URL) == b"imagen"
        assert ruta.called


@pytest.mark.requisito("RF-19")
async def test_una_url_fuera_de_la_allowlist_no_hace_ninguna_peticion() -> None:
    with respx.mock(assert_all_called=False) as router:
        ruta = router.route().mock(return_value=httpx.Response(200, content=b"x"))
        with pytest.raises(ImagenNoDisponible):
            await (await con_obtenedor(router)).obtener("https://interno.local/secreto")
        with pytest.raises(ImagenNoDisponible):
            await (await con_obtenedor(router)).obtener("http://almacen.example/a.jpg")  # sin TLS
        assert not ruta.called


@pytest.mark.requisito("RF-19")
async def test_no_sigue_redirecciones_aunque_apunten_a_un_host_interno() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("https://almacen.example/fotos/a.jpg").mock(
            return_value=httpx.Response(302, headers={"location": "https://interno.local/secreto"}),
        )
        interno = router.get("https://interno.local/secreto").mock(return_value=httpx.Response(200, content=b"x"))
        with pytest.raises(ImagenNoDisponible):
            await (await con_obtenedor(router)).obtener(URL)
        assert not interno.called


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("estado", [403, 404, 500])
async def test_una_respuesta_que_no_es_200_es_imagen_no_disponible(estado: int) -> None:
    with respx.mock() as router:
        router.get("https://almacen.example/fotos/a.jpg").mock(return_value=httpx.Response(estado))
        with pytest.raises(ImagenNoDisponible):
            await (await con_obtenedor(router)).obtener(URL)


@pytest.mark.requisito("RF-19")
async def test_un_content_length_mayor_al_maximo_se_rechaza_sin_leer_el_cuerpo() -> None:
    with respx.mock() as router:
        router.get("https://almacen.example/fotos/a.jpg").mock(
            return_value=httpx.Response(200, headers={"content-length": str(MAX_BYTES + 1)}, content=b"x"),
        )
        with pytest.raises(ImagenNoDisponible, match="tamaño"):
            await (await con_obtenedor(router)).obtener(URL)


@pytest.mark.requisito("RF-19")
async def test_un_cuerpo_que_se_pasa_del_maximo_sin_content_length_se_corta() -> None:
    with respx.mock() as router:
        router.get("https://almacen.example/fotos/a.jpg").mock(
            return_value=httpx.Response(200, content=trozos(b"x" * 600, b"x" * 600)),  # sin Content-Length
        )
        with pytest.raises(ImagenNoDisponible, match="tamaño"):
            await (await con_obtenedor(router)).obtener(URL)


@pytest.mark.requisito("RF-19")
async def test_justo_en_el_maximo_se_acepta() -> None:
    with respx.mock() as router:
        router.get("https://almacen.example/fotos/a.jpg").mock(
            return_value=httpx.Response(200, content=trozos(b"x" * 500, b"x" * 500)),
        )
        assert len(await (await con_obtenedor(router)).obtener(URL)) == MAX_BYTES


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("error", [httpx.ConnectTimeout("t"), httpx.ReadTimeout("t"), httpx.ConnectError("c")])
async def test_timeouts_y_errores_de_conexion_son_imagen_no_disponible(error: httpx.HTTPError) -> None:
    with respx.mock() as router:
        router.get("https://almacen.example/fotos/a.jpg").mock(side_effect=error)
        with pytest.raises(ImagenNoDisponible):
            await (await con_obtenedor(router)).obtener(URL)


@pytest.mark.requisito("RF-19")
async def test_ni_el_error_ni_el_log_incluyen_la_url_firmada(caplog: pytest.LogCaptureFixture) -> None:
    with respx.mock() as router:
        router.get("https://almacen.example/fotos/a.jpg").mock(side_effect=httpx.ConnectError("no conecta"))
        with caplog.at_level("DEBUG"), pytest.raises(ImagenNoDisponible) as error:
            await (await con_obtenedor(router)).obtener(URL)
    assert "SECRETO" not in str(error.value)
    assert "SECRETO" not in caplog.text
