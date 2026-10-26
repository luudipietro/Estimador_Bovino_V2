from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from tests.api.ayudas import RUTA, archivos, encabezados, foto, servicio

MIB = 1024 * 1024
LIMITE_JSON = 64 * 1024
FRONTERA = "frontera-de-prueba"


def cuerpo_multipart_enorme(megas: int) -> tuple[bytes, list[bytes]]:
    """Un multipart válido con una imagen de `megas` MiB, partido en bloques de 1 MiB."""
    cabecera = (
        f"--{FRONTERA}\r\n"
        'Content-Disposition: form-data; name="imagenes"; filename="enorme.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    cierre = f"\r\n--{FRONTERA}--\r\n".encode()
    return cabecera, [b"\xff" * MIB for _ in range(megas)] + [cierre]


async def en_trozos(cabecera: bytes, bloques: list[bytes]) -> AsyncIterator[bytes]:
    """Un cuerpo sin `Content-Length` (`Transfer-Encoding: chunked`): el tope no puede salir del encabezado."""
    yield cabecera
    for bloque in bloques:
        yield bloque


def tipo_multipart() -> dict[str, str]:
    return {"Content-Type": f"multipart/form-data; boundary={FRONTERA}"}


@pytest.mark.requisito("RF-15")
async def test_un_cuerpo_que_declara_ser_enorme_se_rechaza_con_413_sin_leerlo(tmp_path: Path) -> None:
    """El tope es 4 imágenes de `MAX_MB_IMAGEN` más 1 MiB de margen: con 1 MB, 5 MiB."""
    cabecera, bloques = cuerpo_multipart_enorme(8)
    async with servicio(tmp_path, max_mb_imagen=1) as (_, cliente):
        respuesta = await cliente.post(
            RUTA, headers=encabezados() | tipo_multipart(), content=cabecera + b"".join(bloques)
        )

    assert respuesta.status_code == 413
    assert respuesta.json()["codigo"] == "imagen_muy_grande"
    assert respuesta.headers["content-type"] == "application/problem+json"


@pytest.mark.requisito("RF-15")
async def test_sin_content_length_el_tope_se_aplica_mientras_se_lee_el_cuerpo(tmp_path: Path) -> None:
    """Un cliente puede mandar `Transfer-Encoding: chunked` y ahorrarse el `Content-Length`: el límite real está en la
    lectura, no en el encabezado."""
    cabecera, bloques = cuerpo_multipart_enorme(8)
    async with servicio(tmp_path, max_mb_imagen=1) as (_, cliente):
        pedido = cliente.build_request(
            "POST", RUTA, headers=encabezados() | tipo_multipart(), content=en_trozos(cabecera, bloques)
        )
        assert "content-length" not in pedido.headers
        assert pedido.headers["transfer-encoding"] == "chunked"
        respuesta = await cliente.send(pedido)

    assert respuesta.status_code == 413
    assert respuesta.json()["codigo"] == "imagen_muy_grande"


@pytest.mark.requisito("RF-15")
async def test_un_cuerpo_dentro_del_tope_pero_con_un_archivo_mas_grande_que_el_maximo_es_413_por_archivo(
    tmp_path: Path,
) -> None:
    cabecera, bloques = cuerpo_multipart_enorme(2)  # 2 MiB: entra en el tope de 5 MiB, no en el de 1 MB por imagen
    async with servicio(tmp_path, max_mb_imagen=1) as (_, cliente):
        respuesta = await cliente.post(
            RUTA, headers=encabezados() | tipo_multipart(), content=cabecera + b"".join(bloques)
        )
    assert respuesta.status_code == 413
    assert respuesta.json()["codigo"] == "imagen_muy_grande"


@pytest.mark.requisito("RF-15")
async def test_cuatro_imagenes_normales_entran_dentro_del_tope(tmp_path: Path) -> None:
    async with servicio(tmp_path, max_mb_imagen=1) as (_, cliente):
        respuesta = await cliente.post(RUTA, headers=encabezados(), files=archivos(*(foto(i) for i in range(4))))
    assert respuesta.status_code == 202


@pytest.mark.requisito("RF-19")
async def test_un_json_de_mas_de_64_kb_es_imagen_muy_grande(tmp_path: Path) -> None:
    relleno = "x" * (LIMITE_JSON + 1)
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(
            RUTA, headers=encabezados(), json={"referencias_imagen": [], "metadatos": {"nota": relleno}}
        )
    assert respuesta.status_code == 413
    assert respuesta.json()["codigo"] == "imagen_muy_grande"


@pytest.mark.requisito("RF-19")
async def test_un_json_de_mas_de_64_kb_sin_content_length_tambien_se_corta(tmp_path: Path) -> None:
    async def json_enorme() -> AsyncIterator[bytes]:
        yield b'{"referencias_imagen": [], "metadatos": {"nota": "'
        for _ in range(3):
            yield b"x" * LIMITE_JSON
        yield b'"}}'

    async with servicio(tmp_path) as (_, cliente):
        pedido = cliente.build_request(
            "POST", RUTA, headers=encabezados() | {"Content-Type": "application/json"}, content=json_enorme()
        )
        assert "content-length" not in pedido.headers
        respuesta = await cliente.send(pedido)

    assert respuesta.status_code == 413
    assert respuesta.json()["codigo"] == "imagen_muy_grande"


@pytest.mark.requisito("RF-19")
async def test_un_json_justo_dentro_del_tope_se_procesa(tmp_path: Path) -> None:
    """El tope es de 64 KiB: un JSON de ese tamaño llega hasta la validación normal (acá, 0 referencias → 422)."""
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(
            RUTA, headers=encabezados(), json={"referencias_imagen": [], "metadatos": {"nota": "x" * 60_000}}
        )
    assert respuesta.status_code == 422
    assert respuesta.json()["codigo"] == "cantidad_imagenes_invalida"


@pytest.mark.requisito("RF-15")
async def test_el_tope_de_validaciones_es_el_de_una_sola_imagen(tmp_path: Path) -> None:
    cabecera, bloques = cuerpo_multipart_enorme(4)  # 4 MiB: pasaría en /v1/estimaciones (tope de 5 MiB), no acá (2 MiB)
    cabecera = cabecera.replace(b'name="imagenes"', b'name="imagen"')
    async with servicio(tmp_path, max_mb_imagen=1) as (_, cliente):
        respuesta = await cliente.post(
            "/v1/validaciones", headers=encabezados() | tipo_multipart(), content=cabecera + b"".join(bloques)
        )
    assert respuesta.status_code == 413


@pytest.mark.requisito("RF-15")
async def test_un_pedido_rechazado_por_tamano_no_deja_archivos_temporales(tmp_path: Path) -> None:
    cabecera, bloques = cuerpo_multipart_enorme(8)
    async with servicio(tmp_path, max_mb_imagen=1) as (_, cliente):
        await cliente.post(RUTA, headers=encabezados() | tipo_multipart(), content=cabecera + b"".join(bloques))
    assert not (tmp_path / "temporal").exists() or not list((tmp_path / "temporal").iterdir())
