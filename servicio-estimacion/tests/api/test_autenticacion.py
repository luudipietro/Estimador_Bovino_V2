from pathlib import Path

import pytest
from fastapi import Depends, FastAPI

from bv_estimacion.api.seguridad import exigir_token
from tests.api.ayudas import AUTORIZACION, servicio
from tests.modelo_falso import TOKEN


def rutas(app: FastAPI) -> None:
    @app.get("/_protegida", dependencies=[Depends(exigir_token)])
    async def protegida() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/_abierta")
    async def abierta() -> dict[str, bool]:
        return {"ok": True}


@pytest.mark.requisito("RF-22")
async def test_con_el_token_correcto_la_ruta_protegida_responde(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_protegida", headers=AUTORIZACION)
    assert respuesta.status_code == 200


@pytest.mark.requisito("RF-22")
async def test_sin_encabezado_es_401_no_autenticado_con_www_authenticate(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_protegida")

    assert respuesta.status_code == 401
    assert respuesta.headers["content-type"] == "application/problem+json"
    assert respuesta.headers["www-authenticate"] == "Bearer"
    assert respuesta.json()["codigo"] == "no_autenticado"


@pytest.mark.requisito("RF-22")
@pytest.mark.parametrize(
    "encabezado",
    [
        "Bearer token-equivocado",
        "Bearer ",
        "Bearer",
        f"Basic {TOKEN}",  # otro esquema con el token correcto
        TOKEN,  # el token solo, sin esquema
        f"Bearer {TOKEN} extra",
        f"Bearer  {TOKEN}",  # dos espacios
        f"Bearer {TOKEN}x",
        f"Bearer {TOKEN[:-1]}",
    ],
)
async def test_un_token_ausente_incompleto_o_incorrecto_es_401(tmp_path: Path, encabezado: str) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_protegida", headers={"Authorization": encabezado})
    assert respuesta.status_code == 401
    assert respuesta.json()["codigo"] == "no_autenticado"


@pytest.mark.requisito("RF-22")
async def test_el_nombre_del_esquema_no_distingue_mayusculas(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_protegida", headers={"Authorization": f"bearer {TOKEN}"})
    assert respuesta.status_code == 200


@pytest.mark.requisito("RF-22")
async def test_el_mensaje_no_distingue_entre_falta_de_token_y_token_incorrecto(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        sin_token = await cliente.get("/_protegida")
        equivocado = await cliente.get("/_protegida", headers={"Authorization": "Bearer otro"})
    assert sin_token.json()["detail"] == equivocado.json()["detail"]
    assert TOKEN not in sin_token.text + equivocado.text


@pytest.mark.requisito("RF-22")
async def test_el_token_se_compara_en_tiempo_constante(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    comparaciones: list[tuple[bytes, bytes]] = []
    original = secrets.compare_digest

    def espia(a: bytes, b: bytes) -> bool:
        comparaciones.append((a, b))
        return original(a, b)

    monkeypatch.setattr("bv_estimacion.api.seguridad.secrets.compare_digest", espia)
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        await cliente.get("/_protegida", headers={"Authorization": "Bearer otro"})

    assert comparaciones == [(b"otro", TOKEN.encode())]


@pytest.mark.requisito("RF-22")
async def test_un_token_con_caracteres_no_ascii_es_401_y_no_un_error(tmp_path: Path) -> None:
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        respuesta = await cliente.get("/_protegida", headers={"Authorization": "Bearer ñandú".encode()})
    assert respuesta.status_code == 401


@pytest.mark.requisito("RF-22")
async def test_una_ruta_sin_la_dependencia_queda_abierta_por_eso_solo_salud_lo_esta(tmp_path: Path) -> None:
    """Prueba que la autenticación es por ruta y no un filtro global: cada router nuevo debe pedirla."""
    async with servicio(tmp_path, rutas=rutas) as (_, cliente):
        assert (await cliente.get("/_abierta")).status_code == 200
