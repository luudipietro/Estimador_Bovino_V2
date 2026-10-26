from pathlib import Path

import httpx
import pytest

from bv_estimacion.aplicacion.casos_uso.validar_imagen import ValidarImagen
from bv_estimacion.aplicacion.puertos import ModeloNoDisponible
from tests import imagenes_sinteticas as img
from tests.api.ayudas import AUTORIZACION, archivos, foto, servicio
from tests.modelo_falso import FabricaFalsa, respuesta_modelo

RUTA_VALIDACION = "/v1/validaciones"


async def validar(cliente: httpx.AsyncClient, datos: bytes, **campos: str) -> httpx.Response:
    return await cliente.post(RUTA_VALIDACION, headers=AUTORIZACION, files=archivos(datos, campo="imagen"), data=campos)


@pytest.mark.requisito("RF-15")
async def test_una_foto_buena_es_apta_y_dice_si_le_falta_la_metadata(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await validar(cliente, foto())

    assert respuesta.status_code == 200
    assert respuesta.json() == {"apta": True, "motivos": [], "advertencias": ["no trae metadata EXIF"]}


@pytest.mark.requisito("RF-15")
async def test_una_foto_con_exif_no_trae_advertencias(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await validar(cliente, img.jpeg(1024, 768, con_exif=True))
    assert respuesta.json()["advertencias"] == []


@pytest.mark.requisito("R-21")
async def test_una_foto_sin_animal_no_es_apta_pero_no_es_un_error(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    fabrica.respuesta_por_defecto = {
        "peso_kg": None,
        "confiable": False,
        "motivo": "no se detectó ningún bovino en la foto",
    }
    async with servicio(tmp_path, fabrica=fabrica) as (_, cliente):
        respuesta = await validar(cliente, foto())

    assert respuesta.status_code == 200
    assert respuesta.json()["apta"] is False
    assert respuesta.json()["motivos"] == ["sin_animal"]


@pytest.mark.requisito("RF-15")
async def test_el_peso_fuera_de_rango_no_vuelve_no_apta_a_la_foto(tmp_path: Path) -> None:
    """Validar la calidad de la foto no es estimar: `fuera_de_rango` es del peso, no de la imagen."""
    fabrica = FabricaFalsa()
    fabrica.respuesta_por_defecto = respuesta_modelo(700.0)
    async with servicio(tmp_path, fabrica=fabrica) as (_, cliente):
        respuesta = await validar(cliente, foto())
    assert respuesta.json()["apta"] is True
    assert respuesta.json()["motivos"] == []


@pytest.mark.requisito("RF-15")
@pytest.mark.parametrize(
    ("datos", "estado", "codigo"),
    [
        pytest.param(img.gif(), 415, "formato_no_admitido", id="gif"),
        pytest.param(img.jpeg(1024, 768)[:200], 422, "imagen_corrupta", id="jpeg-truncado"),
        pytest.param(img.jpeg(800, 600), 422, "resolucion_insuficiente", id="800x600"),
    ],
)
async def test_una_imagen_invalida_se_rechaza_con_el_mismo_codigo_que_en_estimaciones(
    tmp_path: Path, datos: bytes, estado: int, codigo: str
) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await validar(cliente, datos)
    assert respuesta.status_code == estado
    assert respuesta.json()["codigo"] == codigo


@pytest.mark.requisito("RF-15")
async def test_una_imagen_demasiado_grande_es_413(tmp_path: Path) -> None:
    async with servicio(tmp_path, max_mb_imagen=1) as (_, cliente):
        respuesta = await validar(cliente, img.jpeg(2400, 2000, ruido=True))
    assert respuesta.status_code == 413


@pytest.mark.requisito("RF-15")
async def test_un_heic_tambien_se_valida(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await validar(cliente, img.heic(1024, 768))
    assert respuesta.status_code == 200


@pytest.mark.requisito("RF-15")
async def test_hace_falta_exactamente_una_imagen_en_el_campo_imagen(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        ninguna = await cliente.post(RUTA_VALIDACION, headers=AUTORIZACION, files={"otra": (None, "x")})
        dos = await cliente.post(
            RUTA_VALIDACION, headers=AUTORIZACION, files=archivos(foto(1), foto(2), campo="imagen")
        )
        campo_equivocado = await cliente.post(RUTA_VALIDACION, headers=AUTORIZACION, files=archivos(foto()))
        json_en_vez_de_multipart = await cliente.post(RUTA_VALIDACION, headers=AUTORIZACION, json={})

    for respuesta in (ninguna, dos, campo_equivocado, json_en_vez_de_multipart):
        assert respuesta.status_code == 400
        assert respuesta.json()["codigo"] == "solicitud_invalida"


@pytest.mark.requisito("RF-15")
async def test_validar_no_crea_trabajos_ni_necesita_idempotency_key(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        primera = await validar(cliente, foto())
        segunda = await validar(cliente, foto())
    assert primera.status_code == segunda.status_code == 200


@pytest.mark.requisito("RF-15")
async def test_si_el_modelo_se_pasa_del_presupuesto_es_504_tiempo_excedido(tmp_path: Path) -> None:
    import threading

    fabrica = FabricaFalsa()
    fabrica.frenar = threading.Event()
    async with servicio(tmp_path, fabrica=fabrica, presupuesto_estimacion_s=0.05) as (_, cliente):
        respuesta = await validar(cliente, foto())
        fabrica.frenar.set()

    assert respuesta.status_code == 504
    assert respuesta.json()["codigo"] == "tiempo_excedido"


@pytest.mark.requisito("RF-22")
async def test_si_el_modelo_falla_es_un_500_generico(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    fabrica.error_al_estimar = RuntimeError("detalle interno /srv/app/modelo.py")
    async with servicio(tmp_path, fabrica=fabrica) as (_, cliente):
        respuesta = await validar(cliente, foto())

    assert respuesta.status_code == 500
    assert respuesta.json()["codigo"] == "error_interno"
    assert "/srv/app" not in respuesta.text and "RuntimeError" not in respuesta.text


@pytest.mark.requisito("RF-22")
async def test_validaciones_exige_token(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(RUTA_VALIDACION, files=archivos(foto(), campo="imagen"))
    assert respuesta.status_code == 401


@pytest.mark.requisito("RF-22")
async def test_si_el_modelo_no_esta_disponible_es_503_modelo_no_disponible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def sin_modelo(self: ValidarImagen, contenido: bytes) -> None:
        raise ModeloNoDisponible("el modelo no responde: detalle interno")

    monkeypatch.setattr(ValidarImagen, "ejecutar", sin_modelo)
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await validar(cliente, foto())

    assert respuesta.status_code == 503
    assert respuesta.json()["codigo"] == "modelo_no_disponible"
    assert "detalle interno" not in respuesta.text
