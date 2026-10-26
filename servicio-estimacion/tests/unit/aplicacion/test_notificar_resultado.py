import logging

import pytest

from bv_estimacion.aplicacion.casos_uso.notificar_resultado import NotificarResultado
from bv_estimacion.dominio.estimacion import Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen
from tests.fakes import T0, DormidorFalso, EnviadorCallbackFalso

URL = "https://backend.example/callback"


def trabajo(url: str | None = URL) -> Trabajo:
    referencia = ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, "tmp-1", "a" * 64)
    return Trabajo.crear(IdTrabajo("trb_01"), IdEvento("evt-1"), [referencia], T0, url_notificacion=url)


def armar(fallos: int = 0, intentos: int = 3) -> tuple[NotificarResultado, EnviadorCallbackFalso, DormidorFalso]:
    enviador = EnviadorCallbackFalso(fallos)
    dormidor = DormidorFalso()
    return NotificarResultado(enviador, dormidor, intentos=intentos, espera_base_s=1.0), enviador, dormidor


@pytest.mark.requisito("RF-21")
async def test_sin_url_de_notificacion_no_se_envia_nada() -> None:
    caso, enviador, _ = armar()
    assert not await caso.ejecutar(trabajo(url=None))
    assert enviador.intentos == []


@pytest.mark.requisito("RF-21")
async def test_se_entrega_al_primer_intento_sin_esperar() -> None:
    caso, enviador, dormidor = armar()
    assert await caso.ejecutar(trabajo())
    assert enviador.intentos == [URL]
    assert dormidor.esperas == []


@pytest.mark.requisito("RF-21")
async def test_reintenta_con_backoff_exponencial_hasta_entregar() -> None:
    caso, enviador, dormidor = armar(fallos=2)
    assert await caso.ejecutar(trabajo())
    assert len(enviador.intentos) == 3
    assert dormidor.esperas == [1.0, 2.0]  # base, base * 2


@pytest.mark.requisito("RF-21")
async def test_si_se_agotan_los_intentos_devuelve_false_y_no_lanza() -> None:
    caso, enviador, dormidor = armar(fallos=99)
    original = trabajo()

    assert not await caso.ejecutar(original)

    assert len(enviador.intentos) == 3
    assert dormidor.esperas == [1.0, 2.0]  # no se espera después del último intento
    assert original == trabajo()  # la falla del callback no cambia el trabajo (sigue disponible por GET)


@pytest.mark.requisito("RF-21")
async def test_un_error_inesperado_del_envio_tampoco_escapa_de_la_tarea_de_fondo() -> None:
    caso, enviador, _ = armar()
    enviador.error_inesperado = RuntimeError("se cortó la conexión")
    assert not await caso.ejecutar(trabajo())
    assert len(enviador.intentos) == 3


@pytest.mark.requisito("RF-21")
async def test_el_log_registra_el_id_evento_y_el_numero_de_intento(caplog: pytest.LogCaptureFixture) -> None:
    caso, _, _ = armar(fallos=2)
    with caplog.at_level(logging.INFO):
        await caso.ejecutar(trabajo())

    por_evento = {(r.getMessage(), getattr(r, "intento", None)): r for r in caplog.records}
    assert set(por_evento) == {("callback_fallido", 1), ("callback_fallido", 2), ("callback_entregado", 3)}
    assert all(getattr(r, "id_evento") == "evt-1" for r in caplog.records)  # noqa: B009


@pytest.mark.requisito("RF-21")
async def test_al_agotar_los_intentos_el_log_lo_dice(caplog: pytest.LogCaptureFixture) -> None:
    caso, _, _ = armar(fallos=99)
    with caplog.at_level(logging.INFO):
        await caso.ejecutar(trabajo())
    assert [r.getMessage() for r in caplog.records][-1] == "callback_agotado"


@pytest.mark.requisito("RF-21")
def test_hace_falta_al_menos_un_intento() -> None:
    with pytest.raises(ValueError):
        NotificarResultado(EnviadorCallbackFalso(), DormidorFalso(), intentos=0, espera_base_s=1.0)
