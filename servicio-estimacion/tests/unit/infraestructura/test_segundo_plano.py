import logging
from collections.abc import Callable

import anyio
import pytest

from bv_estimacion.dominio.estimacion import Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen
from bv_estimacion.infraestructura.notificacion.segundo_plano import NotificadorEnSegundoPlano
from tests.fakes import T0


def trabajo(url: str | None = "https://backend.example/cb") -> Trabajo:
    referencia = ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, "tmp-1", "a" * 64)
    return Trabajo.crear(IdTrabajo("trb_01"), IdEvento("evt-1"), [referencia], T0, url_notificacion=url)


async def esperar_hasta(condicion: Callable[[], bool], segundos: float = 3.0) -> None:
    with anyio.fail_after(segundos):
        while not condicion():
            await anyio.sleep(0.005)


class Entrega:
    """Un `NotificarResultado.ejecutar` de mentira que se puede frenar y hacer fallar."""

    def __init__(self) -> None:
        self.recibidos: list[Trabajo] = []
        self.terminadas = 0
        self.compuerta: anyio.Event | None = None
        self.error: Exception | None = None

    async def __call__(self, trabajo: Trabajo) -> bool:
        self.recibidos.append(trabajo)
        if self.compuerta is not None:
            await self.compuerta.wait()
        if self.error is not None:
            raise self.error
        self.terminadas += 1
        return True


@pytest.mark.requisito("RF-21")
async def test_notificar_vuelve_enseguida_sin_esperar_la_entrega() -> None:
    entrega = Entrega()
    entrega.compuerta = anyio.Event()  # la entrega queda esperando al backend
    async with anyio.create_task_group() as grupo:
        notificador = NotificadorEnSegundoPlano(entrega, grupo)

        with anyio.fail_after(1):  # si esperara la entrega, esto se colgaría
            await notificador.notificar(trabajo())

        await esperar_hasta(lambda: len(entrega.recibidos) == 1)
        assert entrega.terminadas == 0  # la entrega sigue en curso, y quien notificó ya siguió

        entrega.compuerta.set()
        await esperar_hasta(lambda: entrega.terminadas == 1)


@pytest.mark.requisito("RF-21")
async def test_entrega_el_mismo_trabajo_que_se_notifico() -> None:
    entrega = Entrega()
    async with anyio.create_task_group() as grupo:
        original = trabajo()
        await NotificadorEnSegundoPlano(entrega, grupo).notificar(original)
        await esperar_hasta(lambda: entrega.terminadas == 1)
    assert entrega.recibidos == [original]


@pytest.mark.requisito("RF-21")
async def test_sin_url_de_notificacion_no_se_agenda_nada() -> None:
    entrega = Entrega()
    async with anyio.create_task_group() as grupo:
        await NotificadorEnSegundoPlano(entrega, grupo).notificar(trabajo(url=None))
        await anyio.sleep(0.05)
    assert entrega.recibidos == []


@pytest.mark.requisito("RF-21")
async def test_una_entrega_que_falla_no_cancela_al_grupo_ni_a_las_demas_tareas(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Si el error escapara, cancelaría el grupo y con él a los workers que comparten el `lifespan`."""
    entrega = Entrega()
    entrega.error = RuntimeError("detalle interno")
    otra_tarea_viva: list[bool] = []

    async def otra_tarea() -> None:
        await anyio.sleep(0.1)  # sigue después de que la entrega ya falló
        otra_tarea_viva.append(True)

    with caplog.at_level(logging.ERROR):
        async with anyio.create_task_group() as grupo:
            grupo.start_soon(otra_tarea)
            await NotificadorEnSegundoPlano(entrega, grupo).notificar(trabajo())
            await esperar_hasta(lambda: len(entrega.recibidos) == 1)

    assert otra_tarea_viva == [True]
    registros = [r for r in caplog.records if r.getMessage() == "notificacion_error"]
    assert [getattr(r, "id_evento", None) for r in registros] == ["evt-1"]


@pytest.mark.requisito("RF-21")
async def test_al_apagar_el_grupo_se_cancelan_las_entregas_pendientes() -> None:
    entrega = Entrega()
    entrega.compuerta = anyio.Event()  # nunca se abre
    with anyio.fail_after(3):  # si el apagado se colgara esperando la entrega, falla en vez de quedarse
        async with anyio.create_task_group() as grupo:
            await NotificadorEnSegundoPlano(entrega, grupo).notificar(trabajo())
            await esperar_hasta(lambda: len(entrega.recibidos) == 1)
            grupo.cancel_scope.cancel()
    assert entrega.terminadas == 0
