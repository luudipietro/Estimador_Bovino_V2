import logging
from collections.abc import Callable

import anyio
import pytest

from bv_estimacion.dominio.valores import IdTrabajo
from bv_estimacion.infraestructura.trabajos.ejecutor import EjecutorTrabajos, barrer_periodicamente
from bv_estimacion.infraestructura.trabajos.memoria import ColaTrabajosMemoria


async def esperar_hasta(condicion: Callable[[], bool], segundos: float = 3.0) -> None:
    with anyio.fail_after(segundos):
        while not condicion():
            await anyio.sleep(0.005)


class ProcesadorFalso:
    def __init__(self) -> None:
        self.procesados: list[IdTrabajo] = []
        self.fallar_con: dict[IdTrabajo, Exception] = {}
        self.en_curso = 0
        self.maximo_en_curso = 0
        self.compuerta: anyio.Event | None = None  # si está, los trabajos esperan a que se abra

    async def ejecutar(self, id_trabajo: IdTrabajo) -> None:
        self.en_curso += 1
        self.maximo_en_curso = max(self.maximo_en_curso, self.en_curso)
        try:
            if self.compuerta is not None:
                await self.compuerta.wait()
            if id_trabajo in self.fallar_con:
                raise self.fallar_con[id_trabajo]
            self.procesados.append(id_trabajo)
        finally:
            self.en_curso -= 1


def ids(n: int) -> list[IdTrabajo]:
    return [IdTrabajo(f"trb_{i:02d}") for i in range(n)]


@pytest.mark.requisito("RF-19")
async def test_procesa_los_trabajos_que_llegan_a_la_cola() -> None:
    cola, procesador = ColaTrabajosMemoria(10), ProcesadorFalso()
    async with anyio.create_task_group() as grupo:
        grupo.start_soon(EjecutorTrabajos(cola, procesador, 2).correr)
        for id_trabajo in ids(5):
            await cola.encolar(id_trabajo)
        await esperar_hasta(lambda: len(procesador.procesados) == 5)
        grupo.cancel_scope.cancel()
    assert len(procesador.procesados) == 5
    assert set(procesador.procesados) == set(ids(5))


@pytest.mark.requisito("RNF-03")
async def test_50_trabajos_se_procesan_todos_con_pocos_workers() -> None:
    cola, procesador = ColaTrabajosMemoria(200), ProcesadorFalso()
    async with anyio.create_task_group() as grupo:
        grupo.start_soon(EjecutorTrabajos(cola, procesador, 4).correr)
        for id_trabajo in ids(50):
            await cola.encolar(id_trabajo)
        await esperar_hasta(lambda: len(procesador.procesados) == 50)
        grupo.cancel_scope.cancel()
    assert len(set(procesador.procesados)) == 50


@pytest.mark.requisito("RNF-03")
async def test_corre_en_paralelo_tantos_trabajos_como_workers_y_no_mas() -> None:
    cola, procesador = ColaTrabajosMemoria(20), ProcesadorFalso()
    procesador.compuerta = anyio.Event()
    async with anyio.create_task_group() as grupo:
        grupo.start_soon(EjecutorTrabajos(cola, procesador, 3).correr)
        for id_trabajo in ids(8):
            await cola.encolar(id_trabajo)
        await esperar_hasta(lambda: procesador.en_curso == 3)
        await anyio.sleep(0.05)  # tiempo de sobra para que un cuarto arranque, si pudiera
        assert procesador.en_curso == 3
        procesador.compuerta.set()
        await esperar_hasta(lambda: len(procesador.procesados) == 8)
        grupo.cancel_scope.cancel()
    assert procesador.maximo_en_curso == 3


@pytest.mark.requisito("RF-22")
async def test_un_error_en_un_trabajo_no_mata_al_worker(caplog: pytest.LogCaptureFixture) -> None:
    cola, procesador = ColaTrabajosMemoria(10), ProcesadorFalso()
    malo, bueno = IdTrabajo("trb_malo"), IdTrabajo("trb_bueno")
    procesador.fallar_con[malo] = RuntimeError("detalle interno")

    with caplog.at_level(logging.ERROR):
        async with anyio.create_task_group() as grupo:
            grupo.start_soon(EjecutorTrabajos(cola, procesador, 1).correr)  # un solo worker: si muriera, no seguiría
            await cola.encolar(malo)
            await cola.encolar(bueno)
            await esperar_hasta(lambda: bueno in procesador.procesados)
            grupo.cancel_scope.cancel()

    registros_de_error = [r for r in caplog.records if r.getMessage() == "worker_error"]
    assert [getattr(r, "id_trabajo", None) for r in registros_de_error] == ["trb_malo"]


@pytest.mark.requisito("RF-22")
async def test_apagar_cancela_a_los_workers_aunque_haya_un_trabajo_en_curso() -> None:
    cola, procesador = ColaTrabajosMemoria(10), ProcesadorFalso()
    procesador.compuerta = anyio.Event()  # nunca se abre: el trabajo queda colgado
    with anyio.fail_after(3):  # si el apagado se cuelga, falla en vez de quedarse esperando
        async with anyio.create_task_group() as grupo:
            grupo.start_soon(EjecutorTrabajos(cola, procesador, 2).correr)
            await cola.encolar(IdTrabajo("trb_01"))
            await esperar_hasta(lambda: procesador.en_curso == 1)
            grupo.cancel_scope.cancel()
    assert procesador.procesados == []
    assert procesador.en_curso == 0  # el `finally` del procesador corrió


@pytest.mark.requisito("RF-22")
async def test_apagar_con_la_cola_vacia_tampoco_se_cuelga() -> None:
    with anyio.fail_after(3):
        async with anyio.create_task_group() as grupo:
            grupo.start_soon(EjecutorTrabajos(ColaTrabajosMemoria(5), ProcesadorFalso(), 4).correr)
            await anyio.sleep(0.02)
            grupo.cancel_scope.cancel()


@pytest.mark.requisito("RNF-03")
def test_hace_falta_al_menos_un_worker() -> None:
    with pytest.raises(ValueError):
        EjecutorTrabajos(ColaTrabajosMemoria(5), ProcesadorFalso(), 0)


# --- Barrido periódico --------------------------------------------------------------------------


class Barrido:
    def __init__(self, devuelve: int = 0, falla: bool = False) -> None:
        self.llamadas = 0
        self._devuelve = devuelve
        self._falla = falla

    async def __call__(self) -> int:
        self.llamadas += 1
        if self._falla:
            raise OSError("disco no disponible")
        return self._devuelve


@pytest.mark.requisito("RF-22")
async def test_el_barrido_corre_cada_intervalo_hasta_que_lo_cancelen() -> None:
    primero, segundo = Barrido(2), Barrido(0)
    async with anyio.create_task_group() as grupo:
        grupo.start_soon(barrer_periodicamente, 0.01, [primero, segundo])
        await esperar_hasta(lambda: primero.llamadas >= 3 and segundo.llamadas >= 3)
        grupo.cancel_scope.cancel()


@pytest.mark.requisito("RF-22")
async def test_un_barrido_que_falla_no_corta_a_los_demas_ni_a_los_ciclos_siguientes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    roto, sano = Barrido(falla=True), Barrido()
    with caplog.at_level(logging.ERROR):
        async with anyio.create_task_group() as grupo:
            grupo.start_soon(barrer_periodicamente, 0.01, [roto, sano])
            await esperar_hasta(lambda: roto.llamadas >= 2 and sano.llamadas >= 2)
            grupo.cancel_scope.cancel()
    assert any(r.getMessage() == "barrido_error" for r in caplog.records)


@pytest.mark.requisito("RF-22")
async def test_antes_del_primer_intervalo_no_se_barre_nada() -> None:
    barrido = Barrido()
    with anyio.move_on_after(0.05):
        await barrer_periodicamente(10.0, [barrido])
    assert barrido.llamadas == 0
