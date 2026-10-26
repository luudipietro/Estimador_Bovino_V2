import asyncio

import anyio
import pytest

from bv_estimacion.aplicacion.puertos import ColaLlena
from bv_estimacion.dominio.valores import IdTrabajo
from bv_estimacion.infraestructura.trabajos.memoria import ColaTrabajosMemoria


def ids(n: int) -> list[IdTrabajo]:
    return [IdTrabajo(f"trb_{i:02d}") for i in range(n)]


@pytest.mark.requisito("RF-19")
async def test_hay_lugar_hasta_llenar_la_capacidad() -> None:
    cola = ColaTrabajosMemoria(2)
    assert await cola.hay_lugar()
    await cola.encolar(IdTrabajo("trb_01"))
    assert await cola.hay_lugar()
    await cola.encolar(IdTrabajo("trb_02"))
    assert not await cola.hay_lugar()
    assert len(cola) == 2


@pytest.mark.requisito("RF-19")
async def test_encolar_con_la_cola_llena_lanza_cola_llena_y_no_pierde_lo_encolado() -> None:
    cola = ColaTrabajosMemoria(1)
    await cola.encolar(IdTrabajo("trb_01"))
    with pytest.raises(ColaLlena):
        await cola.encolar(IdTrabajo("trb_02"))
    assert await cola.siguiente() == IdTrabajo("trb_01")


@pytest.mark.requisito("RF-19")
async def test_sale_en_el_orden_en_que_entro() -> None:
    cola = ColaTrabajosMemoria(10)
    for id_trabajo in ids(5):
        await cola.encolar(id_trabajo)
    assert [await cola.siguiente() for _ in range(5)] == ids(5)


@pytest.mark.requisito("RF-19")
async def test_al_sacar_uno_vuelve_a_haber_lugar() -> None:
    cola = ColaTrabajosMemoria(1)
    await cola.encolar(IdTrabajo("trb_01"))
    await cola.siguiente()
    assert await cola.hay_lugar()


@pytest.mark.requisito("RF-19")
async def test_siguiente_espera_hasta_que_llegue_un_trabajo() -> None:
    cola = ColaTrabajosMemoria(5)
    with anyio.move_on_after(0.05) as espera:
        await cola.siguiente()
    assert espera.cancelled_caught  # no devolvió nada: se quedó esperando

    async def producir() -> None:
        await anyio.sleep(0.02)
        await cola.encolar(IdTrabajo("trb_07"))

    resultado: list[IdTrabajo] = []
    async with anyio.create_task_group() as grupo:
        grupo.start_soon(producir)
        with anyio.fail_after(2):
            resultado.append(await cola.siguiente())
    assert resultado == [IdTrabajo("trb_07")]


@pytest.mark.requisito("RNF-03")
async def test_50_pedidos_simultaneos_entran_todos_si_hay_capacidad() -> None:
    cola = ColaTrabajosMemoria(200)
    resultados = await asyncio.gather(*(cola.encolar(i) for i in ids(50)), return_exceptions=True)
    assert all(r is None for r in resultados)
    assert len(cola) == 50


@pytest.mark.requisito("RNF-03")
async def test_con_50_pedidos_simultaneos_y_capacidad_10_entran_exactamente_10() -> None:
    cola = ColaTrabajosMemoria(10)
    resultados = await asyncio.gather(*(cola.encolar(i) for i in ids(50)), return_exceptions=True)
    assert sum(r is None for r in resultados) == 10
    assert sum(isinstance(r, ColaLlena) for r in resultados) == 40
    assert len(cola) == 10
