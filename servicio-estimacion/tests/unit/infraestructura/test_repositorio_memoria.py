import asyncio
from datetime import datetime, timedelta

import pytest

from bv_estimacion.dominio.errores import CodigoError
from bv_estimacion.dominio.estimacion import EstadoTrabajo, Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen
from bv_estimacion.infraestructura.trabajos.memoria import RepositorioTrabajosMemoria
from tests.fakes import T0, RelojFijo

TTL_S = 3600


def pendiente(id_trabajo: str = "trb_01", id_evento: str = "evt-1", sha: str = "a") -> Trabajo:
    referencia = ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, f"tmp-{id_trabajo}", sha * 64)
    return Trabajo.crear(IdTrabajo(id_trabajo), IdEvento(id_evento), [referencia], T0)


def terminado(trabajo: Trabajo, cuando: datetime = T0) -> Trabajo:
    return trabajo.iniciar(cuando).fallar(CodigoError.ERROR_INTERNO, cuando, reintentable=False)


def armar() -> tuple[RepositorioTrabajosMemoria, RelojFijo]:
    reloj = RelojFijo()
    return RepositorioTrabajosMemoria(TTL_S, reloj), reloj


@pytest.mark.requisito("RF-19")
async def test_agregar_guarda_el_trabajo_y_se_obtiene_por_id_y_por_id_evento() -> None:
    repo, _ = armar()
    trabajo = pendiente()
    assert await repo.agregar(trabajo) == trabajo
    assert await repo.obtener(trabajo.id) == trabajo
    assert await repo.buscar_por_id_evento(trabajo.id_evento) == trabajo


@pytest.mark.requisito("RF-19")
async def test_agregar_con_un_id_evento_existente_devuelve_el_existente_y_no_pisa_nada() -> None:
    repo, _ = armar()
    primero = pendiente("trb_01")
    await repo.agregar(primero)

    devuelto = await repo.agregar(pendiente("trb_02"))  # mismo id_evento "evt-1"

    assert devuelto == primero
    assert await repo.obtener(IdTrabajo("trb_02")) is None
    assert len(repo) == 1


@pytest.mark.requisito("RF-19")
async def test_agregar_es_atomico_con_50_pedidos_simultaneos_del_mismo_id_evento() -> None:
    repo, _ = armar()
    candidatos = [pendiente(f"trb_{i:02d}") for i in range(50)]

    devueltos = await asyncio.gather(*(repo.agregar(t) for t in candidatos))

    assert len({t.id for t in devueltos}) == 1  # todos vieron al mismo ganador
    assert len(repo) == 1


@pytest.mark.requisito("RNF-03")
async def test_50_trabajos_de_distintos_id_evento_se_guardan_todos() -> None:
    repo, _ = armar()
    await asyncio.gather(*(repo.agregar(pendiente(f"trb_{i:02d}", f"evt-{i}")) for i in range(50)))
    assert len(repo) == 50


@pytest.mark.requisito("RF-19")
async def test_guardar_actualiza_el_estado_del_trabajo() -> None:
    repo, _ = armar()
    trabajo = await repo.agregar(pendiente())
    await repo.guardar(trabajo.iniciar(T0))
    guardado = await repo.obtener(trabajo.id)
    assert guardado is not None
    assert guardado.estado is EstadoTrabajo.PROCESANDO


@pytest.mark.requisito("RF-19")
async def test_un_id_inexistente_devuelve_none() -> None:
    repo, _ = armar()
    assert await repo.obtener(IdTrabajo("trb_99")) is None
    assert await repo.buscar_por_id_evento(IdEvento("nunca")) is None


@pytest.mark.requisito("RF-19")
async def test_eliminar_borra_el_trabajo_y_libera_su_id_evento_y_es_idempotente() -> None:
    repo, _ = armar()
    trabajo = await repo.agregar(pendiente())

    await repo.eliminar(trabajo.id)
    await repo.eliminar(trabajo.id)  # borrar dos veces no es un error

    assert await repo.obtener(trabajo.id) is None
    assert await repo.buscar_por_id_evento(trabajo.id_evento) is None
    assert (await repo.agregar(pendiente("trb_02"))).id == IdTrabajo("trb_02")  # el id_evento quedó libre


# --- TTL ----------------------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_un_trabajo_pendiente_o_procesando_no_vence_nunca() -> None:
    repo, reloj = armar()
    trabajo = await repo.agregar(pendiente())
    await repo.guardar(trabajo.iniciar(T0))

    reloj.avanzar(segundos=TTL_S * 10)

    assert await repo.obtener(trabajo.id) is not None
    assert await repo.purgar_vencidos() == 0


@pytest.mark.requisito("RF-19")
async def test_un_trabajo_terminado_se_conserva_hasta_su_ttl_y_despues_es_404() -> None:
    repo, reloj = armar()
    trabajo = terminado(pendiente())
    await repo.agregar(trabajo)

    reloj.avanzar(segundos=TTL_S - 1)
    assert await repo.obtener(trabajo.id) == trabajo

    reloj.avanzar(segundos=2)
    assert await repo.obtener(trabajo.id) is None
    assert await repo.buscar_por_id_evento(trabajo.id_evento) is None
    assert len(repo) == 0


@pytest.mark.requisito("RF-19")
async def test_el_ttl_cuenta_desde_que_termina_y_no_desde_que_se_creo() -> None:
    repo, reloj = armar()
    trabajo = await repo.agregar(pendiente())
    reloj.avanzar(segundos=TTL_S * 5)  # estuvo mucho tiempo en la cola
    await repo.guardar(terminado(trabajo, cuando=reloj.ahora()))

    reloj.avanzar(segundos=TTL_S - 1)

    assert await repo.obtener(trabajo.id) is not None


@pytest.mark.requisito("RF-19")
async def test_un_id_evento_vencido_se_puede_volver_a_usar() -> None:
    repo, reloj = armar()
    await repo.agregar(terminado(pendiente("trb_01")))
    reloj.avanzar(segundos=TTL_S + 1)

    nuevo = await repo.agregar(pendiente("trb_02"))

    assert nuevo.id == IdTrabajo("trb_02")


@pytest.mark.requisito("RF-19")
async def test_purgar_vencidos_borra_solo_los_vencidos_y_cuenta() -> None:
    repo, reloj = armar()
    await repo.agregar(terminado(pendiente("trb_01", "evt-1")))
    await repo.agregar(terminado(pendiente("trb_02", "evt-2")))
    await repo.agregar(pendiente("trb_03", "evt-3"))  # sigue pendiente
    reloj.avanzar(segundos=TTL_S + 1)

    assert await repo.purgar_vencidos() == 2
    assert len(repo) == 1
    assert await repo.obtener(IdTrabajo("trb_03")) is not None
    assert await repo.purgar_vencidos() == 0


@pytest.mark.requisito("RF-19")
def test_el_ttl_se_aplica_con_el_reloj_inyectado_y_no_con_la_hora_real() -> None:
    reloj = RelojFijo(T0 + timedelta(days=365))
    assert len(RepositorioTrabajosMemoria(TTL_S, reloj)) == 0
