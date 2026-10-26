import threading
from collections.abc import Callable

import anyio
import pytest

from bv_estimacion.infraestructura.ml.pool import PoolDeInstancias


class Instancia:
    """Una 'instancia del modelo' que detecta si dos hilos la usan a la vez (lo que Ultralytics no tolera)."""

    def __init__(self, nombre: str) -> None:
        self.nombre = nombre
        self.en_uso = 0
        self.maximo_en_uso = 0
        self._candado = threading.Lock()

    def entrar(self) -> None:
        with self._candado:
            self.en_uso += 1
            self.maximo_en_uso = max(self.maximo_en_uso, self.en_uso)

    def salir(self) -> None:
        with self._candado:
            self.en_uso -= 1


async def esperar_hasta(condicion: Callable[[], bool], segundos: float = 3.0) -> None:
    with anyio.fail_after(segundos):
        while not condicion():
            await anyio.sleep(0.005)


@pytest.mark.requisito("RNF-03")
async def test_devuelve_lo_que_devuelve_la_funcion_y_la_instancia_queda_libre() -> None:
    pool = PoolDeInstancias([Instancia("a")])
    assert await pool.ejecutar(lambda i: i.nombre) == "a"
    assert pool.libres == 1


@pytest.mark.requisito("RNF-03")
async def test_corre_fuera_del_hilo_del_event_loop() -> None:
    pool = PoolDeInstancias([Instancia("a")])
    hilo_del_loop = threading.get_ident()
    assert await pool.ejecutar(lambda _: threading.get_ident()) != hilo_del_loop


@pytest.mark.requisito("RNF-03")
async def test_nunca_hay_dos_hilos_en_la_misma_instancia() -> None:
    instancias = [Instancia("a"), Instancia("b")]
    pool = PoolDeInstancias(instancias)

    def trabajo(instancia: Instancia) -> None:
        instancia.entrar()
        threading.Event().wait(0.02)
        instancia.salir()

    async with anyio.create_task_group() as grupo:
        for _ in range(8):
            grupo.start_soon(pool.ejecutar, trabajo)

    assert [i.maximo_en_uso for i in instancias] == [1, 1]
    assert pool.libres == 2


@pytest.mark.requisito("RNF-03")
async def test_con_n_instancias_corren_a_lo_sumo_n_a_la_vez() -> None:
    pool = PoolDeInstancias([Instancia("a"), Instancia("b")])
    simultaneos = 0
    maximo = 0
    candado = threading.Lock()

    def trabajo(_: Instancia) -> None:
        nonlocal simultaneos, maximo
        with candado:
            simultaneos += 1
            maximo = max(maximo, simultaneos)
        threading.Event().wait(0.03)
        with candado:
            simultaneos -= 1

    async with anyio.create_task_group() as grupo:
        for _ in range(6):
            grupo.start_soon(pool.ejecutar, trabajo)

    assert maximo == 2


@pytest.mark.requisito("RNF-01")
async def test_cancelar_la_espera_no_libera_la_instancia_hasta_que_el_hilo_termina() -> None:
    """El contrato de `ModeloPeso.estimar`: tras un timeout el hilo sigue corriendo y su lugar sigue ocupado."""
    instancia = Instancia("a")
    pool = PoolDeInstancias([instancia])
    arrancó = threading.Event()
    soltar = threading.Event()

    def colgado(i: Instancia) -> None:
        i.entrar()
        arrancó.set()
        soltar.wait(5)
        i.salir()

    with anyio.move_on_after(0.05) as alcance:  # el timeout del caso de uso
        await pool.ejecutar(colgado)
    assert alcance.cancelled_caught
    assert arrancó.is_set()

    assert pool.libres == 0  # el hilo sigue vivo: el lugar sigue ocupado

    # un segundo trabajo tiene que esperar ese lugar, no arrancar otro hilo sobre la misma instancia
    with anyio.move_on_after(0.1) as espera:
        await pool.ejecutar(lambda i: i.entrar())
    assert espera.cancelled_caught
    assert instancia.maximo_en_uso == 1

    soltar.set()  # el hilo termina de verdad
    await esperar_hasta(lambda: pool.libres == 1)
    assert instancia.en_uso == 0
    assert await pool.ejecutar(lambda i: i.nombre) == "a"  # y ahora sí hay lugar


@pytest.mark.requisito("RNF-01")
async def test_no_se_acumulan_hilos_colgados_por_trabajos_que_vencen() -> None:
    pool = PoolDeInstancias([Instancia("a")])
    soltar = threading.Event()
    hilos_activos = 0
    maximo = 0
    candado = threading.Lock()

    def colgado(_: Instancia) -> None:
        nonlocal hilos_activos, maximo
        with candado:
            hilos_activos += 1
            maximo = max(maximo, hilos_activos)
        soltar.wait(5)
        with candado:
            hilos_activos -= 1

    for _ in range(3):  # tres trabajos seguidos que se pasan de tiempo
        with anyio.move_on_after(0.03):
            await pool.ejecutar(colgado)

    assert maximo == 1
    soltar.set()
    await esperar_hasta(lambda: pool.libres == 1)


@pytest.mark.requisito("RNF-03")
async def test_si_la_funcion_falla_la_instancia_se_devuelve_y_el_error_se_propaga() -> None:
    pool = PoolDeInstancias([Instancia("a")])

    def falla(_: Instancia) -> None:
        raise RuntimeError("falló la inferencia")

    with pytest.raises(RuntimeError, match="falló la inferencia"):
        await pool.ejecutar(falla)
    assert pool.libres == 1


@pytest.mark.requisito("RNF-03")
async def test_cancelar_mientras_espera_un_lugar_no_pierde_ninguna_instancia() -> None:
    pool = PoolDeInstancias([Instancia("a")])
    soltar = threading.Event()
    async with anyio.create_task_group() as grupo:
        grupo.start_soon(pool.ejecutar, lambda _: soltar.wait(5))
        await esperar_hasta(lambda: pool.libres == 0)
        with anyio.move_on_after(0.03):
            await pool.ejecutar(lambda i: i.nombre)  # espera y se cancela: no tenía ninguna instancia
        soltar.set()
    assert pool.libres == 1


def test_un_pool_vacio_no_tiene_sentido() -> None:
    with pytest.raises(ValueError, match="al menos una instancia"):
        PoolDeInstancias([])


@pytest.mark.requisito("RNF-03")
async def test_el_tamano_es_la_cantidad_de_instancias() -> None:
    assert PoolDeInstancias([Instancia("a"), Instancia("b"), Instancia("c")]).tamano == 3


# --- Los bordes de la carrera con la cancelación y con el apagado --------------------------------


@pytest.mark.requisito("RNF-03")
async def test_si_cancelan_antes_de_que_el_hilo_arranque_la_instancia_se_devuelve_una_sola_vez(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La carrera que `_Ficha` resuelve: la cancelación llega, quien llamó devuelve la instancia y recién después
    el hilo arranca. El hilo no tiene que usar la instancia ni devolverla por segunda vez."""
    pendientes: list[Callable[[], object]] = []

    async def run_sync_que_cancela(funcion: Callable[[], object], **_: object) -> object:
        pendientes.append(funcion)  # el hilo todavía no arrancó
        raise anyio.get_cancelled_exc_class()

    monkeypatch.setattr("bv_estimacion.infraestructura.ml.pool.to_thread.run_sync", run_sync_que_cancela)
    usada: list[Instancia] = []
    pool = PoolDeInstancias([Instancia("a")])

    with pytest.raises(anyio.get_cancelled_exc_class()):
        await pool.ejecutar(usada.append)

    assert pool.libres == 1  # la devolvió quien llamó
    assert pendientes[0]() is None  # el hilo arranca tarde: no hace nada
    assert usada == []
    assert pool.libres == 1  # y no la devolvió por segunda vez


@pytest.mark.requisito("RNF-03")
async def test_si_el_event_loop_ya_no_esta_el_hilo_termina_sin_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """En el apagado el loop puede haber terminado cuando el hilo vuelve: no hay a quién devolverle la instancia."""

    async def run_sync_fuera_de_anyio(funcion: Callable[[], object], **_: object) -> object:
        return funcion()  # sin hilo de anyio, `from_thread.run_sync` falla con RuntimeError

    monkeypatch.setattr("bv_estimacion.infraestructura.ml.pool.to_thread.run_sync", run_sync_fuera_de_anyio)
    pool = PoolDeInstancias([Instancia("a")])

    assert await pool.ejecutar(lambda i: i.nombre) == "a"  # el RuntimeError no se propaga
