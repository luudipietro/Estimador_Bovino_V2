"""Pool de instancias del modelo: una instancia, un hilo a la vez; el lugar se libera cuando el hilo termina.

Ultralytics no es seguro si varios hilos comparten una instancia, y Python no puede matar un hilo. Entonces, si
quien llama cancela la espera (por ejemplo, por el presupuesto de tiempo), **el hilo sigue corriendo**. `anyio`
devolvería su `CapacityLimiter` al cancelar (con `abandon_on_cancel=True`), así que acá la instancia se devuelve
desde el propio hilo, cuando termina de verdad. Ver el contrato de `ModeloPeso.estimar`.
"""

import threading
from collections import deque
from collections.abc import Callable, Sequence

import anyio
from anyio import from_thread, to_thread


class _Ficha:
    """Decide, una sola vez, quién devuelve la instancia: el hilo (si llegó a arrancar) o quien llama (si lo
    cancelaron antes de que el hilo arrancara). Sin esto, una cancelación justo en el medio la perdería o la
    devolvería dos veces."""

    def __init__(self) -> None:
        self._candado = threading.Lock()
        self._reclamada = False

    def reclamar(self) -> bool:
        with self._candado:
            if self._reclamada:
                return False
            self._reclamada = True
            return True


class PoolDeInstancias[T]:
    """Todo el estado se toca desde el hilo del event loop: el hilo de inferencia devuelve la instancia con
    `from_thread.run_sync`, no escribiendo en el pool."""

    def __init__(self, instancias: Sequence[T]) -> None:
        if not instancias:
            raise ValueError("el pool necesita al menos una instancia")
        self._libres: deque[T] = deque(instancias)
        self._lugares = anyio.Semaphore(len(instancias))
        self._tamano = len(instancias)

    @property
    def tamano(self) -> int:
        return self._tamano

    @property
    def libres(self) -> int:
        return len(self._libres)

    async def ejecutar[R](self, funcion: Callable[[T], R]) -> R:
        """Corre `funcion(instancia)` en un hilo. Espera un lugar si no hay instancias libres."""
        await self._lugares.acquire()
        instancia = self._libres.popleft()
        ficha = _Ficha()

        def en_el_hilo() -> R | None:
            if not ficha.reclamar():
                return None  # nos cancelaron antes de arrancar: quien llamó ya devolvió la instancia
            try:
                return funcion(instancia)
            finally:
                try:
                    from_thread.run_sync(self._devolver, instancia)
                except RuntimeError:
                    pass  # el event loop ya no existe (apagado): no hay a quién devolverle nada

        try:
            resultado = await to_thread.run_sync(en_el_hilo, abandon_on_cancel=True)
        except BaseException:
            if ficha.reclamar():  # el hilo nunca arrancó: la devolvemos nosotros
                self._devolver(instancia)
            raise
        return resultado  # type: ignore[return-value]

    def _devolver(self, instancia: T) -> None:
        self._libres.append(instancia)
        self._lugares.release()
