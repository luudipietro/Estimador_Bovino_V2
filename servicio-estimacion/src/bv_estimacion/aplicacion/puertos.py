"""Puertos: lo que la aplicación necesita del mundo exterior. La infraestructura los implementa."""

from datetime import datetime
from typing import Protocol

from bv_estimacion.aplicacion.dto import EstimacionFoto
from bv_estimacion.dominio.estimacion import Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, VersionModelo


class ColaLlena(Exception):
    """La cola no admite más trabajos."""


class ImagenNoDisponible(Exception):
    """No se pudo obtener la imagen de un trabajo (falta, no se pudo descargar o no es la que se declaró)."""


class ModeloNoDisponible(Exception):
    """El modelo no cargó o no responde."""


class CallbackFallido(Exception):
    """El backend no recibió el callback."""


class ModeloPeso(Protocol):
    @property
    def version(self) -> VersionModelo: ...

    async def estimar(self, contenido: bytes) -> EstimacionFoto:
        """Estima una foto. Corre la inferencia fuera del event loop.

        **Contrato de concurrencia.** Si quien llama cancela la espera (por ejemplo, por el presupuesto de
        tiempo), el hilo de inferencia **sigue corriendo**: Python no puede matar un hilo. El adaptador tiene que
        liberar su lugar en el pool cuando el hilo **termina de verdad**, no cuando se cancela la espera. Si no,
        un trabajo que se pasó de tiempo dejaría que el siguiente arranque otro hilo y se acumularían hilos
        colgados. Los casos de uso asumen este contrato: no limitan ellos la concurrencia.
        """
        ...


class RepositorioTrabajos(Protocol):
    async def agregar(self, trabajo: Trabajo) -> Trabajo:
        """Guarda el trabajo si no existe otro con el mismo `id_evento`; si existe, devuelve el existente.
        Tiene que ser atómico: es lo que resuelve dos pedidos simultáneos con el mismo `id_evento`."""
        ...

    async def guardar(self, trabajo: Trabajo) -> None: ...

    async def obtener(self, id_trabajo: IdTrabajo) -> Trabajo | None: ...

    async def buscar_por_id_evento(self, id_evento: IdEvento) -> Trabajo | None: ...

    async def eliminar(self, id_trabajo: IdTrabajo) -> None: ...


class ColaTrabajos(Protocol):
    async def hay_lugar(self) -> bool: ...

    async def encolar(self, id_trabajo: IdTrabajo) -> None:
        """Lanza `ColaLlena` si no hay lugar (otro pedido pudo ocupar el último entre `hay_lugar` y acá)."""
        ...

    async def siguiente(self) -> IdTrabajo: ...


class AlmacenTemporalImagenes(Protocol):
    """Guarda las imágenes subidas por multipart entre el `POST` y el worker.

    Las imágenes se descartan al terminar el trabajo (no se persisten). La implementación debe tener un **TTL de
    respaldo** por si el proceso se cae a mitad de camino y nadie las borra.
    """

    async def guardar(self, contenido: bytes) -> str:
        """Devuelve la referencia con la que se recupera el contenido."""
        ...

    async def obtener(self, referencia: str) -> bytes | None:
        """`None` si no está (por ejemplo, venció su TTL)."""
        ...

    async def borrar(self, referencia: str) -> None:
        """Idempotente: borrar algo que no está no es un error."""
        ...


class ObtenedorImagen(Protocol):
    async def obtener(self, url: str) -> bytes:
        """Descarga la imagen de una URL firmada del almacenamiento del backend. Lanza `ImagenNoDisponible`."""
        ...


class Notificador(Protocol):
    async def notificar(self, trabajo: Trabajo) -> None:
        """Agenda el aviso al backend y **vuelve enseguida**: no espera la entrega ni sus reintentos."""
        ...


class EnviadorCallback(Protocol):
    async def enviar(self, url: str, trabajo: Trabajo) -> None:
        """Un intento de entrega. Lanza `CallbackFallido` si no llegó."""
        ...


class Dormidor(Protocol):
    async def dormir(self, segundos: float) -> None: ...


class GeneradorIds(Protocol):
    def nuevo_id_trabajo(self) -> IdTrabajo: ...


class Reloj(Protocol):
    def ahora(self) -> datetime:
        """Con zona horaria."""
        ...
