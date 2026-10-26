"""RF-19: consultar el estado de un trabajo."""

from bv_estimacion.aplicacion.puertos import RepositorioTrabajos
from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.dominio.estimacion import Trabajo
from bv_estimacion.dominio.valores import IdTrabajo


class ConsultarEstimacion:
    def __init__(self, trabajos: RepositorioTrabajos) -> None:
        self._trabajos = trabajos

    async def ejecutar(self, id_trabajo: IdTrabajo) -> Trabajo:
        trabajo = await self._trabajos.obtener(id_trabajo)
        if trabajo is None:
            # No se distingue entre "nunca existió" y "ya venció": el backend reenvía con el mismo id_evento.
            raise ErrorDeDominio(CodigoError.TRABAJO_NO_ENCONTRADO, "el trabajo no existe o ya venció")
        return trabajo
