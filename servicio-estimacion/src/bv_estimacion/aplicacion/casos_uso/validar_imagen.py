"""RF-15: validación rápida de la calidad de una imagen, sin estimar ni crear un trabajo."""

from bv_estimacion.aplicacion.dto import ResultadoValidacion
from bv_estimacion.aplicacion.puertos import ModeloPeso
from bv_estimacion.dominio.errores import MotivoNoSe


class ValidarImagen:
    """Dice si la foto sirve para estimar el peso.

    `fuera_de_rango` **no** es un problema de la imagen: depende del peso estimado (R-20) y pertenece a la
    estimación. Por eso no se informa acá. La entrada inválida (formato, corrupción, resolución) se rechaza antes,
    en el borde, y no llega a este caso de uso.
    """

    def __init__(self, modelo: ModeloPeso) -> None:
        self._modelo = modelo

    async def ejecutar(self, contenido: bytes) -> ResultadoValidacion:
        estimacion = await self._modelo.estimar(contenido)
        motivos = tuple(m for m in estimacion.resultado.motivos if m is not MotivoNoSe.FUERA_DE_RANGO)
        return ResultadoValidacion(apta=not motivos, motivos=motivos, advertencias=estimacion.advertencias)
