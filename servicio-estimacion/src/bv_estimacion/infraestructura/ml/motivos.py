"""Traducción de los motivos en texto libre de `EstimadorPeso` a los códigos estables de `docs/errores.md`.

El módulo devuelve `motivo` como un string con los motivos unidos por `"; "`. La tabla es **exacta**: un texto que
no esté en ella no se descarta en silencio. Un test de contrato lee `estimador.py` y falla si el módulo agrega,
cambia o quita un motivo.
"""

import logging
from collections.abc import Mapping
from types import MappingProxyType

from bv_estimacion.dominio.errores import MotivoNoSe

logger = logging.getLogger(__name__)

SEPARADOR = "; "

TABLA_MOTIVOS: Mapping[str, MotivoNoSe] = MappingProxyType(
    {
        "no se detectó ningún bovino en la foto": MotivoNoSe.SIN_ANIMAL,
        "el bovino no se distingue con claridad": MotivoNoSe.DETECCION_DUDOSA,
        "hay partes del cuerpo tapadas o fuera de cuadro": MotivoNoSe.PARTES_TAPADAS,
        "el animal aparece cortado por el borde de la foto": MotivoNoSe.ANIMAL_CORTADO,
        "el peso estimado cae fuera del rango con el que se entrenó el modelo": MotivoNoSe.FUERA_DE_RANGO,
    }
)


class MotivoDesconocido(Exception):
    """`estimador` informó un motivo que la tabla no conoce: el módulo cambió y hay que actualizarla."""


def traducir_motivos(texto: str | None) -> tuple[MotivoNoSe, ...]:
    if not texto:
        return ()
    motivos: list[MotivoNoSe] = []
    for parte in texto.split(SEPARADOR):
        motivo = TABLA_MOTIVOS.get(parte)
        if motivo is None:
            logger.error("motivo_desconocido_del_modelo", extra={"motivo": parte})
            raise MotivoDesconocido(f"el modelo informó un motivo que no está en la tabla: {parte!r}")
        motivos.append(motivo)
    return tuple(motivos)
