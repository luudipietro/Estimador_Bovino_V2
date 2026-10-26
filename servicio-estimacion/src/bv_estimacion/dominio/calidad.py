"""Reglas de calidad de la foto y del "no sé" (R-24, R-25). Funciones puras sobre números: no hay imágenes.

El adaptador del modelo mide (varianza del Laplaciano, confianzas, caja del animal) y este módulo decide.
Los umbrales llegan por parámetro: los de R-25 se importan de `estimador` en la infraestructura y el de
R-24 sale de la configuración. Así la regla se prueba sin imágenes ni PyTorch.
"""

from dataclasses import dataclass

from bv_estimacion.dominio.errores import MotivoNoSe

# Margen contra el borde de la foto original, relativo al lado mayor. Replica el criterio de `estimador`.
MARGEN_BORDE_RELATIVO = 0.01

Caja = tuple[float, float, float, float]  # x0, y0, x1, y1 en píxeles de la foto original


@dataclass(frozen=True, slots=True)
class UmbralesCalidad:
    confianza_deteccion_min: float
    confianza_keypoint_min: float
    # None = R-24 no se aplica (el umbral todavía no se calibró). No hay un número por defecto.
    umbral_borrosidad: float | None = None


@dataclass(frozen=True, slots=True)
class MedicionesFoto:
    animal_detectado: bool
    confianza_deteccion: float
    confianza_keypoint_minima: float
    caja_animal: Caja | None
    ancho: int
    alto: int
    # None = no se midió (por ejemplo, porque R-24 está apagada)
    varianza_laplaciano: float | None = None


def es_borrosa(varianza_laplaciano: float | None, umbral: float | None) -> bool:
    """R-24. Sin umbral configurado, o sin medición, la regla no se aplica y la foto nunca se marca borrosa."""
    if umbral is None or varianza_laplaciano is None:
        return False
    return varianza_laplaciano < umbral


def animal_cortado(caja: Caja | None, ancho: int, alto: int) -> bool:
    """R-25: el animal toca el borde de la foto original. Un animal descentrado pero completo es válido."""
    if caja is None:
        return False
    margen = MARGEN_BORDE_RELATIVO * max(ancho, alto)
    x0, y0, x1, y1 = caja
    return x0 <= margen or y0 <= margen or x1 >= ancho - margen or y1 >= alto - margen


def evaluar_calidad(mediciones: MedicionesFoto, umbrales: UmbralesCalidad) -> tuple[MotivoNoSe, ...]:
    """Devuelve los motivos de "no sé" de una foto (R-24, R-25). Vacío = la foto es apta."""
    if not mediciones.animal_detectado:
        return (MotivoNoSe.SIN_ANIMAL,)

    motivos: list[MotivoNoSe] = []
    if mediciones.confianza_deteccion < umbrales.confianza_deteccion_min:
        motivos.append(MotivoNoSe.DETECCION_DUDOSA)
    if mediciones.confianza_keypoint_minima < umbrales.confianza_keypoint_min:
        motivos.append(MotivoNoSe.PARTES_TAPADAS)
    if animal_cortado(mediciones.caja_animal, mediciones.ancho, mediciones.alto):
        motivos.append(MotivoNoSe.ANIMAL_CORTADO)
    if es_borrosa(mediciones.varianza_laplaciano, umbrales.umbral_borrosidad):
        motivos.append(MotivoNoSe.IMAGEN_BORROSA)
    return tuple(motivos)
