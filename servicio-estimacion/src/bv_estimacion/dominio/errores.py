"""Catálogo cerrado de códigos (RF-22). La fuente de verdad es `docs/errores.md`.

Los valores son `snake_case` estables: el backend C# los mapea a sus mensajes. No se renombran ni se
reutilizan sin versionar la API (`/v2`).
"""

from enum import StrEnum


class MotivoNoSe(StrEnum):
    """Por qué el modelo no puede estimar con confianza. **No es un error** (R-21): el trabajo termina
    `completado` con `confiable = false`."""

    SIN_ANIMAL = "sin_animal"
    DETECCION_DUDOSA = "deteccion_dudosa"
    PARTES_TAPADAS = "partes_tapadas"
    ANIMAL_CORTADO = "animal_cortado"
    IMAGEN_BORROSA = "imagen_borrosa"
    FUERA_DE_RANGO = "fuera_de_rango"


class CodigoError(StrEnum):
    # --- Entrada inválida: se rechaza al recibir (4xx) ---
    FORMATO_NO_ADMITIDO = "formato_no_admitido"
    IMAGEN_CORRUPTA = "imagen_corrupta"
    IMAGEN_MUY_GRANDE = "imagen_muy_grande"
    RESOLUCION_INSUFICIENTE = "resolucion_insuficiente"
    CANTIDAD_IMAGENES_INVALIDA = "cantidad_imagenes_invalida"
    ID_EVENTO_EN_CONFLICTO = "id_evento_en_conflicto"
    REFERENCIA_NO_PERMITIDA = "referencia_no_permitida"
    NO_AUTENTICADO = "no_autenticado"
    TRABAJO_NO_ENCONTRADO = "trabajo_no_encontrado"  # propuesto
    COLA_LLENA = "cola_llena"  # propuesto

    # --- Falla interna: el trabajo queda `fallido` ---
    MODELO_NO_DISPONIBLE = "modelo_no_disponible"
    TIEMPO_EXCEDIDO = "tiempo_excedido"
    REFERENCIA_INACCESIBLE = "referencia_inaccesible"
    ERROR_INTERNO = "error_interno"


# Códigos que el equipo todavía no confirmó (ver docs/errores.md). Si se rechazan, se borran de acá y del enum.
CODIGOS_PROPUESTOS: frozenset[CodigoError] = frozenset(
    {CodigoError.TRABAJO_NO_ENCONTRADO, CodigoError.COLA_LLENA},
)

# Los únicos códigos con los que un `Trabajo` puede quedar `fallido`.
FALLAS_INTERNAS: frozenset[CodigoError] = frozenset(
    {
        CodigoError.MODELO_NO_DISPONIBLE,
        CodigoError.TIEMPO_EXCEDIDO,
        CodigoError.REFERENCIA_INACCESIBLE,
        CodigoError.ERROR_INTERNO,
    },
)


class ErrorDeDominio(Exception):
    """Error de negocio con un código del catálogo y, si aplica, la regla `R-xx` involucrada."""

    def __init__(self, codigo: CodigoError, detalle: str, regla: str | None = None) -> None:
        super().__init__(detalle)
        self.codigo = codigo
        self.detalle = detalle
        self.regla = regla


class TransicionInvalida(ErrorDeDominio):
    """Un `Trabajo` intentó una transición de estado que no existe. Es un defecto de programación."""

    def __init__(self, detalle: str) -> None:
        super().__init__(CodigoError.ERROR_INTERNO, detalle)
