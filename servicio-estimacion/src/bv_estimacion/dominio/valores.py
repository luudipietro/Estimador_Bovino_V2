"""Value objects del dominio. Solo stdlib."""

import re
from dataclasses import dataclass
from enum import StrEnum

_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True, slots=True)
class IdEvento:
    """ID del evento de pesaje que genera el backend C#. Es también la `Idempotency-Key`."""

    valor: str

    def __post_init__(self) -> None:
        if not self.valor.strip():
            raise ValueError("el id_evento no puede estar vacío")

    def __str__(self) -> str:
        return self.valor


@dataclass(frozen=True, slots=True)
class IdTrabajo:
    """ID del trabajo de estimación (por ejemplo `trb_01J…`). Lo genera la infraestructura."""

    valor: str

    def __post_init__(self) -> None:
        if not self.valor.strip():
            raise ValueError("el id del trabajo no puede estar vacío")

    def __str__(self) -> str:
        return self.valor


class OrigenImagen(StrEnum):
    """De dónde se lee la imagen cuando el worker la necesita."""

    ALMACEN_TEMPORAL = "almacen_temporal"  # subida por multipart y guardada hasta que termine el trabajo
    URL = "url"  # URL firmada del almacenamiento privado del backend


@dataclass(frozen=True, slots=True)
class ReferenciaImagen:
    """Una imagen de un pedido: dónde está y su SHA-256.

    El SHA-256 es la identidad del contenido: con él se calcula la huella de idempotencia y se verifica,
    al leer la imagen, que sea la que el backend declaró. Se valida acá (64 caracteres hexadecimales) para
    que una referencia mal formada no llegue a los casos de uso. La allowlist de hosts de una URL **no** se
    valida en el dominio: es configuración del servicio y se controla en el borde.
    """

    origen: OrigenImagen
    ubicacion: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.ubicacion.strip():
            raise ValueError("la ubicación de la imagen no puede estar vacía")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("el sha256 debe tener 64 caracteres hexadecimales")
        object.__setattr__(self, "sha256", self.sha256.lower())  # se compara siempre en minúsculas


@dataclass(frozen=True, slots=True)
class VersionModelo:
    """Versión del modelo y rango de pesos en el que fue entrenado (R-20, R-23).

    Los tres datos salen de `config.json`; el rango nunca se escribe a mano en el código.
    """

    version: str
    rango_min_kg: float
    rango_max_kg: float

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("la versión del modelo no puede estar vacía")
        if self.rango_min_kg >= self.rango_max_kg:
            raise ValueError("el rango de validez es inválido: el mínimo debe ser menor que el máximo")

    @property
    def rango_validez_kg(self) -> tuple[float, float]:
        return (self.rango_min_kg, self.rango_max_kg)

    def en_rango(self, peso_kg: float) -> bool:
        """R-20: los extremos del rango de entrenamiento cuentan como válidos."""
        return self.rango_min_kg <= peso_kg <= self.rango_max_kg
