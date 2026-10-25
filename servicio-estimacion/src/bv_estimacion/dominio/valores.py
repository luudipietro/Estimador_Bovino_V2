"""Value objects del dominio. Solo stdlib."""

from dataclasses import dataclass


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
