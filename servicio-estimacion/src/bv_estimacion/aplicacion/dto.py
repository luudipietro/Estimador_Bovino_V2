"""Datos que entran y salen de los casos de uso."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from bv_estimacion.dominio.errores import MotivoNoSe
from bv_estimacion.dominio.estimacion import ResultadoFoto, Trabajo
from bv_estimacion.dominio.valores import IdEvento


@dataclass(frozen=True, slots=True)
class EstimacionFoto:
    """Lo que devuelve el modelo para una foto: el resultado y las advertencias de la imagen (por ejemplo, que no
    traiga EXIF). Las advertencias no llevan el número de imagen: lo agrega el caso de uso."""

    resultado: ResultadoFoto
    advertencias: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReferenciaExterna:
    """Una imagen que el backend deja en su almacenamiento privado: URL firmada y SHA-256 del contenido."""

    url: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SolicitarEstimacionCmd:
    """Imágenes por multipart (`imagenes`) **o** por URL firmada (`referencias`), nunca las dos."""

    id_evento: IdEvento
    imagenes: tuple[bytes, ...] = ()
    referencias: tuple[ReferenciaExterna, ...] = ()
    metadatos: Mapping[str, object] = field(default_factory=dict)
    url_notificacion: str | None = None


@dataclass(frozen=True, slots=True)
class SolicitudAceptada:
    trabajo: Trabajo
    nuevo: bool  # False = el `id_evento` ya existía con el mismo pedido (idempotencia)


@dataclass(frozen=True, slots=True)
class ResultadoValidacion:
    apta: bool
    motivos: tuple[MotivoNoSe, ...]
    advertencias: tuple[str, ...]
