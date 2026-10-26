from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bv_estimacion.dominio.errores import MotivoNoSe

SHA256_HEX = r"^[0-9a-fA-F]{64}$"


class ReferenciaImagenIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)
    sha256: str = Field(pattern=SHA256_HEX)


class PedidoPorReferencias(BaseModel):
    """El cuerpo JSON de `POST /v1/estimaciones` cuando las imágenes no viajan, sino que se bajan de una URL firmada.
    Un campo que no está acá es un error (`solicitud_invalida`): atrapa un typo en vez de ignorarlo."""

    model_config = ConfigDict(extra="forbid")

    referencias_imagen: list[ReferenciaImagenIn] = Field(default_factory=list)
    metadatos: dict[str, Any] | None = None
    url_notificacion: str | None = Field(default=None, max_length=2048)


class TrabajoAceptado(BaseModel):
    id: str
    id_evento: str
    estado: Literal["pendiente", "procesando", "completado", "fallido"]


class ResultadoValidacionRespuesta(BaseModel):
    apta: bool
    motivos: list[MotivoNoSe]
    advertencias: list[str]
