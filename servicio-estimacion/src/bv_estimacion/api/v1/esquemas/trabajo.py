"""El cuerpo de `GET /v1/estimaciones/{id}` y del callback: **una sola función** para los dos, así que no pueden
decir cosas distintas. Ver los ejemplos en `docs/contrato-backend.md`."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from bv_estimacion.dominio.errores import CodigoError, MotivoNoSe
from bv_estimacion.dominio.estimacion import EstadoTrabajo, Trabajo

# Detalle legible y genérico de cada falla interna: nunca trazas, rutas ni nombres de clases (RF-22).
DETALLE_DE_FALLA: dict[CodigoError, str] = {
    CodigoError.MODELO_NO_DISPONIBLE: "El modelo de estimación no está disponible.",
    CodigoError.TIEMPO_EXCEDIDO: "La estimación superó el presupuesto de tiempo.",
    CodigoError.REFERENCIA_INACCESIBLE: "No se pudo obtener la imagen del pedido.",
    CodigoError.ERROR_INTERNO: "La estimación falló por un error interno.",
}


class TrabajoEnCurso(BaseModel):
    id: str
    id_evento: str
    estado: Literal["pendiente", "procesando"]
    creado: datetime


class ImagenResultado(BaseModel):
    indice: int = Field(description="Número de imagen, desde 1, en el orden del primer pedido")
    peso_kg: float | None
    confiable: bool
    motivos: list[MotivoNoSe]
    confianza_deteccion: float | None
    confianza_keypoint_minima: float | None
    bovinos_en_la_escena: int | None


class DuracionMs(BaseModel):
    cola: int | None
    inferencia: int
    total: int | None


class TrabajoCompletado(BaseModel):
    id: str
    id_evento: str
    estado: Literal["completado"]
    peso_kg: float | None = Field(description="`null` cuando el modelo no puede estimar (R-21)")
    confiable: bool
    motivos: list[MotivoNoSe]
    version_modelo: str
    rango_validez_kg: tuple[float, float]
    imagenes: list[ImagenResultado]
    advertencias: list[str]
    metadatos: dict[str, Any] = Field(description="Se devuelven tal cual llegaron (R-26)")
    duracion_ms: DuracionMs
    creado: datetime
    finalizado: datetime


class TrabajoFallido(BaseModel):
    id: str
    id_evento: str
    estado: Literal["fallido"]
    codigo: CodigoError
    detalle: str
    reintentable: bool
    creado: datetime
    finalizado: datetime


RespuestaTrabajo = Annotated[TrabajoEnCurso | TrabajoCompletado | TrabajoFallido, Field(discriminator="estado")]
_ADAPTADOR: TypeAdapter[TrabajoEnCurso | TrabajoCompletado | TrabajoFallido] = TypeAdapter(RespuestaTrabajo)


def a_respuesta(trabajo: Trabajo) -> TrabajoEnCurso | TrabajoCompletado | TrabajoFallido:
    if trabajo.estado in (EstadoTrabajo.PENDIENTE, EstadoTrabajo.PROCESANDO):
        return TrabajoEnCurso(
            id=str(trabajo.id),
            id_evento=str(trabajo.id_evento),
            estado=trabajo.estado.value,  # type: ignore[arg-type]  # "pendiente" o "procesando", los dos del Literal
            creado=trabajo.creado,
        )

    assert trabajo.finalizado is not None  # un trabajo terminado siempre tiene fecha de fin

    if trabajo.estado is EstadoTrabajo.FALLIDO:
        assert trabajo.codigo_error is not None and trabajo.reintentable is not None
        return TrabajoFallido(
            id=str(trabajo.id),
            id_evento=str(trabajo.id_evento),
            estado="fallido",
            codigo=trabajo.codigo_error,
            detalle=DETALLE_DE_FALLA[trabajo.codigo_error],
            reintentable=trabajo.reintentable,
            creado=trabajo.creado,
            finalizado=trabajo.finalizado,
        )

    resultado = trabajo.resultado
    assert resultado is not None  # un trabajo completado siempre tiene resultado
    return TrabajoCompletado(
        id=str(trabajo.id),
        id_evento=str(trabajo.id_evento),
        estado="completado",
        peso_kg=resultado.peso_kg,
        confiable=resultado.confiable,
        motivos=list(resultado.motivos),
        version_modelo=resultado.version.version,
        rango_validez_kg=resultado.version.rango_validez_kg,
        imagenes=[
            ImagenResultado(
                indice=indice,
                peso_kg=foto.peso_kg,
                confiable=foto.confiable,
                motivos=list(foto.motivos),
                confianza_deteccion=foto.confianza_deteccion,
                confianza_keypoint_minima=foto.confianza_keypoint_minima,
                bovinos_en_la_escena=foto.bovinos_en_la_escena,
            )
            for indice, foto in enumerate(resultado.fotos, start=1)
        ],
        advertencias=list(resultado.advertencias),
        metadatos=dict(trabajo.metadatos),
        duracion_ms=DuracionMs(cola=trabajo.cola_ms, inferencia=trabajo.inferencia_ms, total=trabajo.total_ms),
        creado=trabajo.creado,
        finalizado=trabajo.finalizado,
    )


def serializar_trabajo(trabajo: Trabajo) -> bytes:
    """Los bytes JSON (UTF-8) que ven el `GET` y el callback. Es lo que se firma, así que sale idéntico cada vez."""
    return _ADAPTADOR.dump_json(a_respuesta(trabajo))
