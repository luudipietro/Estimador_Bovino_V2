"""Resultado de la estimación y ciclo de vida del trabajo (RF-19, RF-20, RF-22; R-20…R-23). Solo stdlib."""

import hashlib
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum

from bv_estimacion.dominio.errores import (
    FALLAS_INTERNAS,
    CodigoError,
    ErrorDeDominio,
    MotivoNoSe,
    TransicionInvalida,
)
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, ReferenciaImagen, VersionModelo

MIN_IMAGENES = 1
MAX_IMAGENES = 4


def _validar_coherencia(peso_kg: float | None, confiable: bool, motivos: tuple[MotivoNoSe, ...]) -> None:
    """Un resultado confiable tiene peso y ningún motivo; uno no confiable (el "no sé") siempre dice por qué."""
    if confiable and (peso_kg is None or motivos):
        raise ValueError("un resultado confiable necesita peso y no puede tener motivos")
    if not confiable and not motivos:
        raise ValueError("un resultado no confiable necesita al menos un motivo")


@dataclass(frozen=True, slots=True)
class ResultadoFoto:
    peso_kg: float | None
    confiable: bool
    motivos: tuple[MotivoNoSe, ...] = ()
    confianza_deteccion: float | None = None
    confianza_keypoint_minima: float | None = None
    bovinos_en_la_escena: int | None = None

    def __post_init__(self) -> None:
        _validar_coherencia(self.peso_kg, self.confiable, self.motivos)


@dataclass(frozen=True, slots=True)
class ResultadoEstimacion:
    peso_kg: float | None
    confiable: bool
    motivos: tuple[MotivoNoSe, ...]
    version: VersionModelo  # R-23
    advertencias: tuple[str, ...] = ()
    # El detalle de cada foto, en el orden del pedido: el contrato lo devuelve en `imagenes[]`.
    fotos: tuple[ResultadoFoto, ...] = ()

    def __post_init__(self) -> None:
        _validar_coherencia(self.peso_kg, self.confiable, self.motivos)


def _redondear(peso_kg: float) -> float:
    """El contrato expresa los pesos en kg con un decimal."""
    return round(peso_kg, 1)


def _mediana_o_none(fotos: Sequence[ResultadoFoto]) -> float | None:
    pesos = [f.peso_kg for f in fotos if f.peso_kg is not None]
    return _redondear(statistics.median(pesos)) if pesos else None


def agregar_resultados(
    fotos: Sequence[ResultadoFoto],
    version: VersionModelo,
    advertencias: tuple[str, ...] = (),
) -> ResultadoEstimacion:
    """R-22: mediana de las fotos confiables. R-20: un peso fuera del rango de `version` no es confiable,
    sin importar lo que haya informado el modelo. R-23: el resultado lleva la versión."""
    if not MIN_IMAGENES <= len(fotos) <= MAX_IMAGENES:
        raise ErrorDeDominio(
            CodigoError.CANTIDAD_IMAGENES_INVALIDA,
            f"se esperaban entre {MIN_IMAGENES} y {MAX_IMAGENES} fotos y llegaron {len(fotos)}",
            regla="R-22",
        )

    detalle = tuple(fotos)
    confiables = [f.peso_kg for f in fotos if f.confiable and f.peso_kg is not None]
    if not confiables:
        motivos = tuple(sorted({m for f in fotos for m in f.motivos}))
        return ResultadoEstimacion(_mediana_o_none(fotos), False, motivos, version, advertencias, detalle)

    peso = statistics.median(confiables)
    if not version.en_rango(peso):
        return ResultadoEstimacion(
            _redondear(peso), False, (MotivoNoSe.FUERA_DE_RANGO,), version, advertencias, detalle
        )
    return ResultadoEstimacion(_redondear(peso), True, (), version, advertencias, detalle)


def huella_de_imagenes(hashes_sha256: Sequence[str]) -> str:
    """Huella del pedido para la idempotencia: mismo `id_evento` + misma huella = mismo trabajo.

    Se calcula sobre los hashes **ordenados**: el mismo conjunto de fotos en otro orden da la misma huella, así
    un reintento (por ejemplo, al sincronizar tras un corte, RF-17) recibe el trabajo original y no un 409.
    Los índices de las imágenes ("imagen 1", "imagen 2") siguen el orden del **primer** pedido, que es el que
    quedó guardado en el `Trabajo`.
    """
    if not hashes_sha256:
        raise ValueError("hacen falta los hashes de las imágenes")
    ordenados = sorted(h.lower() for h in hashes_sha256)
    return hashlib.sha256("\n".join(ordenados).encode("utf-8")).hexdigest()


class EstadoTrabajo(StrEnum):
    PENDIENTE = "pendiente"
    PROCESANDO = "procesando"
    COMPLETADO = "completado"
    FALLIDO = "fallido"


def _ms(desde: datetime, hasta: datetime) -> int:
    return round((hasta - desde).total_seconds() * 1000)


def _exigir_zona_horaria(momento: datetime) -> None:
    if momento.tzinfo is None:
        raise ValueError("las fechas del trabajo llevan zona horaria (ISO 8601)")


@dataclass(frozen=True, slots=True)
class Trabajo:
    """Agregado raíz. Inmutable: cada transición devuelve un trabajo nuevo.

    `PENDIENTE → PROCESANDO → COMPLETADO | FALLIDO`, sin vuelta atrás.
    """

    id: IdTrabajo
    id_evento: IdEvento
    huella_imagenes: str
    # En el orden del primer pedido: de ahí salen los índices de la respuesta. El servicio no es dueño de las imágenes.
    imagenes: tuple[ReferenciaImagen, ...]
    creado: datetime
    estado: EstadoTrabajo = EstadoTrabajo.PENDIENTE
    metadatos: Mapping[str, object] = field(default_factory=dict)  # R-26: no alteran la estimación
    # El dominio solo guarda el dato; que el host esté en la allowlist se valida en el borde.
    url_notificacion: str | None = None
    iniciado: datetime | None = None
    finalizado: datetime | None = None
    resultado: ResultadoEstimacion | None = None
    codigo_error: CodigoError | None = None
    reintentable: bool | None = None
    descarga_ms: int = 0
    inferencia_ms: int = 0

    @classmethod
    def crear(
        cls,
        id: IdTrabajo,
        id_evento: IdEvento,
        imagenes: Sequence[ReferenciaImagen],
        creado: datetime,
        metadatos: Mapping[str, object] | None = None,
        url_notificacion: str | None = None,
    ) -> "Trabajo":
        if not MIN_IMAGENES <= len(imagenes) <= MAX_IMAGENES:
            raise ErrorDeDominio(
                CodigoError.CANTIDAD_IMAGENES_INVALIDA,
                f"se esperaban entre {MIN_IMAGENES} y {MAX_IMAGENES} imágenes y llegaron {len(imagenes)}",
                regla="R-18",
            )
        _exigir_zona_horaria(creado)
        return cls(
            id=id,
            id_evento=id_evento,
            huella_imagenes=huella_de_imagenes([i.sha256 for i in imagenes]),
            imagenes=tuple(imagenes),
            creado=creado,
            metadatos=dict(metadatos or {}),
            url_notificacion=url_notificacion,
        )

    def es_mismo_pedido(self, huella_imagenes: str) -> bool:
        """Idempotencia: mismo `id_evento` con la misma huella devuelve este trabajo; con otra, es un conflicto."""
        return self.huella_imagenes == huella_imagenes

    def _exigir(self, esperado: EstadoTrabajo, accion: str) -> None:
        if self.estado is not esperado:
            raise TransicionInvalida(f"no se puede {accion} un trabajo {self.estado.value}")

    def iniciar(self, ahora: datetime) -> "Trabajo":
        self._exigir(EstadoTrabajo.PENDIENTE, "iniciar")
        _exigir_zona_horaria(ahora)
        return replace(self, estado=EstadoTrabajo.PROCESANDO, iniciado=ahora)

    def completar(
        self,
        resultado: ResultadoEstimacion,
        ahora: datetime,
        *,
        descarga_ms: int = 0,
        inferencia_ms: int = 0,
    ) -> "Trabajo":
        """R-21: el "no sé" también es `completado`."""
        self._exigir(EstadoTrabajo.PROCESANDO, "completar")
        _exigir_zona_horaria(ahora)
        return replace(
            self,
            estado=EstadoTrabajo.COMPLETADO,
            finalizado=ahora,
            resultado=resultado,
            descarga_ms=descarga_ms,
            inferencia_ms=inferencia_ms,
        )

    def fallar(self, codigo: CodigoError, ahora: datetime, *, reintentable: bool) -> "Trabajo":
        self._exigir(EstadoTrabajo.PROCESANDO, "fallar")
        _exigir_zona_horaria(ahora)
        if codigo not in FALLAS_INTERNAS:
            raise ValueError(
                f"{codigo.value} no es una falla interna: se rechaza al recibir, no deja un trabajo fallido"
            )
        return replace(
            self,
            estado=EstadoTrabajo.FALLIDO,
            finalizado=ahora,
            codigo_error=codigo,
            reintentable=reintentable,
        )

    @property
    def cola_ms(self) -> int | None:
        """Tiempo que esperó en la cola antes de empezar."""
        return None if self.iniciado is None else _ms(self.creado, self.iniciado)

    @property
    def total_ms(self) -> int | None:
        return None if self.finalizado is None else _ms(self.creado, self.finalizado)
