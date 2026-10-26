"""Adaptador de `estimador.EstimadorPeso` al puerto `ModeloPeso` (sin modificar el módulo).

No importa `estimador` (arrastraría torch y ultralytics): recibe una **fábrica** de instancias y el lado mayor al que
`estimador` reduce la foto. El cableado de la app (Fase 4) pasa `EstimadorPeso` y las constantes reales; los
tests, un falso.
"""

import io
import logging
from collections.abc import Callable
from typing import Any, Protocol

from anyio import to_thread
from PIL import Image

from bv_estimacion.aplicacion.dto import EstimacionFoto
from bv_estimacion.aplicacion.puertos import ModeloNoDisponible
from bv_estimacion.dominio.calidad import es_borrosa
from bv_estimacion.dominio.errores import MotivoNoSe
from bv_estimacion.dominio.estimacion import ResultadoFoto
from bv_estimacion.dominio.valores import VersionModelo
from bv_estimacion.infraestructura.ml.borrosidad import medir_borrosidad
from bv_estimacion.infraestructura.ml.motivos import traducir_motivos
from bv_estimacion.infraestructura.ml.pool import PoolDeInstancias

logger = logging.getLogger(__name__)

ADVERTENCIA_SIN_EXIF = "no trae metadata EXIF"


class Estimador(Protocol):
    """Lo único que se usa de `EstimadorPeso`."""

    def estimar(self, imagen: bytes) -> dict[str, Any]: ...


class AdaptadorModelo:
    """Implementa `ModeloPeso`.

    - **Pool de N instancias** (una por hilo, no son seguras compartidas). Se cargan todas en `cargar()`; hasta
      entonces `cargado` es `False` y `/salud` responde 503.
    - **El rango de validez es el de `config.json`** (`version`), no el que trae escrito el módulo (R-20): se
      recalcula acá y el motivo `fuera_de_rango` del módulo cede ante él.
    - **R-24:** si hay umbral, se mide la nitidez del recorte del animal. Sin umbral no se decodifica nada.
    - **La orientación EXIF no se corrige:** el modelo se entrenó con las imágenes tal como están guardadas
      (`Image.open().convert("RGB")`) y `EstimadorPeso` hace lo mismo. Pendiente de medir con fotos del campo.
    """

    def __init__(
        self,
        fabrica: Callable[[], Estimador],
        cantidad: int,
        version: VersionModelo,
        *,
        lado_max_px: int,
        umbral_borrosidad: float | None,
    ) -> None:
        if cantidad < 1:
            raise ValueError("hace falta al menos una instancia del modelo")
        self._fabrica = fabrica
        self._cantidad = cantidad
        self._version = version
        self._lado_max_px = lado_max_px
        self._umbral_borrosidad = umbral_borrosidad
        self._pool: PoolDeInstancias[Estimador] | None = None

    @property
    def version(self) -> VersionModelo:
        return self._version

    @property
    def cargado(self) -> bool:
        return self._pool is not None

    async def cargar(self) -> None:
        """Crea las N instancias (lento: lee ~130 MB de pesos cada una) fuera del event loop."""
        if self._pool is not None:
            return
        instancias = [await to_thread.run_sync(self._fabrica) for _ in range(self._cantidad)]
        self._pool = PoolDeInstancias(instancias)
        logger.info("modelo_cargado", extra={"instancias": self._cantidad, "version": self._version.version})

    async def estimar(self, contenido: bytes) -> EstimacionFoto:
        if self._pool is None:
            raise ModeloNoDisponible("el modelo todavía no terminó de cargar")
        return await self._pool.ejecutar(lambda estimador: self._inferir(estimador, contenido))

    # --- Corre en el hilo de inferencia -----------------------------------------------------------

    def _inferir(self, estimador: Estimador, contenido: bytes) -> EstimacionFoto:
        crudo = estimador.estimar(contenido)
        advertencias = (ADVERTENCIA_SIN_EXIF,) if _sin_exif(contenido) else ()
        return EstimacionFoto(self._traducir(crudo, contenido), advertencias)

    def _traducir(self, crudo: dict[str, Any], contenido: bytes) -> ResultadoFoto:
        peso = crudo["peso_kg"]
        bovinos = crudo.get("bovinos_en_la_escena")
        if peso is None:
            return ResultadoFoto(None, False, (MotivoNoSe.SIN_ANIMAL,), bovinos_en_la_escena=bovinos)

        # El rango del módulo está escrito en su código: manda el de `config.json` (R-20).
        motivos = [m for m in traducir_motivos(crudo.get("motivo")) if m is not MotivoNoSe.FUERA_DE_RANGO]
        if not self._version.en_rango(peso):
            motivos.append(MotivoNoSe.FUERA_DE_RANGO)
        if self._es_borrosa(contenido, crudo.get("recorte")):
            motivos.append(MotivoNoSe.IMAGEN_BORROSA)

        return ResultadoFoto(
            peso_kg=peso,
            confiable=not motivos,
            motivos=tuple(motivos),
            confianza_deteccion=crudo.get("confianza_deteccion"),
            confianza_keypoint_minima=crudo.get("confianza_keypoint_minima"),
            bovinos_en_la_escena=bovinos,
        )

    def _es_borrosa(self, contenido: bytes, recorte: list[float] | None) -> bool:
        if self._umbral_borrosidad is None:
            return False  # R-24 no se aplica hasta calibrar el umbral
        with Image.open(io.BytesIO(contenido)) as imagen:
            varianza = medir_borrosidad(imagen, recorte, self._lado_max_px)
        return es_borrosa(varianza, self._umbral_borrosidad)


def _sin_exif(contenido: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(contenido)) as imagen:
            return not imagen.getexif()
    except (OSError, ValueError):
        return False  # si no se puede leer, no se inventa una advertencia
