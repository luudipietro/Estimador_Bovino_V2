"""Cómo se crean las instancias del modelo real. Es lo único que importa `estimador` (y con él torch y ultralytics).

La app recibe un `ModeloConfigurado`: en producción sale de `fabrica_real`; los tests pasan uno falso y nunca cargan
torch.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bv_estimacion.infraestructura.ml.adaptador_modelo import Estimador


@dataclass(frozen=True, slots=True)
class ModeloConfigurado:
    crear: Callable[[], Estimador]  # una instancia nueva por llamada (no se comparten entre hilos)
    lado_max_px: int  # el lado mayor al que `estimador` reduce la foto antes de recortar (para medir R-24)


def fabrica_real(carpeta_modelos: Path) -> ModeloConfigurado:
    """Recién acá se importa `estimador`. Corre en CPU: las ruedas de torch del proyecto son de CPU."""
    from estimador import LADO_MAX, EstimadorPeso

    def crear() -> Estimador:
        return EstimadorPeso(carpeta_modelos, dispositivo="cpu")

    return ModeloConfigurado(crear=crear, lado_max_px=LADO_MAX)
