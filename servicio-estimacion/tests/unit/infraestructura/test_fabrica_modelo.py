import sys
import types
from pathlib import Path

import pytest

from bv_estimacion.infraestructura.ml.fabrica import fabrica_real


class EstimadorPesoFalso:
    def __init__(self, carpeta: Path, dispositivo: str = "auto") -> None:
        self.carpeta = carpeta
        self.dispositivo = dispositivo


@pytest.fixture
def estimador_falso(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Un módulo `estimador` de mentira: así el test no importa torch ni ultralytics."""
    modulo = types.ModuleType("estimador")
    modulo.EstimadorPeso = EstimadorPesoFalso  # type: ignore[attr-defined]
    modulo.LADO_MAX = 1024  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "estimador", modulo)
    return modulo


@pytest.mark.requisito("RNF-03")
def test_la_fabrica_real_usa_el_lado_maximo_del_modulo_y_crea_instancias_sobre_la_carpeta(
    estimador_falso: types.ModuleType, tmp_path: Path
) -> None:
    estimador_falso.LADO_MAX = 800  # type: ignore[attr-defined]

    modelo = fabrica_real(tmp_path)

    assert modelo.lado_max_px == 800
    primera, segunda = modelo.crear(), modelo.crear()
    assert isinstance(primera, EstimadorPesoFalso)
    assert primera is not segunda  # una instancia por hilo
    assert primera.carpeta == tmp_path


@pytest.mark.requisito("RNF-03")
def test_la_fabrica_real_corre_en_cpu(estimador_falso: types.ModuleType, tmp_path: Path) -> None:
    """Las ruedas de torch del proyecto son de CPU: el dispositivo no se deja en "auto"."""
    instancia = fabrica_real(tmp_path).crear()
    assert isinstance(instancia, EstimadorPesoFalso)
    assert instancia.dispositivo == "cpu"


@pytest.mark.requisito("RNF-03")
def test_el_modulo_estimador_no_se_importa_hasta_que_se_pide_la_fabrica(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importar `bv_estimacion.infraestructura.ml.fabrica` no arrastra torch: solo `fabrica_real` lo hace."""
    monkeypatch.delitem(sys.modules, "estimador", raising=False)
    monkeypatch.setitem(sys.modules, "estimador", None)  # un `import estimador` fallaría con ImportError
    with pytest.raises(ImportError):
        fabrica_real(Path("."))
