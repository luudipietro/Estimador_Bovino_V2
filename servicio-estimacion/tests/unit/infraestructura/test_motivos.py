import ast
import logging
from pathlib import Path

import pytest

from bv_estimacion.dominio.errores import MotivoNoSe
from bv_estimacion.infraestructura.ml.motivos import TABLA_MOTIVOS, MotivoDesconocido, traducir_motivos

ESTIMADOR_PY = Path(__file__).resolve().parents[4] / "estimador" / "estimador.py"


@pytest.mark.requisito("RF-22")
def test_sin_motivo_no_hay_motivos() -> None:
    assert traducir_motivos(None) == ()


@pytest.mark.requisito("RF-22")
@pytest.mark.parametrize(("texto", "motivo"), list(TABLA_MOTIVOS.items()))
def test_cada_texto_del_modulo_se_traduce_a_su_codigo(texto: str, motivo: MotivoNoSe) -> None:
    assert traducir_motivos(texto) == (motivo,)


@pytest.mark.requisito("RF-22")
def test_varios_motivos_unidos_por_punto_y_coma_se_separan_en_orden() -> None:
    texto = "el bovino no se distingue con claridad; el animal aparece cortado por el borde de la foto"
    assert traducir_motivos(texto) == (MotivoNoSe.DETECCION_DUDOSA, MotivoNoSe.ANIMAL_CORTADO)


@pytest.mark.requisito("RF-22")
def test_un_texto_desconocido_no_se_descarta_en_silencio(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR), pytest.raises(MotivoDesconocido):
        traducir_motivos("el bovino no se distingue con claridad; motivo nuevo que nadie tradujo")
    registros = [r for r in caplog.records if r.getMessage() == "motivo_desconocido_del_modelo"]
    assert [getattr(r, "motivo", None) for r in registros] == ["motivo nuevo que nadie tradujo"]


def _textos_que_el_modulo_puede_devolver() -> set[str]:
    """Lee `estimador.py` como texto (sin importarlo: arrastraría torch) y junta los motivos que puede emitir."""
    arbol = ast.parse(ESTIMADOR_PY.read_text(encoding="utf-8"))
    textos: set[str] = set()
    for nodo in ast.walk(arbol):
        # motivos.append("...")
        if (
            isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Attribute)
            and nodo.func.attr == "append"
            and isinstance(nodo.func.value, ast.Name)
            and nodo.func.value.id == "motivos"
            and nodo.args
            and isinstance(nodo.args[0], ast.Constant)
        ):
            textos.add(str(nodo.args[0].value))
        # {"peso_kg": None, ..., "motivo": "..."}
        if isinstance(nodo, ast.Dict):
            for clave, valor in zip(nodo.keys, nodo.values, strict=True):
                if (
                    isinstance(clave, ast.Constant)
                    and clave.value == "motivo"
                    and isinstance(valor, ast.Constant)
                    and isinstance(valor.value, str)
                ):
                    textos.add(valor.value)
    return textos


@pytest.mark.requisito("RF-22")
def test_de_contrato_la_tabla_cubre_exactamente_lo_que_el_modulo_puede_decir() -> None:
    """Si `estimador.py` agrega, cambia o quita un motivo, este test falla: hay que actualizar la tabla (y
    `docs/errores.md`) a propósito, no descubrirlo en producción."""
    emitidos = _textos_que_el_modulo_puede_devolver()
    assert emitidos, "no se encontró ningún motivo en estimador.py: ¿cambió la forma del código?"
    assert emitidos == set(TABLA_MOTIVOS)
