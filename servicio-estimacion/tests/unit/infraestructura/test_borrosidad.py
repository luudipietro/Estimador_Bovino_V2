import numpy as np
import pytest
from PIL import Image

from bv_estimacion.infraestructura.ml.borrosidad import medir_borrosidad, varianza_laplaciano

LADO_MAX = 1024


def nitida(ancho: int = 200, alto: int = 200) -> Image.Image:
    """Tablero de ajedrez: bordes duros por todos lados."""
    celdas = (np.indices((alto, ancho)).sum(axis=0) // 4 % 2 * 255).astype(np.uint8)
    return Image.fromarray(celdas).convert("RGB")


def lisa(ancho: int = 200, alto: int = 200) -> Image.Image:
    return Image.new("RGB", (ancho, alto), (120, 90, 60))


@pytest.mark.requisito("RF-15")
def test_una_imagen_lisa_tiene_varianza_cero() -> None:
    assert varianza_laplaciano(np.full((50, 50), 128.0)) == 0.0


@pytest.mark.requisito("RF-15")
def test_la_varianza_del_laplaciano_coincide_con_el_calculo_a_mano() -> None:
    # 3x3: un único píxel interior, así que hay un solo valor de Laplaciano y su varianza es 0.
    gris = np.zeros((3, 3))
    gris[1, 1] = 255.0
    assert varianza_laplaciano(gris) == 0.0

    # 3x4: dos píxeles interiores, (1,1) y (1,2). Con un píxel brillante en (1,1) sus Laplacianos son
    # -4*255 (él mismo) y +255 (su vecino de la derecha).
    gris = np.zeros((3, 4))
    gris[1, 1] = 255.0
    interior = [-4 * 255.0, 255.0]  # (1,1) y (1,2)
    assert varianza_laplaciano(gris) == pytest.approx(float(np.var(interior)))


@pytest.mark.requisito("RF-15")
@pytest.mark.parametrize("forma", [(2, 10), (10, 2), (1, 1)])
def test_una_imagen_demasiado_chica_no_se_puede_medir(forma: tuple[int, int]) -> None:
    assert varianza_laplaciano(np.zeros(forma)) is None


@pytest.mark.requisito("RF-15")
def test_una_imagen_nitida_tiene_mas_varianza_que_la_misma_desenfocada() -> None:
    from PIL import ImageFilter

    original = nitida()
    desenfocada = original.filter(ImageFilter.GaussianBlur(radius=6))
    nitida_v = medir_borrosidad(original, None, LADO_MAX)
    borrosa_v = medir_borrosidad(desenfocada, None, LADO_MAX)
    assert nitida_v is not None and borrosa_v is not None
    assert nitida_v > 10 * borrosa_v


@pytest.mark.requisito("RF-15")
def test_mide_solo_el_recorte_del_animal() -> None:
    """La mitad izquierda es nítida y la derecha lisa: según la caja, el resultado cambia por completo."""
    imagen = lisa(200, 100)
    imagen.paste(nitida(100, 100), (0, 0))
    en_la_nitida = medir_borrosidad(imagen, [0, 0, 100, 100], LADO_MAX)
    en_la_lisa = medir_borrosidad(imagen, [100, 0, 200, 100], LADO_MAX)
    assert en_la_nitida is not None and en_la_nitida > 0
    assert en_la_lisa == 0.0


@pytest.mark.requisito("RF-15")
def test_sin_caja_mide_la_imagen_entera() -> None:
    assert medir_borrosidad(nitida(), None, LADO_MAX) == medir_borrosidad(nitida(), [0, 0, 200, 200], LADO_MAX)


@pytest.mark.requisito("RF-15")
def test_la_caja_esta_en_coordenadas_de_la_imagen_reducida_como_las_devuelve_el_modulo() -> None:
    """`estimador` reduce la foto a `LADO_MAX` antes de calcular la caja: hay que reducir igual antes de cortar."""
    grande = lisa(2000, 1000)
    grande.paste(nitida(1000, 1000), (0, 0))  # mitad izquierda nítida; reducida a 1000x500 -> nítida en x < 500
    en_nitida = medir_borrosidad(grande, [0, 0, 400, 400], 1000)
    en_lisa = medir_borrosidad(grande, [600, 0, 1000, 400], 1000)
    assert en_nitida is not None and en_nitida > 0
    assert en_lisa == 0.0


@pytest.mark.requisito("RF-15")
def test_una_caja_que_se_sale_de_la_foto_se_recorta_a_la_foto() -> None:
    assert medir_borrosidad(nitida(), [-50, -50, 500, 500], LADO_MAX) == medir_borrosidad(nitida(), None, LADO_MAX)


@pytest.mark.requisito("RF-15")
def test_una_caja_degenerada_cae_a_la_imagen_entera() -> None:
    assert medir_borrosidad(nitida(), [50, 50, 50, 50], LADO_MAX) == medir_borrosidad(nitida(), None, LADO_MAX)
