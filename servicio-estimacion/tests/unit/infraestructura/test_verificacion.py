import hashlib
import json
from pathlib import Path

import pytest

from bv_estimacion.infraestructura.ml.verificacion import (
    VerificacionModeloFallida,
    leer_sumas,
    sha256_de_archivo,
    verificar_hashes_modelos,
)
from tests.conftest import CARPETA_MODELOS_REPO


def sha(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


def preparar(carpeta: Path, archivos: dict[str, bytes], config: dict[str, object] | None = None) -> Path:
    for nombre, contenido in archivos.items():
        (carpeta / nombre).write_bytes(contenido)
    (carpeta / "SHA256SUMS.txt").write_text("".join(f"{sha(c)} *{n}\n" for n, c in archivos.items()), encoding="utf-8")
    (carpeta / "config.json").write_text(
        json.dumps(config if config is not None else {"archivo_pose": "pose.pt", "archivo_cnn": "cnn.pt"}),
        encoding="utf-8",
    )
    return carpeta


ARCHIVOS = {"pose.pt": b"pesos de pose", "cnn.pt": b"pesos de la cnn"}


@pytest.mark.requisito("RF-22")
def test_con_los_hashes_correctos_no_lanza(tmp_path: Path) -> None:
    verificar_hashes_modelos(preparar(tmp_path, ARCHIVOS))


@pytest.mark.requisito("RF-22")
def test_un_archivo_corrupto_impide_arrancar(tmp_path: Path) -> None:
    carpeta = preparar(tmp_path, ARCHIVOS)
    (carpeta / "cnn.pt").write_bytes(b"pesos alterados")
    with pytest.raises(VerificacionModeloFallida, match=r"cnn\.pt"):
        verificar_hashes_modelos(carpeta)


@pytest.mark.requisito("RF-22")
def test_un_archivo_faltante_impide_arrancar(tmp_path: Path) -> None:
    carpeta = preparar(tmp_path, ARCHIVOS)
    (carpeta / "pose.pt").unlink()
    with pytest.raises(VerificacionModeloFallida, match=r"pose\.pt"):
        verificar_hashes_modelos(carpeta)


@pytest.mark.requisito("RF-22")
def test_un_pt_de_config_json_que_no_esta_en_la_lista_impide_arrancar(tmp_path: Path) -> None:
    """Un `.pt` fuera de SHA256SUMS podría reemplazarse sin que nadie lo note."""
    carpeta = preparar(
        tmp_path, ARCHIVOS, {"archivo_pose": "pose.pt", "archivo_cnn": "cnn.pt", "archivo_detector": "x.pt"}
    )
    (carpeta / "x.pt").write_bytes(b"sin hash declarado")
    with pytest.raises(VerificacionModeloFallida, match=r"x\.pt"):
        verificar_hashes_modelos(carpeta)


@pytest.mark.requisito("RF-22")
def test_sin_sha256sums_o_sin_config_json_impide_arrancar(tmp_path: Path) -> None:
    with pytest.raises(VerificacionModeloFallida):
        verificar_hashes_modelos(tmp_path)  # no hay nada
    carpeta = preparar(tmp_path, ARCHIVOS)
    (carpeta / "config.json").unlink()
    with pytest.raises(VerificacionModeloFallida):
        verificar_hashes_modelos(carpeta)


@pytest.mark.requisito("RF-22")
def test_config_json_roto_impide_arrancar(tmp_path: Path) -> None:
    carpeta = preparar(tmp_path, ARCHIVOS)
    (carpeta / "config.json").write_text("{ roto", encoding="utf-8")
    with pytest.raises(VerificacionModeloFallida):
        verificar_hashes_modelos(carpeta)


@pytest.mark.requisito("RF-22")
def test_sha256sums_con_finales_de_linea_de_windows_se_lee_igual(tmp_path: Path) -> None:
    """Git en Windows convierte el archivo a CRLF y `sha256sum -c` se rompe con eso; la app no tiene que romperse."""
    preparar(tmp_path, ARCHIVOS)
    lista = tmp_path / "SHA256SUMS.txt"
    lista.write_bytes(lista.read_bytes().replace(b"\n", b"\r\n"))
    assert b"\r\n" in lista.read_bytes()

    assert set(leer_sumas(lista)) == set(ARCHIVOS)  # los nombres no arrastran el retorno de carro
    verificar_hashes_modelos(tmp_path)


@pytest.mark.requisito("RF-22")
def test_leer_sumas_acepta_el_formato_de_sha256sum_con_o_sin_asterisco(tmp_path: Path) -> None:
    ruta = tmp_path / "SHA256SUMS.txt"
    ruta.write_text(f"{'A' * 64} *uno.pt\n\n{'b' * 64}  dos.pt\n", encoding="utf-8")
    assert leer_sumas(ruta) == {"uno.pt": "a" * 64, "dos.pt": "b" * 64}


@pytest.mark.requisito("RF-22")
@pytest.mark.parametrize("contenido", ["", "\n\n", "abc *x.pt\n", "solo-una-columna\n"])
def test_leer_sumas_rechaza_una_lista_vacia_o_mal_formada(tmp_path: Path, contenido: str) -> None:
    ruta = tmp_path / "SHA256SUMS.txt"
    ruta.write_text(contenido, encoding="utf-8")
    with pytest.raises(VerificacionModeloFallida):
        leer_sumas(ruta)


@pytest.mark.requisito("RF-22")
def test_sha256_de_archivo_lee_por_bloques(tmp_path: Path) -> None:
    contenido = b"x" * (3 * 1024 * 1024 + 17)  # más de un bloque
    ruta = tmp_path / "grande.bin"
    ruta.write_bytes(contenido)
    assert sha256_de_archivo(ruta) == sha(contenido)


@pytest.mark.modelo
@pytest.mark.requisito("RF-22")
def test_los_pesos_reales_coinciden_con_sha256sums(carpeta_modelos_reales: Path) -> None:
    """Necesita los `.pt` (python ../estimador/descargar_modelos.py). Si no están, se saltea."""
    assert carpeta_modelos_reales == CARPETA_MODELOS_REPO
    verificar_hashes_modelos(carpeta_modelos_reales)
