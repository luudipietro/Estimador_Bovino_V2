import pytest


@pytest.fixture(autouse=True)
def sin_logging_global(monkeypatch: pytest.MonkeyPatch) -> None:
    """`create_app` instala un manejador JSON en el root logger: los tests no quieren cambiarlo para el resto."""
    monkeypatch.setattr("bv_estimacion.api.main._configurar_logs", lambda *args, **kwargs: None)
