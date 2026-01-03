"""Targeted CLI main/handlers tests for error and success paths."""

from __future__ import annotations

import importlib
import types

import pytest

cli_main = importlib.import_module("nlsq.cli.main")
from nlsq.cli.errors import CLIError, ConfigError, DataLoadError, FitError, ModelError


@pytest.mark.parametrize(
    "error_cls, label",
    [
        (ConfigError, "Configuration Error"),
        (DataLoadError, "Data Loading Error"),
        (ModelError, "Model Error"),
        (FitError, "Fitting Error"),
        (CLIError, "CLI Error"),
    ],
)
def test_handle_fit_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_cls: type[CLIError],
    label: str,
) -> None:
    """handle_fit should format and report CLIError subclasses."""
    from nlsq.cli.commands import fit as fit_module

    def _raise(**_kwargs: object) -> None:
        raise error_cls("boom", context={"where": "test"}, suggestion="do x")

    monkeypatch.setattr(fit_module, "run_fit", _raise)

    args = types.SimpleNamespace(
        workflow="w.yaml", output=None, stdout=False, verbose=False
    )
    exit_code = cli_main.handle_fit(args)

    assert exit_code == 1
    captured = capsys.readouterr().err
    assert label in captured
    assert "boom" in captured
    assert "where: test" in captured
    assert "Suggestion" in captured


def test_handle_fit_result_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_fit should return 1 when run_fit returns None."""
    from nlsq.cli.commands import fit as fit_module

    monkeypatch.setattr(fit_module, "run_fit", lambda **_k: None)
    args = types.SimpleNamespace(
        workflow="w.yaml", output=None, stdout=False, verbose=False
    )
    assert cli_main.handle_fit(args) == 1


def test_handle_batch_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """handle_batch should return 1 when any workflow failed."""
    from nlsq.cli.commands import batch as batch_module

    monkeypatch.setattr(
        batch_module,
        "run_batch",
        lambda **_k: [{"status": "failed", "workflow": "w.yaml"}],
    )
    args = types.SimpleNamespace(
        workflows=["w.yaml"],
        summary=None,
        workers=None,
        continue_on_error=False,
        verbose=False,
    )
    assert cli_main.handle_batch(args) == 1
    assert "Batch completed with 1 failure" in capsys.readouterr().out


def test_handle_batch_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_batch should return 1 when run_batch raises CLIError."""
    from nlsq.cli.commands import batch as batch_module

    def _raise(**_kwargs: object) -> None:
        raise CLIError("batch failed")

    monkeypatch.setattr(batch_module, "run_batch", _raise)
    args = types.SimpleNamespace(
        workflows=["w.yaml"],
        summary=None,
        workers=None,
        continue_on_error=False,
        verbose=False,
    )
    assert cli_main.handle_batch(args) == 1


def test_handle_info_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_info should call the info runner."""
    from nlsq.cli.commands import info as info_module

    called: dict[str, bool] = {"ok": False}

    def _run_info(verbose: bool = False) -> None:
        called["ok"] = verbose

    monkeypatch.setattr(info_module, "run_info", _run_info)
    args = types.SimpleNamespace(verbose=True)
    assert cli_main.handle_info(args) == 0
    assert called["ok"] is True


def test_handle_config_file_exists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """handle_config should return 1 on FileExistsError."""
    from nlsq.cli.commands import config as config_module

    def _raise(**_kwargs: object) -> None:
        raise FileExistsError("exists")

    monkeypatch.setattr(config_module, "run_config", _raise)
    args = types.SimpleNamespace(
        workflow=False,
        model=False,
        output=None,
        force=False,
        verbose=False,
    )
    assert cli_main.handle_config(args) == 1
    assert "exists" in capsys.readouterr().err


def test_handle_gui_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_gui should return 1 if Qt dependencies are missing."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "nlsq.gui_qt" or name.startswith("nlsq.gui_qt"):
            raise ImportError("PySide6 not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    args = types.SimpleNamespace()
    assert cli_main.handle_gui(args) == 1


def test_handle_gui_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_gui should return 0 on KeyboardInterrupt."""

    def mock_run_desktop() -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr("nlsq.gui_qt.run_desktop", mock_run_desktop)
    args = types.SimpleNamespace()
    assert cli_main.handle_gui(args) == 0
