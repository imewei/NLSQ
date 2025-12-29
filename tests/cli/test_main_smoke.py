"""Lightweight CLI smoke tests for main dispatch and error branches."""

from __future__ import annotations

import importlib
import types

import pytest

cli_main = importlib.import_module("nlsq.cli.main")


def test_main_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """main() with no args should print help and return 0."""
    exit_code = cli_main.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out.lower()


def test_handle_config_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle_config should return 1 on ValueError from run_config."""
    from nlsq.cli.commands import config as config_module

    def _raise_value_error(**_kwargs: object) -> None:
        raise ValueError("bad config")

    monkeypatch.setattr(config_module, "run_config", _raise_value_error)

    args = types.SimpleNamespace(
        workflow=False,
        model=False,
        output=None,
        force=False,
        verbose=False,
    )
    assert cli_main.handle_config(args) == 1


def test_main_dispatch_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() should dispatch to handle_config for the config subcommand."""
    calls: list[dict[str, object]] = []

    def _fake_run_config(**kwargs: object) -> None:
        calls.append(kwargs)

    from nlsq.cli.commands import config as config_module

    monkeypatch.setattr(config_module, "run_config", _fake_run_config)

    exit_code = cli_main.main(["config", "--workflow", "-o", "workflow.yaml"])

    assert exit_code == 0
    assert calls
    assert calls[0]["workflow"] is True
    assert calls[0]["output"] == "workflow.yaml"
