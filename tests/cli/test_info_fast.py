"""Fast tests for info command helpers with mocked dependencies."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from nlsq.cli.commands import info as info_module


def test_get_info_dict_without_jax_or_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_info_dict should handle missing optional deps gracefully."""
    original_import = builtins.__import__

    def _import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name in ("jax", "psutil"):
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)

    info = info_module.get_info_dict()
    assert info["jax_version"] is None
    assert info["jax_backend"] is None
    assert info["jax_devices"] == []
    assert info["memory"] is None


def test_print_jax_info_gpu_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_print_jax_info should report GPU backend when devices indicate GPU."""
    dummy = types.SimpleNamespace(
        __version__="0",
        devices=lambda: [types.SimpleNamespace(platform="gpu", device_kind="test")],
    )
    monkeypatch.setitem(sys.modules, "jax", dummy)

    info_module._print_jax_info(verbose=False)
    out = capsys.readouterr().out
    assert "Backend: GPU" in out


def test_print_memory_info_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_print_memory_info should report errors if psutil fails."""

    class DummyPsutil:
        def virtual_memory(self):
            raise RuntimeError("boom")

    monkeypatch.setitem(info_module.sys.modules, "psutil", DummyPsutil())

    info_module._print_memory_info()
    out = capsys.readouterr().out
    assert "Error detecting memory" in out
