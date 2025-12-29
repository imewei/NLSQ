"""Tests for compatibility shims."""

from __future__ import annotations

import types

import pytest

from nlsq.compat import get_deprecated_module


def test_get_deprecated_module_warns_and_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_deprecated_module should warn and delegate to importlib."""
    sentinel = types.SimpleNamespace(name="ok")

    def _import_module(name: str):
        assert name == "nlsq.utils.error_messages"
        return sentinel

    monkeypatch.setattr("importlib.import_module", _import_module)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        module = get_deprecated_module("error_messages")

    assert module is sentinel


def test_get_deprecated_module_unknown_raises() -> None:
    """Unknown compatibility mappings should raise ImportError."""
    with pytest.raises(ImportError, match="No compatibility mapping"):
        get_deprecated_module("unknown_module")
