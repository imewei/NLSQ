"""Tests for Results/Export pages when no fit results exist."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


class _SessionState(dict):
    def __getattr__(self, name: str) -> object:
        return self.get(name)

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


class _ContextManager:
    def __init__(self, value: object) -> None:
        self._value = value

    def __enter__(self) -> object:
        return self._value

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _install_streamlit_stub() -> types.ModuleType:
    module = types.ModuleType("streamlit")
    module.session_state = _SessionState()
    module._messages: list[tuple[str, str | None]] = []

    def _record(kind: str):
        def _fn(message: str | None = None, *_a: object, **_k: object) -> None:
            module._messages.append((kind, message))

        return _fn

    module.title = lambda *_a, **_k: None
    module.markdown = lambda *_a, **_k: None
    module.warning = _record("warning")
    module.success = _record("success")
    module.error = _record("error")
    module.caption = _record("caption")
    module.columns = lambda n=1: tuple(_ContextManager(module) for _ in range(n))
    module.sidebar = module
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    stub = _install_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    def restore() -> None:
        sys.modules.pop("nlsq.gui.pages.4_Results", None)
        sys.modules.pop("nlsq.gui.pages.5_Export", None)

    request.addfinalizer(restore)
    return stub


@pytest.mark.gui
@pytest.mark.unit
def test_results_page_no_results(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.4_Results")
    monkeypatch.setattr(module, "st", streamlit_stub)

    state = SimpleNamespace(xdata=None, fit_result=None)
    monkeypatch.setattr(module, "initialize_state", lambda: state)
    streamlit_stub.session_state.current_model = None

    module.render_no_results()
    assert ("warning", "No fitting results available yet.") in streamlit_stub._messages
    assert ("error", "No data loaded") in streamlit_stub._messages
    assert ("error", "No model selected") in streamlit_stub._messages


@pytest.mark.gui
@pytest.mark.unit
def test_export_page_no_results(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.5_Export")
    monkeypatch.setattr(module, "st", streamlit_stub)

    state = SimpleNamespace(xdata=[1.0], fit_result=None)
    monkeypatch.setattr(module, "initialize_state", lambda: state)
    streamlit_stub.session_state.current_model = object()

    module.render_no_results()
    assert (
        "warning",
        "No fitting results available to export.",
    ) in streamlit_stub._messages
    assert ("success", "Data loaded") in streamlit_stub._messages
    assert ("success", "Model selected") in streamlit_stub._messages
