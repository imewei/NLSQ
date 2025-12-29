"""Fast render-path tests for iteration_table using Streamlit stubs."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
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
    module._dataframes: list[object] = []
    module._metrics: list[tuple[str, str]] = []

    def _record(kind: str):
        def _fn(message: str | None = None, *_a: object, **_k: object) -> None:
            module._messages.append((kind, message))

        return _fn

    module.caption = _record("caption")
    module.success = _record("success")
    module.info = _record("info")
    module.markdown = lambda *_a, **_k: None
    module.dataframe = lambda df, **_k: module._dataframes.append(df)
    module.metric = lambda label, value, **_k: module._metrics.append((label, value))
    module.columns = lambda n=1: tuple(
        _ContextManager(module) for _ in range(n if isinstance(n, int) else len(n))
    )
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    stub = _install_streamlit_stub()
    module_name = "nlsq.gui.components.iteration_table"
    cached_module = sys.modules.get(module_name)
    cached_streamlit = sys.modules.get("streamlit")
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    sys.modules.pop(module_name, None)

    def restore() -> None:
        sys.modules.pop(module_name, None)
        if cached_module is not None:
            sys.modules[module_name] = cached_module
        if cached_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = cached_streamlit

    request.addfinalizer(restore)
    return stub


@pytest.mark.gui
@pytest.mark.unit
def test_render_iteration_table_empty(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.components.iteration_table")
    module.render_iteration_table({"iterations": [], "params": [], "costs": []})
    assert (
        "caption",
        "Parameter values will appear during fitting",
    ) in streamlit_stub._messages


@pytest.mark.gui
@pytest.mark.unit
def test_render_iteration_table_convergence_messages(
    streamlit_stub: types.ModuleType,
) -> None:
    module = importlib.import_module("nlsq.gui.components.iteration_table")

    history = {
        "iterations": [1, 2],
        "params": [np.array([1.0, 2.0]), np.array([1.0, 2.0])],
        "costs": [1.0, 0.5],
        "param_names": ["a", "b"],
    }
    module.render_iteration_table(history)
    assert streamlit_stub._dataframes
    assert any("converging" in (msg or "") for _, msg in streamlit_stub._messages)

    streamlit_stub._messages.clear()
    history = {
        "iterations": [1, 2],
        "params": [np.array([1.0, 2.0]), np.array([1.00005, 2.00005])],
        "costs": [1.0, 0.9],
        "param_names": ["a", "b"],
    }
    module.render_iteration_table(history)
    assert any("stabilizing" in (msg or "") for _, msg in streamlit_stub._messages)


@pytest.mark.gui
@pytest.mark.unit
def test_render_convergence_summary(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.components.iteration_table")

    history = {
        "iterations": [1, 2],
        "params": [np.array([1.0, 2.0]), np.array([1.5, 2.5])],
        "costs": [1.0, 0.5],
        "param_names": ["a", "b"],
    }
    module.render_convergence_summary(history)
    assert streamlit_stub._metrics
