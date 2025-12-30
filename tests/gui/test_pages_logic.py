"""Fast logic tests for GUI page helpers with Streamlit stubs."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class _SessionState(dict):
    """Dict-backed session state with attribute access."""

    def __getattr__(self, name: str) -> object:
        return self.get(name)

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


def _install_streamlit_stub() -> types.ModuleType:
    module = types.ModuleType("streamlit")
    module.session_state = _SessionState()
    module._messages: list[tuple[str, str | None]] = []

    def _record(kind: str):
        def _fn(message: str | None = None, *_args: object, **_kwargs: object) -> None:
            module._messages.append((kind, message))

        return _fn

    module.set_page_config = lambda **_kwargs: None
    module.success = _record("success")
    module.warning = _record("warning")
    module.error = _record("error")
    module.markdown = lambda *_args, **_kwargs: None
    module.write = lambda *_args, **_kwargs: None
    module.sidebar = module
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture()
def _streamlit_pages_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    stub = _install_streamlit_stub()
    cached_modules: dict[str, types.ModuleType] = {}
    streamlit_cache = sys.modules.get("streamlit")

    for name in list(sys.modules):
        if name.startswith("nlsq.gui.pages."):
            cached_modules[name] = sys.modules.pop(name)

    monkeypatch.setitem(sys.modules, "streamlit", stub)
    yield stub

    for name in list(sys.modules):
        if name.startswith("nlsq.gui.pages."):
            sys.modules.pop(name, None)

    if streamlit_cache is None:
        sys.modules.pop("streamlit", None)
    else:
        sys.modules["streamlit"] = streamlit_cache

    sys.modules.update(cached_modules)


@pytest.mark.gui
def test_data_page_state_and_clipboard_flow(
    monkeypatch: pytest.MonkeyPatch, _streamlit_pages_stub: types.ModuleType
) -> None:
    """Data page helpers should initialize state and parse clipboard data."""
    stub = _streamlit_pages_stub

    module = importlib.import_module("nlsq.gui.pages.1_Data_Loading")

    state = SimpleNamespace()
    monkeypatch.setattr(module, "initialize_state", lambda: state)
    stub.session_state.nlsq_state = state
    assert module.get_session_state() is state

    module.init_data_page_state()
    assert "raw_data" in stub.session_state
    assert "column_assignments" in stub.session_state
    assert stub.session_state.column_assignments["x"] == 0

    module.handle_clipboard_paste("", has_header=False)
    assert ("warning", "Please paste some data first") in stub._messages

    stub._messages.clear()
    stub.session_state.column_assignments = {"x": 0, "y": 1, "z": None, "sigma": None}

    x = np.array([1.0, 2.0])
    y = np.array([2.0, 4.0])
    sigma = None
    monkeypatch.setattr(module, "load_from_clipboard", lambda *_a, **_k: (x, y, sigma))

    module.handle_clipboard_paste("x,y\n1,2\n2,4", has_header=True)

    assert state.xdata is x
    assert state.ydata is y
    assert state.data_file_name == "clipboard"
    assert stub.session_state.data_source == "clipboard"
    assert isinstance(stub.session_state.raw_data, pd.DataFrame)
    assert ("success", "Parsed 2 data points from clipboard") in stub._messages


@pytest.mark.gui
def test_model_selection_load_custom_fallback(
    monkeypatch: pytest.MonkeyPatch, _streamlit_pages_stub: types.ModuleType
) -> None:
    """Custom model loader should return None on adapter errors."""
    assert _streamlit_pages_stub

    module = importlib.import_module("nlsq.gui.pages.2_Model_Selection")

    monkeypatch.setattr(module, "get_model", lambda *_a, **_k: "ok")
    assert module.load_builtin_model("linear") == "ok"
    assert module.load_polynomial_model(2) == "ok"

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise ValueError("bad model")

    monkeypatch.setattr(module, "get_model", _raise)
    assert module.load_custom_model("def model(x): return x", "model") is None


@pytest.mark.gui
def test_fitting_options_is_ready_to_fit(
    monkeypatch: pytest.MonkeyPatch, _streamlit_pages_stub: types.ModuleType
) -> None:
    """is_ready_to_fit should enforce data/model/p0 readiness rules."""
    stub = _streamlit_pages_stub

    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    from nlsq.gui.state import SessionState

    state = SessionState()
    ready, msg = module.is_ready_to_fit(state)
    assert ready is False
    assert "Data not loaded" in msg

    state.xdata = np.array([1.0])
    state.ydata = np.array([2.0])
    ready, msg = module.is_ready_to_fit(state)
    assert ready is False
    assert "Model not selected" in msg

    class ModelWithP0:
        def estimate_p0(self, *_args: object) -> None:
            return None

    stub.session_state.current_model = ModelWithP0()
    state.auto_p0 = False
    ready, msg = module.is_ready_to_fit(state)
    assert ready is False
    assert "Initial parameters not set" in msg

    state.auto_p0 = True
    ready, msg = module.is_ready_to_fit(state)
    assert ready is True
    assert msg == "Ready to fit"


@pytest.mark.gui
def test_results_and_export_validity(
    monkeypatch: pytest.MonkeyPatch, _streamlit_pages_stub: types.ModuleType
) -> None:
    """Results/export pages should detect whether a valid result exists."""
    assert _streamlit_pages_stub

    results_module = importlib.import_module("nlsq.gui.pages.4_Results")
    export_module = importlib.import_module("nlsq.gui.pages.5_Export")

    state = SimpleNamespace(fit_result=None)
    monkeypatch.setattr(results_module, "initialize_state", lambda: state)
    monkeypatch.setattr(export_module, "initialize_state", lambda: state)

    assert results_module.has_valid_result() is False
    assert export_module.has_valid_result() is False

    class Result:
        popt = np.array([1.0])

    state.fit_result = Result()
    assert results_module.has_valid_result() is True
    assert export_module.has_valid_result() is True
