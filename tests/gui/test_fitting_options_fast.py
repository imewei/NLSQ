"""Fast tests for fitting options page helper logic and render paths."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

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
    module._radio_value: object | None = None
    module._button_values: list[bool] = []
    module._checkbox_values: list[object] = []
    module._text_area_value = ""
    module._selectbox_values: list[object] = []
    module._slider_values: list[object] = []
    module._number_values: list[object] = []
    module._progress_calls: list[float] = []
    module._rerun_called = False

    def _record(kind: str):
        def _fn(message: str | None = None, *_a: object, **_k: object) -> None:
            module._messages.append((kind, message))

        return _fn

    def _pop(values: list[object], default: object) -> object:
        return values.pop(0) if values else default

    module.set_page_config = lambda **_k: None
    module.header = lambda *_a, **_k: None
    module.subheader = lambda *_a, **_k: None
    module.markdown = lambda *_a, **_k: None
    module.caption = _record("caption")
    module.divider = lambda *_a, **_k: None
    module.metric = lambda *_a, **_k: None
    module.info = _record("info")
    module.success = _record("success")
    module.warning = _record("warning")
    module.error = _record("error")
    module.progress = lambda value, **_k: module._progress_calls.append(value)
    module.button = lambda *_a, **_k: bool(_pop(module._button_values, False))
    module.radio = lambda _label, options, **_k: module._radio_value or options[0]
    module.checkbox = lambda _label, **_k: _pop(
        module._checkbox_values, _k.get("value", False)
    )
    module.text_area = lambda *_a, **_k: module._text_area_value
    module.selectbox = lambda _label, options, **_k: _pop(
        module._selectbox_values, options[0]
    )
    module.slider = lambda _label, **_k: _pop(module._slider_values, _k.get("value", 0))
    module.number_input = lambda _label, **_k: _pop(
        module._number_values, _k.get("value", 0)
    )
    module.columns = lambda n=1: tuple(
        _ContextManager(module) for _ in range(n if isinstance(n, int) else len(n))
    )
    module.tabs = lambda labels: tuple(_ContextManager(module) for _ in labels)
    module.expander = lambda *_a, **_k: _ContextManager(module)
    module.rerun = lambda *_a, **_k: setattr(module, "_rerun_called", True)
    module.sidebar = module
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    stub = _install_streamlit_stub()
    module_name = "nlsq.gui.pages.3_Fitting_Options"
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
def test_is_ready_to_fit_messages(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    module.st = streamlit_stub

    state = SimpleNamespace(
        xdata=None,
        ydata=None,
        p0=None,
        auto_p0=False,
        fit_running=False,
    )
    ready, message = module.is_ready_to_fit(state)
    assert ready is False
    assert "Data not loaded" in message

    state.xdata = np.array([1.0])
    state.ydata = np.array([2.0])
    streamlit_stub.session_state.current_model = None
    ready, message = module.is_ready_to_fit(state)
    assert ready is False
    assert "Model not selected" in message

    streamlit_stub.session_state.current_model = object()
    ready, message = module.is_ready_to_fit(state)
    assert ready is False
    assert "Initial parameters not set" in message

    state.p0 = [None]
    ready, message = module.is_ready_to_fit(state)
    assert ready is False
    assert "Initial parameters not set" in message

    state.p0 = [1.0]
    state.fit_running = True
    ready, message = module.is_ready_to_fit(state)
    assert ready is False
    assert message == "Fit is already running."

    state.fit_running = False
    ready, message = module.is_ready_to_fit(state)
    assert ready is True
    assert message == "Ready to fit"


@pytest.mark.gui
@pytest.mark.unit
def test_run_fit_auto_p0_failure(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    module.st = streamlit_stub

    class Model:
        def estimate_p0(self, *_a, **_k):
            raise RuntimeError("boom")

    streamlit_stub.session_state.current_model = Model()
    state = SimpleNamespace(
        xdata=np.array([1.0]),
        ydata=np.array([2.0]),
        sigma=None,
        auto_p0=True,
        p0=None,
        fit_running=False,
        fit_aborted=False,
        fit_result=None,
    )

    module.create_cost_history = lambda: {"iterations": []}
    module.create_iteration_history = lambda: {"iterations": []}

    error = module.run_fit(state)
    assert error is not None
    assert "Auto p0 estimation failed" in error
    assert state.fit_running is False


@pytest.mark.gui
@pytest.mark.unit
def test_run_fit_p0_missing_error(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    module.st = streamlit_stub

    streamlit_stub.session_state.current_model = object()
    state = SimpleNamespace(
        xdata=np.array([1.0]),
        ydata=np.array([2.0]),
        sigma=None,
        auto_p0=False,
        p0=None,
        fit_running=False,
        fit_aborted=False,
        fit_result=None,
    )

    module.create_cost_history = lambda: {"iterations": []}
    module.create_iteration_history = lambda: {"iterations": []}

    error = module.run_fit(state)
    assert error == "p0 contains undefined values. Please set initial parameters."
    assert state.fit_running is False


@pytest.mark.gui
@pytest.mark.unit
def test_run_fit_success_path(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    module.st = streamlit_stub

    class Model:
        def estimate_p0(self, *_a, **_k):
            return [1.0]

    streamlit_stub.session_state.current_model = Model()
    state = SimpleNamespace(
        xdata=np.array([1.0]),
        ydata=np.array([2.0]),
        sigma=None,
        auto_p0=True,
        p0=None,
        fit_running=False,
        fit_aborted=False,
        fit_result=None,
    )

    module.create_cost_history = lambda: {"iterations": []}
    module.create_iteration_history = lambda: {"iterations": []}
    module.create_fit_config_from_state = lambda *_a, **_k: {"ok": True}
    module.execute_fit = lambda *_a, **_k: SimpleNamespace(success=True)

    error = module.run_fit(state)
    assert error is None
    assert state.fit_result is not None
    assert state.fit_running is False


@pytest.mark.gui
@pytest.mark.unit
def test_render_fit_execution_and_summary(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    module.st = streamlit_stub

    state = SimpleNamespace(
        xdata=np.array([1.0]),
        ydata=np.array([2.0]),
        sigma=None,
        auto_p0=False,
        p0=[1.0],
        fit_running=False,
        fit_aborted=False,
        fit_result=None,
        max_iterations=10,
    )

    module.is_ready_to_fit = lambda *_a, **_k: (False, "not ready")
    module.render_fit_execution_section(state)
    assert ("info", "not ready") in streamlit_stub._messages

    streamlit_stub._messages.clear()
    module.is_ready_to_fit = lambda *_a, **_k: (True, "Ready to fit")
    streamlit_stub.session_state.fit_error = "boom"
    module.render_fit_execution_section(state)
    assert ("error", "boom") in streamlit_stub._messages

    streamlit_stub._messages.clear()
    streamlit_stub.session_state.fit_error = None
    state.fit_result = SimpleNamespace(success=True)
    module.render_fit_execution_section(state)
    assert ("success", "Fit completed successfully") in streamlit_stub._messages

    streamlit_stub._messages.clear()
    module.render_live_cost_plot = lambda *_a, **_k: None
    module.render_iteration_table = lambda *_a, **_k: None
    module.is_large_dataset = lambda *_a, **_k: True
    streamlit_stub.session_state.cost_history = {"iterations": [1, 2]}
    streamlit_stub.session_state.iteration_history = {"iterations": [1, 2]}
    module.render_fit_progress(state)
    assert streamlit_stub._progress_calls

    module.extract_fit_statistics = lambda *_a, **_k: {
        "r_squared": 0.9,
        "rmse": 0.1,
        "aic": 1.0,
        "bic": 2.0,
    }
    module.extract_convergence_info = lambda *_a, **_k: {
        "nfev": 5,
        "cost": 0.01,
    }
    module.extract_confidence_intervals = lambda *_a, **_k: [(0.9, 1.1)]
    module.get_param_names_from_model = lambda *_a, **_k: ["a"]
    streamlit_stub.session_state.current_model = object()
    state.fit_result = SimpleNamespace(popt=np.array([1.0]))
    module.render_fit_summary(state)
    assert any(
        "View detailed results" in (msg or "") for _, msg in streamlit_stub._messages
    )
