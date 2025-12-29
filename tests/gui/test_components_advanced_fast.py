"""Fast tests for advanced GUI components and helpers."""

from __future__ import annotations

import importlib
import io
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


class _Upload:
    def __init__(self, content: str, name: str = "model.py") -> None:
        self._buffer = io.BytesIO(content.encode("utf-8"))
        self.name = name

    def read(self) -> bytes:
        return self._buffer.read()


def _install_streamlit_stub() -> types.ModuleType:
    module = types.ModuleType("streamlit")
    module.session_state = _SessionState()
    module._messages: list[tuple[str, str | None]] = []
    module._metrics: list[tuple[str, object]] = []
    module._selectbox_values: list[object] = []
    module._slider_values: list[object] = []
    module._number_values: list[object] = []
    module._checkbox_values: list[object] = []
    module._toggle_values: list[object] = []
    module._text_input_values: list[object] = []
    module._text_area_value = ""
    module._file_uploader_value = None
    module._button_values: list[bool] = []

    def _record(kind: str):
        def _fn(message: str | None = None, *_a: object, **_k: object) -> None:
            module._messages.append((kind, message))

        return _fn

    def _pop(values: list[object], default: object) -> object:
        return values.pop(0) if values else default

    module.subheader = lambda *_a, **_k: None
    module.header = lambda *_a, **_k: None
    module.markdown = _record("markdown")
    module.caption = _record("caption")
    module.divider = lambda *_a, **_k: None
    module.code = lambda *_a, **_k: None
    module.info = _record("info")
    module.success = _record("success")
    module.warning = _record("warning")
    module.error = _record("error")
    module.metric = lambda label, value, **_k: module._metrics.append((label, value))
    module.columns = lambda n=1: tuple(_ContextManager(module) for _ in range(n))
    module.tabs = lambda labels: tuple(_ContextManager(module) for _ in labels)
    module.expander = lambda *_a, **_k: _ContextManager(module)
    module.text_area = lambda *_a, **_k: module._text_area_value
    module.selectbox = lambda _label, options, **_k: _pop(
        module._selectbox_values, options[0]
    )
    module.slider = lambda _label, **_k: _pop(module._slider_values, _k.get("value", 0))
    module.number_input = lambda _label, **_k: _pop(
        module._number_values, _k.get("value", 0)
    )
    module.checkbox = lambda _label, **_k: _pop(
        module._checkbox_values, _k.get("value", False)
    )
    module.toggle = lambda _label, **_k: _pop(
        module._toggle_values, _k.get("value", False)
    )
    module.text_input = lambda _label, **_k: _pop(
        module._text_input_values, _k.get("value", "")
    )
    module.file_uploader = lambda *_a, **_k: module._file_uploader_value
    module.button = lambda *_a, **_k: bool(_pop(module._button_values, False))
    module.dataframe = lambda *_a, **_k: None
    module.latex = lambda *_a, **_k: None
    module.sidebar = module
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    stub = _install_streamlit_stub()
    module_names = (
        "nlsq.gui.components.advanced_options",
        "nlsq.gui.components.code_editor",
        "nlsq.gui.components.model_preview",
    )
    cached_modules = {
        name: sys.modules[name] for name in module_names if name in sys.modules
    }
    cached_streamlit = sys.modules.get("streamlit")
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    for module_name in module_names:
        sys.modules.pop(module_name, None)

    def restore() -> None:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        for module_name, module in cached_modules.items():
            sys.modules[module_name] = module
        if cached_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = cached_streamlit

    request.addfinalizer(restore)
    return stub


@pytest.mark.gui
@pytest.mark.unit
def test_advanced_options_helpers_and_tabs(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.components.advanced_options")
    from nlsq.gui.state import SessionState

    state = SessionState()

    assert module.validate_max_iterations(1) is True
    assert module.validate_max_iterations(0) is False
    assert module.validate_chunk_size(10) is True
    assert module.validate_n_starts(0) is False

    streamlit_stub._selectbox_values = ["lm", "huber"]
    streamlit_stub._number_values = [10, 20]
    streamlit_stub._slider_values = [-6, -7, -8]
    module.render_fitting_tab(state)
    assert state.method == "lm"
    assert state.loss == "huber"

    streamlit_stub._toggle_values = [True]
    streamlit_stub._slider_values = [5, 1.2]
    streamlit_stub._selectbox_values = ["sobol"]
    streamlit_stub._checkbox_values = [False]
    module.render_multistart_tab(state)
    assert state.enable_multistart is True
    assert state.n_starts == 5
    assert state.sampler == "sobol"

    streamlit_stub._selectbox_values = [module.get_streaming_preset_names()[0]]
    streamlit_stub._number_values = [500, 50, 200]
    streamlit_stub._checkbox_values = [True, True, False, True, True]
    streamlit_stub._slider_values = [0.01, 0.2, 0.5]
    module.render_streaming_tab(state)
    assert state.chunk_size == 500
    assert state.normalize is True
    assert state.layer1_enabled is True
    assert state.layer3_enabled is True
    assert state.layer4_enabled is True

    streamlit_stub._checkbox_values = [True, True]
    streamlit_stub._text_input_values = ["/tmp/checkpoints"]
    module.render_hpc_tab(state)
    assert state.enable_multi_device is True
    assert state.checkpoint_dir == "/tmp/checkpoints"

    streamlit_stub._checkbox_values = [False, False]
    streamlit_stub._number_values = [8]
    streamlit_stub._selectbox_values = ["csv"]
    module.render_batch_tab(state)
    assert state.batch_max_workers == 8
    assert state.batch_continue_on_error is False
    assert state.batch_summary_format == "csv"


@pytest.mark.gui
@pytest.mark.unit
def test_render_advanced_options_tabs(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.components.advanced_options")
    from nlsq.gui.state import SessionState

    state = SessionState()
    module.render_advanced_options(state)


@pytest.mark.gui
@pytest.mark.unit
def test_multistart_disabled_caption(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.components.advanced_options")
    from nlsq.gui.state import SessionState

    state = SessionState()
    streamlit_stub._toggle_values = [False]
    module.render_multistart_tab(state)
    assert any(
        "Enable multi-start" in (msg or "") for _, msg in streamlit_stub._messages
    )


@pytest.mark.gui
@pytest.mark.unit
def test_batch_tab_auto_workers_caption(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.components.advanced_options")
    from nlsq.gui.state import SessionState

    state = SessionState()
    streamlit_stub._checkbox_values = [True, False]
    streamlit_stub._selectbox_values = ["json"]
    module.render_batch_tab(state)
    assert state.batch_max_workers is None
    assert any("Workers: Auto" in (msg or "") for _, msg in streamlit_stub._messages)


@pytest.mark.gui
@pytest.mark.unit
def test_code_editor_validation_and_upload(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.components.code_editor")

    is_valid, msg = module.validate_code_syntax("")
    assert is_valid is False
    assert msg == "Code is empty"

    monkeypatch.setattr(module, "list_functions_in_module", lambda *_a, **_k: [])
    monkeypatch.setattr(module, "validate_jit_compatibility", lambda *_a, **_k: False)
    status = module.get_code_validation_status("def f(x):\n    return x")
    assert status["is_valid"] is True
    assert status["functions"] == []
    assert status["warnings"]

    streamlit_stub._text_area_value = "def f(x):\n    return x"
    monkeypatch.setattr(module, "get_current_theme", lambda: "dark")
    monkeypatch.setattr(module, "get_code_editor_theme", lambda *_a, **_k: "dark")
    monkeypatch.setattr(
        module,
        "get_code_validation_status",
        lambda *_a, **_k: {
            "is_valid": True,
            "error_message": "",
            "is_jit_compatible": True,
            "functions": ["f"],
            "warnings": ["warn"],
        },
    )
    module.render_code_editor("def f(x):\n    return x")
    assert ("success", "Syntax is valid") in streamlit_stub._messages

    streamlit_stub._file_uploader_value = _Upload("def model(x):\n    return x")
    monkeypatch.setattr(module, "validate_code_syntax", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(module, "list_functions_in_module", lambda *_a, **_k: ["model"])
    streamlit_stub._selectbox_values = ["model"]
    file_name, func_name = module.render_file_upload()
    assert file_name == "model.py"
    assert func_name == "model"
    assert "model_file_content" in streamlit_stub.session_state


@pytest.mark.gui
@pytest.mark.unit
def test_model_preview_rendering(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.components.model_preview")

    assert module.format_parameter_list([]) == "(none)"
    assert "f(x" in module.get_equation_for_model("custom")

    monkeypatch.setattr(
        module,
        "get_model_summary",
        lambda *_a, **_k: {
            "param_count": 2,
            "param_names": ["a", "b"],
            "param_display": "a, b",
            "has_auto_p0": True,
            "has_auto_bounds": False,
            "equation": "y=ax+b",
        },
    )
    monkeypatch.setattr(module, "render_equation_display", lambda *_a, **_k: None)

    module.render_model_preview(object(), "builtin", model_name="linear")
    assert ("success", "Auto initial guess available") in streamlit_stub._messages

    streamlit_stub._messages.clear()
    module.render_parameter_table([])
    assert ("info", "No parameters to display") in streamlit_stub._messages

    streamlit_stub._messages.clear()
    monkeypatch.setattr(
        module,
        "get_model_summary",
        lambda *_a, **_k: {
            "has_auto_p0": False,
            "has_auto_bounds": True,
        },
    )
    module.render_model_capabilities(object())
    assert any("Auto p0" in (msg or "") for _, msg in streamlit_stub._messages)
