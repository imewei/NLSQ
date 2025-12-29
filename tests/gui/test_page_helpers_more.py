"""Extra fast GUI helper tests to cover branch-heavy paths."""

from __future__ import annotations

import importlib
import io
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


class _Upload:
    def __init__(self, data: bytes, name: str = "file.csv") -> None:
        self._data = data
        self.name = name
        self._buffer = io.BytesIO(data)

    def read(self) -> bytes:
        return self._buffer.read()

    def seek(self, offset: int) -> None:
        self._buffer.seek(offset)


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
    module._rerun_called = False
    module._text_area_value = ""
    module._selectbox_value = None
    module._button_value = False
    module._file_value = None

    def _record(kind: str):
        def _fn(message: str | None = None, *_args: object, **_kwargs: object) -> None:
            module._messages.append((kind, message))

        return _fn

    module.set_page_config = lambda **_kwargs: None
    module.success = _record("success")
    module.warning = _record("warning")
    module.error = _record("error")
    module.info = _record("info")
    module.caption = _record("caption")
    module.subheader = lambda *_a, **_k: None
    module.title = lambda *_a, **_k: None
    module.markdown = lambda *_a, **_k: None
    module.code = lambda *_a, **_k: None
    module.divider = lambda *_a, **_k: None
    module.sidebar = module
    module.tabs = lambda labels: tuple(_ContextManager(module) for _ in labels)
    module.text_area = lambda *_a, **_k: module._text_area_value
    module.selectbox = lambda *_a, **_k: module._selectbox_value
    module.button = lambda *_a, **_k: module._button_value
    module.file_uploader = lambda *_a, **_k: module._file_value
    module.radio = lambda *_a, **_k: "Guided"
    module.rerun = lambda *_a, **_k: setattr(module, "_rerun_called", True)
    module.columns = lambda n=1: tuple(_ContextManager(module) for _ in range(n))
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.mark.gui
@pytest.mark.unit
def test_data_loading_file_upload_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _install_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    sys.modules.pop("nlsq.gui.pages.1_Data_Loading", None)
    module = importlib.import_module("nlsq.gui.pages.1_Data_Loading")
    monkeypatch.setattr(module, "st", stub)
    monkeypatch.setattr(module, "initialize_state", lambda: SimpleNamespace())

    stub.session_state.column_assignments = {"x": 0, "y": 1, "z": None, "sigma": None}

    def _raise(*_a, **_k):
        raise ValueError("bad file")

    monkeypatch.setattr(module, "load_from_file", _raise)

    upload = _Upload(b"x,y\n1,2\n2,4")
    module.handle_file_upload(upload, format_override="auto")

    assert ("error", "Error loading file: bad file") in stub._messages


@pytest.mark.gui
@pytest.mark.unit
def test_data_loading_file_upload_fallback_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    sys.modules.pop("nlsq.gui.pages.1_Data_Loading", None)
    module = importlib.import_module("nlsq.gui.pages.1_Data_Loading")
    monkeypatch.setattr(module, "st", stub)
    state = SimpleNamespace()
    monkeypatch.setattr(module, "initialize_state", lambda: state)

    stub.session_state.column_assignments = {"x": 0, "y": 1, "z": None, "sigma": None}

    x = np.array([1.0, 2.0])
    y = np.array([2.0, 4.0])
    sigma = None
    monkeypatch.setattr(module, "load_from_file", lambda *_a, **_k: (x, y, sigma))

    upload = _Upload(b"x,y\n1,2\n2,4", name="data.csv")
    module.handle_file_upload(upload, format_override="auto")

    assert isinstance(stub.session_state.raw_data, pd.DataFrame)
    assert list(stub.session_state.raw_data.columns) == ["x", "y"]
    assert stub.session_state.data_source == "file"
    assert ("success", "Loaded 2 data points from data.csv") in stub._messages


@pytest.mark.gui
@pytest.mark.unit
def test_model_selection_custom_code_editor_error_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    sys.modules.pop("nlsq.gui.pages.2_Model_Selection", None)
    module = importlib.import_module("nlsq.gui.pages.2_Model_Selection")
    monkeypatch.setattr(module, "st", stub)
    state = SimpleNamespace(custom_code="", custom_function_name="model")

    stub.session_state.custom_code = ""
    stub._text_area_value = "def model(x):\n    return x"
    monkeypatch.setattr(
        module, "validate_code_syntax", lambda *_a, **_k: (False, "bad")
    )

    module.render_custom_code_editor(state)
    assert ("error", "Syntax error: bad") in stub._messages

    stub._messages.clear()
    monkeypatch.setattr(module, "validate_code_syntax", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(module, "list_functions_in_module", lambda *_a, **_k: ["model"])
    stub._selectbox_value = "model"
    stub._button_value = True

    def _model(x, a):
        return x + a

    monkeypatch.setattr(module, "load_custom_model", lambda *_a, **_k: _model)
    monkeypatch.setattr(
        module,
        "get_model_info",
        lambda *_a, **_k: {"param_names": ["a"], "param_count": 1},
    )

    module.render_custom_code_editor(state)
    assert stub.session_state.model_loaded is True
    assert stub.session_state.current_model is _model
    assert any("loaded successfully" in (msg or "") for kind, msg in stub._messages)


@pytest.mark.gui
@pytest.mark.unit
def test_fitting_options_yaml_import_invalid_and_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    sys.modules.pop("nlsq.gui.pages.3_Fitting_Options", None)
    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    monkeypatch.setattr(module, "st", stub)
    from nlsq.gui.state import SessionState

    state = SessionState()
    monkeypatch.setattr(module, "get_session_state", lambda: state)

    stub._file_value = _Upload(b"bad: [", name="config.yml")
    monkeypatch.setattr(
        module, "validate_yaml_config", lambda *_a, **_k: (False, "oops")
    )

    module.render_yaml_import()
    assert ("error", "Invalid YAML: oops") in stub._messages

    stub._messages.clear()
    stub._rerun_called = False
    stub._file_value = _Upload(b"ok: true", name="config.yml")
    monkeypatch.setattr(module, "validate_yaml_config", lambda *_a, **_k: (True, ""))
    imported = SessionState()
    imported.model_type = "polynomial"
    monkeypatch.setattr(module, "load_yaml_config", lambda *_a, **_k: imported)

    module.render_yaml_import()
    assert state.model_type == "polynomial"
    assert any("Configuration imported" in (msg or "") for kind, msg in stub._messages)
    assert stub._rerun_called is True


@pytest.mark.gui
@pytest.mark.unit
def test_fitting_options_guided_mode_applies_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    sys.modules.pop("nlsq.gui.pages.3_Fitting_Options", None)
    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    monkeypatch.setattr(module, "st", stub)
    from nlsq.gui.state import SessionState

    state = SessionState()
    state.preset = "standard"
    stub.radio = lambda *_a, **_k: "fast"

    applied: list[str] = []
    monkeypatch.setattr(
        module, "apply_preset_to_state", lambda _s, name: applied.append(name)
    )

    module.render_guided_mode(state)
    assert applied == ["fast"]
    assert stub._rerun_called is True
