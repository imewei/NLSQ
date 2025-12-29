"""Fast render-path tests for GUI pages using Streamlit stubs."""

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
    def __init__(self, content: str, name: str = "data.csv") -> None:
        self._buffer = io.BytesIO(content.encode("utf-8"))
        self.name = name

    def read(self) -> bytes:
        return self._buffer.read()

    def seek(self, offset: int) -> None:
        self._buffer.seek(offset)


def _install_streamlit_stub() -> types.ModuleType:
    module = types.ModuleType("streamlit")
    module.session_state = _SessionState()
    module._messages: list[tuple[str, str | None]] = []
    module._selectbox_values: list[object] = []
    module._slider_values: list[object] = []
    module._number_values: list[object] = []
    module._checkbox_values: list[object] = []
    module._button_values: list[bool] = []
    module._radio_value: object | None = None
    module._file_uploader_value = None
    module._text_area_value = ""
    module._rerun_called = False

    def _record(kind: str):
        def _fn(message: str | None = None, *_a: object, **_k: object) -> None:
            module._messages.append((kind, message))

        return _fn

    def _pop(values: list[object], default: object) -> object:
        return values.pop(0) if values else default

    module.set_page_config = lambda **_k: None
    module.title = lambda *_a, **_k: None
    module.header = lambda *_a, **_k: None
    module.subheader = lambda *_a, **_k: None
    module.markdown = lambda *_a, **_k: None
    module.caption = _record("caption")
    module.divider = lambda *_a, **_k: None
    module.metric = lambda *_a, **_k: None
    module.dataframe = lambda *_a, **_k: None
    module.code = lambda *_a, **_k: None
    module.info = _record("info")
    module.success = _record("success")
    module.warning = _record("warning")
    module.error = _record("error")
    module.download_button = lambda *_a, **_k: None
    module.latex = lambda *_a, **_k: None
    module.write = lambda *_a, **_k: None
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
    module.button = lambda *_a, **_k: bool(_pop(module._button_values, False))
    module.radio = lambda _label, options, **_k: module._radio_value or options[0]
    module.tabs = lambda labels: tuple(_ContextManager(module) for _ in labels)
    module.columns = lambda n=1: tuple(
        _ContextManager(module) for _ in range(n if isinstance(n, int) else len(n))
    )
    module.expander = lambda *_a, **_k: _ContextManager(module)
    module.file_uploader = lambda *_a, **_k: module._file_uploader_value
    module.rerun = lambda *_a, **_k: setattr(module, "_rerun_called", True)
    module.sidebar = module
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


def _has_message(stub: types.ModuleType, kind: str, text: str) -> bool:
    return any(
        message_kind == kind and message and text in message
        for message_kind, message in stub._messages
    )


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    stub = _install_streamlit_stub()
    module_names = (
        "nlsq.gui.pages.1_Data_Loading",
        "nlsq.gui.pages.2_Model_Selection",
        "nlsq.gui.pages.3_Fitting_Options",
        "nlsq.gui.pages.4_Results",
        "nlsq.gui.pages.5_Export",
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
def test_data_loading_render_paths(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.1_Data_Loading")
    monkeypatch.setattr(module, "st", streamlit_stub)

    state = SimpleNamespace(data_mode="1d", xdata=None, ydata=None, sigma=None)
    monkeypatch.setattr(module, "initialize_state", lambda: state)

    streamlit_stub._file_uploader_value = _Upload("x,y\n1,2")
    streamlit_stub._button_values = [True, True]
    streamlit_stub._text_area_value = "x,y\n1,2"

    called = {"file": False, "clip": False}
    monkeypatch.setattr(
        module, "handle_file_upload", lambda *_a, **_k: called.__setitem__("file", True)
    )
    monkeypatch.setattr(
        module,
        "handle_clipboard_paste",
        lambda *_a, **_k: called.__setitem__("clip", True),
    )

    module.render_data_input_section()
    assert called["file"] is True
    assert called["clip"] is True

    streamlit_stub.session_state.raw_data = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [2.0, 4.0]}
    )
    streamlit_stub.session_state.column_assignments = {"x": 0, "y": 1}

    monkeypatch.setattr(module, "get_available_roles", lambda *_a, **_k: ["x", "y"])
    monkeypatch.setattr(module, "get_required_roles", lambda *_a, **_k: ["x", "y"])
    monkeypatch.setattr(module, "get_role_display_name", lambda role: role.upper())
    monkeypatch.setattr(module, "get_column_color", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "validate_column_selections",
        lambda *_a, **_k: {"is_valid": True, "message": ""},
    )
    streamlit_stub._selectbox_values = [0, 1]
    module.render_column_selector()
    assert ("success", "Column assignments are valid") in streamlit_stub._messages

    streamlit_stub._slider_values = [2]
    module.render_data_preview()

    streamlit_stub._button_values = [True]
    module.render_apply_button()
    assert state.xdata is not None
    assert streamlit_stub._rerun_called is True

    validation = SimpleNamespace(
        is_valid=True, point_count=2, message="", nan_count=0, inf_count=0
    )
    monkeypatch.setattr(module, "validate_data", lambda *_a, **_k: validation)
    monkeypatch.setattr(
        module,
        "compute_statistics",
        lambda *_a, **_k: {
            "point_count": 2,
            "is_2d": False,
            "x_min": 1.0,
            "x_max": 2.0,
            "y_min": 2.0,
            "y_max": 4.0,
            "x_mean": 1.5,
            "y_mean": 3.0,
            "has_sigma": False,
        },
    )
    module.render_statistics()


@pytest.mark.gui
@pytest.mark.unit
def test_data_loading_error_paths(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.1_Data_Loading")
    monkeypatch.setattr(module, "st", streamlit_stub)

    state = SimpleNamespace(xdata=None, ydata=None, sigma=None)
    monkeypatch.setattr(module, "initialize_state", lambda: state)
    streamlit_stub.session_state.column_assignments = {"x": 0, "y": 1}

    monkeypatch.setattr(
        module,
        "load_from_file",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")),
    )
    module.handle_file_upload(_Upload("x,y\n1,2"), "auto")
    assert any(
        "Error loading file" in (msg or "") for _, msg in streamlit_stub._messages
    )

    streamlit_stub._messages.clear()
    module.handle_clipboard_paste("", has_header=True)
    assert ("warning", "Please paste some data first") in streamlit_stub._messages

    streamlit_stub._messages.clear()
    monkeypatch.setattr(
        module,
        "load_from_clipboard",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    module.handle_clipboard_paste("x,y\n1,2", has_header=True)
    assert any(
        "Error parsing clipboard data" in (msg or "")
        for _, msg in streamlit_stub._messages
    )


@pytest.mark.gui
@pytest.mark.unit
def test_model_selection_render_paths(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.2_Model_Selection")
    monkeypatch.setattr(module, "st", streamlit_stub)

    state = SimpleNamespace(
        model_type="builtin",
        model_name="linear",
        polynomial_degree=2,
        custom_code="",
        custom_function_name="model",
    )
    monkeypatch.setattr(module, "initialize_state", lambda: state)

    streamlit_stub._radio_value = "Polynomial"
    module.render_model_type_selector()
    assert state.model_type == "polynomial"

    monkeypatch.setattr(
        module, "list_builtin_models", lambda: [{"name": "linear", "n_params": 2}]
    )
    monkeypatch.setattr(module, "load_builtin_model", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "render_model_preview", lambda *_a, **_k: None)
    streamlit_stub._selectbox_values = ["linear"]
    module.render_builtin_model_selector()
    assert streamlit_stub.session_state.model_loaded is True

    streamlit_stub._file_uploader_value = _Upload("def model(x):\n    return x")
    monkeypatch.setattr(module, "validate_code_syntax", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(module, "list_functions_in_module", lambda *_a, **_k: ["model"])
    monkeypatch.setattr(module, "load_custom_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        module,
        "get_model_info",
        lambda *_a, **_k: {"param_names": ["a"], "param_count": 1},
    )
    streamlit_stub._selectbox_values = ["model"]
    streamlit_stub._button_values = [True]
    module.render_custom_file_upload(state)
    assert streamlit_stub.session_state.model_loaded is True


@pytest.mark.gui
@pytest.mark.unit
def test_model_selection_error_paths(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.2_Model_Selection")
    monkeypatch.setattr(module, "st", streamlit_stub)

    state = SimpleNamespace(
        model_type="builtin",
        model_name="linear",
        polynomial_degree=2,
        custom_code="def model(x):\n    return x",
        custom_function_name="model",
    )
    monkeypatch.setattr(module, "initialize_state", lambda: state)

    monkeypatch.setattr(
        module, "list_builtin_models", lambda: [{"name": "linear", "n_params": 2}]
    )
    monkeypatch.setattr(
        module,
        "load_builtin_model",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")),
    )
    streamlit_stub._selectbox_values = ["linear"]
    module.render_builtin_model_selector()
    assert streamlit_stub.session_state.model_loaded is False
    assert any(
        "Error loading model" in (msg or "") for _, msg in streamlit_stub._messages
    )

    streamlit_stub._messages.clear()
    monkeypatch.setattr(
        module,
        "load_polynomial_model",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    module.render_polynomial_selector()
    assert streamlit_stub.session_state.model_loaded is False
    assert any(
        "Error generating polynomial" in (msg or "")
        for _, msg in streamlit_stub._messages
    )

    streamlit_stub._messages.clear()
    monkeypatch.setattr(
        module, "validate_code_syntax", lambda *_a, **_k: (False, "syntax")
    )
    streamlit_stub._text_area_value = "def model(x): return x"
    module.render_custom_code_editor(state)
    assert any("Syntax error" in (msg or "") for _, msg in streamlit_stub._messages)

    streamlit_stub._messages.clear()
    monkeypatch.setattr(module, "validate_code_syntax", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(module, "list_functions_in_module", lambda *_a, **_k: [])
    module.render_custom_code_editor(state)
    assert ("warning", "No functions found in code") in streamlit_stub._messages

    streamlit_stub._messages.clear()
    monkeypatch.setattr(module, "list_functions_in_module", lambda *_a, **_k: ["model"])
    monkeypatch.setattr(module, "load_custom_model", lambda *_a, **_k: None)
    streamlit_stub._selectbox_values = ["model"]
    streamlit_stub._button_values = [True]
    module.render_custom_code_editor(state)
    assert ("error", "Failed to load model") in streamlit_stub._messages


@pytest.mark.gui
@pytest.mark.unit
def test_fitting_options_render_helpers(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    monkeypatch.setattr(module, "st", streamlit_stub)
    from nlsq.gui.state import SessionState

    state = SessionState()
    monkeypatch.setattr(module, "get_session_state", lambda: state)

    streamlit_stub._radio_value = "Advanced"
    assert module.render_mode_toggle() == "advanced"

    state.enable_multistart = True
    module.render_preset_details(state)
    assert any("Enabled" in (msg or "") for _, msg in streamlit_stub._messages)


@pytest.mark.gui
@pytest.mark.unit
def test_results_and_export_render_valid(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = importlib.import_module("nlsq.gui.pages.4_Results")
    export = importlib.import_module("nlsq.gui.pages.5_Export")
    monkeypatch.setattr(results, "st", streamlit_stub)
    monkeypatch.setattr(export, "st", streamlit_stub)

    class Result:
        popt = np.array([1.0, 2.0])
        pcov = np.eye(2)
        residuals = np.array([0.1, -0.1])

    state = SimpleNamespace(fit_result=Result(), xdata=np.array([1.0, 2.0]))
    monkeypatch.setattr(results, "initialize_state", lambda: state)
    monkeypatch.setattr(export, "initialize_state", lambda: state)
    streamlit_stub.session_state.current_model = object()

    monkeypatch.setattr(results, "render_fit_plot", lambda *_a, **_k: None)
    monkeypatch.setattr(results, "render_residuals_plot", lambda *_a, **_k: None)
    monkeypatch.setattr(results, "render_residuals_histogram", lambda *_a, **_k: None)
    monkeypatch.setattr(
        results, "get_param_names_from_model", lambda *_a, **_k: ["a", "b"]
    )
    monkeypatch.setattr(
        results, "format_parameter_table", lambda *_a, **_k: pd.DataFrame({"a": [1]})
    )
    monkeypatch.setattr(
        results,
        "format_statistics",
        lambda *_a, **_k: {
            "r_squared": "0.9",
            "adj_r_squared": "0.9",
            "rmse": "0.1",
            "mae": "0.1",
            "aic": "1",
            "bic": "1",
        },
    )
    monkeypatch.setattr(
        results,
        "format_convergence_info",
        lambda *_a, **_k: {
            "success_str": "Yes",
            "message": "ok",
            "nfev_str": "5",
            "cost_str": "0.1",
        },
    )

    streamlit_stub._checkbox_values = [True, True, True, True, True]
    results.render_visualizations_section()
    results.render_export_section()

    monkeypatch.setattr(export, "create_session_bundle", lambda *_a, **_k: b"zip")
    monkeypatch.setattr(export, "export_json", lambda *_a, **_k: "{}")
    monkeypatch.setattr(export, "export_csv", lambda *_a, **_k: "a,b")
    monkeypatch.setattr(export, "generate_fit_script", lambda *_a, **_k: "print('hi')")
    monkeypatch.setattr(
        export, "get_param_names_from_model", lambda *_a, **_k: ["a", "b"]
    )
    export.render_session_bundle_section()
    export.render_individual_exports_section()
    export.render_python_code_section()


@pytest.mark.gui
@pytest.mark.unit
def test_results_and_export_error_paths(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = importlib.import_module("nlsq.gui.pages.4_Results")
    export = importlib.import_module("nlsq.gui.pages.5_Export")
    monkeypatch.setattr(results, "st", streamlit_stub)
    monkeypatch.setattr(export, "st", streamlit_stub)

    class Result:
        popt = np.array([1.0])
        pcov = np.eye(1)
        residuals = np.array([0.1])

    state = SimpleNamespace(fit_result=Result(), xdata=np.array([1.0]))
    monkeypatch.setattr(results, "initialize_state", lambda: state)
    monkeypatch.setattr(export, "initialize_state", lambda: state)
    streamlit_stub.session_state.current_model = object()

    monkeypatch.setattr(
        results,
        "render_fit_plot",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")),
    )
    monkeypatch.setattr(
        results,
        "render_residuals_plot",
        lambda *_a, **_k: (_ for _ in ()).throw(AttributeError("bad")),
    )
    monkeypatch.setattr(
        results,
        "render_residuals_histogram",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")),
    )
    streamlit_stub._checkbox_values = [True, True, True, True, True]
    results.render_visualizations_section()
    assert any(
        "Cannot display fit plot" in (msg or "") for _, msg in streamlit_stub._messages
    )
    assert any(
        "Cannot display residuals plot" in (msg or "")
        for _, msg in streamlit_stub._messages
    )

    streamlit_stub._messages.clear()
    monkeypatch.setattr(
        export,
        "create_session_bundle",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    export.render_session_bundle_section()
    assert any(
        "Failed to create session bundle" in (msg or "")
        for _, msg in streamlit_stub._messages
    )

    streamlit_stub._messages.clear()
    monkeypatch.setattr(
        export,
        "export_json",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")),
    )
    monkeypatch.setattr(
        export,
        "export_csv",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")),
    )
    export.render_individual_exports_section()
    assert any(
        "Failed to generate JSON" in (msg or "") for _, msg in streamlit_stub._messages
    )
    assert any(
        "Failed to generate CSV" in (msg or "") for _, msg in streamlit_stub._messages
    )

    streamlit_stub._messages.clear()
    monkeypatch.setattr(
        export,
        "generate_fit_script",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    export.render_python_code_section()
    assert any(
        "Failed to generate Python script" in (msg or "")
        for _, msg in streamlit_stub._messages
    )


@pytest.mark.gui
@pytest.mark.unit
def test_model_selection_custom_code_paths(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.2_Model_Selection")
    monkeypatch.setattr(module, "st", streamlit_stub)

    state = SimpleNamespace(
        model_type="custom", custom_code="", custom_function_name=""
    )
    monkeypatch.setattr(module, "get_session_state", lambda: state)
    streamlit_stub.session_state.custom_code = "x = 1"
    streamlit_stub._text_area_value = "x = 1"
    streamlit_stub.session_state.custom_function_name = ""

    monkeypatch.setattr(module, "validate_code_syntax", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(module, "list_functions_in_module", lambda *_a, **_k: [])

    module.render_custom_code_editor(state)
    assert _has_message(streamlit_stub, "warning", "No functions found")

    streamlit_stub._messages.clear()
    streamlit_stub._text_area_value = "def model(x): return x"
    streamlit_stub._selectbox_values = ["model"]
    streamlit_stub._button_values = [True]

    monkeypatch.setattr(module, "list_functions_in_module", lambda *_a, **_k: ["model"])
    monkeypatch.setattr(module, "load_custom_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        module,
        "get_model_info",
        lambda *_a, **_k: {"param_names": ["a", "b"], "param_count": 2},
    )
    monkeypatch.setattr(module, "format_parameter_list", lambda names: ", ".join(names))

    module.render_custom_code_editor(state)
    assert streamlit_stub.session_state.model_loaded is True
    assert _has_message(streamlit_stub, "success", "loaded successfully")


@pytest.mark.gui
@pytest.mark.unit
def test_model_selection_custom_file_upload_error(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.2_Model_Selection")
    monkeypatch.setattr(module, "st", streamlit_stub)

    class _UploadModel:
        def __init__(self) -> None:
            self.name = "model.py"

        def read(self) -> bytes:
            return b"def model(x): return x"

    state = SimpleNamespace()
    streamlit_stub._file_uploader_value = _UploadModel()
    monkeypatch.setattr(
        module, "validate_code_syntax", lambda *_a, **_k: (False, "bad")
    )

    module.render_custom_file_upload(state)
    assert _has_message(streamlit_stub, "error", "syntax errors")


@pytest.mark.gui
@pytest.mark.unit
def test_model_selection_summary_sidebar_empty(
    streamlit_stub: types.ModuleType,
) -> None:
    module = importlib.import_module("nlsq.gui.pages.2_Model_Selection")
    module.st = streamlit_stub

    streamlit_stub.session_state.model_loaded = False
    streamlit_stub._messages.clear()
    module.render_model_summary_sidebar()
    assert _has_message(streamlit_stub, "info", "No model selected")


@pytest.mark.gui
@pytest.mark.unit
def test_fitting_options_sidebar_status_branches(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.3_Fitting_Options")
    monkeypatch.setattr(module, "st", streamlit_stub)
    monkeypatch.setattr(
        module, "get_param_names_from_model", lambda *_a, **_k: ["a", "b"]
    )

    class _Result:
        success = True

    state = SimpleNamespace(
        mode="guided",
        preset="fast",
        gtol=1e-6,
        enable_multistart=True,
        n_starts=3,
        xdata=[1.0, 2.0],
        ydata=[2.0, 3.0],
        fit_running=False,
        fit_result=_Result(),
    )
    streamlit_stub.session_state.current_model = object()
    streamlit_stub._messages.clear()
    module.render_sidebar_status(state)

    assert _has_message(streamlit_stub, "success", "Multi-start")
    assert _has_message(streamlit_stub, "success", "Last fit: Success")

    streamlit_stub._messages.clear()
    state.enable_multistart = False
    state.fit_running = True
    state.fit_result = None
    streamlit_stub.session_state.current_model = None
    state.xdata = None
    state.ydata = None
    module.render_sidebar_status(state)
    assert _has_message(streamlit_stub, "info", "Multi-start: Off")
    assert _has_message(streamlit_stub, "warning", "Fit running")
    assert _has_message(streamlit_stub, "warning", "No model selected")
    assert _has_message(streamlit_stub, "warning", "No data loaded")


@pytest.mark.gui
@pytest.mark.unit
def test_results_visualization_warning_paths(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.pages.4_Results")
    monkeypatch.setattr(module, "st", streamlit_stub)

    result = SimpleNamespace(residuals=np.array([1.0, -1.0]))
    monkeypatch.setattr(module, "get_fit_result", lambda: result)
    monkeypatch.setattr(module, "get_session_state", lambda: SimpleNamespace())

    def _raise_value_error(*_a: object, **_k: object) -> None:
        raise ValueError("no plot")

    monkeypatch.setattr(module, "render_fit_plot", _raise_value_error)
    monkeypatch.setattr(module, "render_residuals_plot", _raise_value_error)
    monkeypatch.setattr(module, "render_residuals_histogram", _raise_value_error)

    module.render_visualizations_section()
    assert _has_message(streamlit_stub, "warning", "Cannot display fit plot")
    assert _has_message(streamlit_stub, "warning", "Cannot display residuals plot")
    assert _has_message(streamlit_stub, "warning", "Cannot display histogram")
