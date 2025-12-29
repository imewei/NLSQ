"""Fast unit tests for GUI app navigation and sidebar helpers."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


class _SessionState(dict):
    """Dict-backed session state with attribute access."""

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
    module._rerun_called = False

    def _record(kind: str):
        def _fn(message: str | None = None, *_args: object, **_kwargs: object) -> None:
            module._messages.append((kind, message))

        return _fn

    module.set_page_config = lambda **_kwargs: None
    module.success = _record("success")
    module.warning = _record("warning")
    module.info = _record("info")
    module.caption = _record("caption")
    module.subheader = lambda *_a, **_k: None
    module.title = lambda *_a, **_k: None
    module.markdown = lambda *_a, **_k: None
    module.divider = lambda *_a, **_k: None
    module.sidebar = module
    module.toggle = lambda *_a, **_k: False
    module.rerun = lambda *_a, **_k: setattr(module, "_rerun_called", True)
    module.expander = lambda *_a, **_k: _ContextManager(module)
    module.columns = lambda n=1: tuple(_ContextManager(module) for _ in range(n))
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def app_with_streamlit_stub(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[types.ModuleType, types.ModuleType]:
    """Provide the GUI app imported with a Streamlit stub and clean up afterwards."""

    stub = _install_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    created_gui_pkg = "nlsq.gui" not in sys.modules
    original_gui = sys.modules.get("nlsq.gui")
    original_app = sys.modules.pop("nlsq.gui.app", None)
    original_gui_app_attr = getattr(original_gui, "app", None) if original_gui else None

    app = importlib.import_module("nlsq.gui.app")
    monkeypatch.setattr(app, "st", stub, raising=False)

    def restore_modules() -> None:
        if original_app is not None:
            sys.modules["nlsq.gui.app"] = original_app
        else:
            sys.modules.pop("nlsq.gui.app", None)

        if created_gui_pkg:
            sys.modules.pop("nlsq.gui", None)
        elif original_gui is not None:
            if original_gui_app_attr is not None:
                original_gui.app = original_gui_app_attr
            elif hasattr(original_gui, "app"):
                delattr(original_gui, "app")

    request.addfinalizer(restore_modules)
    return app, stub


@pytest.mark.gui
def test_app_status_and_navigation_messages(
    app_with_streamlit_stub: tuple[types.ModuleType, types.ModuleType],
) -> None:
    app, _stub = app_with_streamlit_stub
    from nlsq.gui.state import SessionState

    state = SessionState()
    status = app.get_page_status(state)
    assert status["data"]["loaded"] is False
    assert status["model"]["selected"] is True

    msg = app.get_navigation_message("model", state)
    assert "Load data first" in msg

    msg = app.get_navigation_message("fitting", state)
    assert "Load data first" in msg

    state.xdata = np.array([1.0])
    state.ydata = np.array([2.0])
    state.model_type = "custom"
    state.custom_code = ""
    msg = app.get_navigation_message("fitting", state)
    assert "Select a model first" in msg

    state.model_type = "builtin"
    state.fit_running = True
    step_msg = app.get_workflow_step_message(state)
    assert "Fit is running" in step_msg


@pytest.mark.gui
def test_sidebar_status_and_theme_toggle(
    app_with_streamlit_stub: tuple[types.ModuleType, types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, stub = app_with_streamlit_stub
    from nlsq.gui.state import SessionState

    state = SessionState()
    state.xdata = np.array([1.0, 2.0])
    state.ydata = np.array([2.0, 4.0])
    state.data_file_name = "data.csv"
    state.model_type = "polynomial"
    state.fit_running = True

    app.render_sidebar_status(state)
    assert any("Data:" in (msg or "") for kind, msg in stub._messages)
    assert any("Model: Polynomial" in (msg or "") for kind, msg in stub._messages)
    assert ("warning", "Fit: Running...") in stub._messages

    stub._messages.clear()
    stub.toggle = lambda *_a, **_k: True
    monkeypatch.setattr(app, "get_current_theme", lambda: "light")
    set_theme_calls: list[str] = []
    monkeypatch.setattr(app, "set_theme", lambda theme: set_theme_calls.append(theme))

    app.render_theme_toggle()
    assert set_theme_calls == ["dark"]
    assert stub._rerun_called is True


@pytest.mark.gui
def test_check_page_access_shows_warning(
    app_with_streamlit_stub: tuple[types.ModuleType, types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, stub = app_with_streamlit_stub
    from nlsq.gui.state import SessionState

    state = SessionState()
    monkeypatch.setattr(app, "get_session_state", lambda: state)

    allowed = app.check_page_access_and_show_message("model")
    assert allowed is False
    assert ("warning", "Load data first to select a model.") in stub._messages
