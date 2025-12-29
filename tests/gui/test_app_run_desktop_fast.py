"""Fast tests for gui app helpers and run_desktop branches."""

from __future__ import annotations

import importlib
import sys
import types

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
    module._set_page_config: list[dict[str, object]] = []

    def _record(kind: str):
        def _fn(message: str | None = None, *_a: object, **_k: object) -> None:
            module._messages.append((kind, message))

        return _fn

    module.set_page_config = lambda **kwargs: module._set_page_config.append(kwargs)
    module.sidebar = module
    module.info = _record("info")
    module.success = _record("success")
    module.warning = _record("warning")
    module.error = _record("error")
    module.markdown = lambda *_a, **_k: None
    module.subheader = lambda *_a, **_k: None
    module.caption = _record("caption")
    module.columns = lambda n=1: tuple(
        _ContextManager(module) for _ in range(n if isinstance(n, int) else len(n))
    )
    module.metric = lambda *_a, **_k: None
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    stub = _install_streamlit_stub()
    module_names = ("nlsq.gui.app",)
    cached_modules = {
        name: sys.modules[name] for name in module_names if name in sys.modules
    }
    cached_streamlit = sys.modules.get("streamlit")
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    for name in module_names:
        sys.modules.pop(name, None)

    def restore() -> None:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in cached_modules.items():
            sys.modules[name] = module
        if cached_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = cached_streamlit

    request.addfinalizer(restore)
    return stub


@pytest.mark.gui
@pytest.mark.unit
def test_app_helpers(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = importlib.import_module("nlsq.gui.app")
    monkeypatch.setattr(app, "st", streamlit_stub)

    from nlsq.gui.state import SessionState

    state = SessionState()
    state.xdata = None
    state.ydata = None
    state.model_type = ""

    status = app.get_page_status(state)
    assert status["data"]["loaded"] is False
    assert app.can_access_page("data", state) is True
    assert app.can_access_page("model", state) is False
    assert "Load data" in app.get_navigation_message("model", state)
    assert "Load data" in app.get_workflow_step_message(state)

    state.xdata = [1.0]
    state.ydata = [2.0]
    assert app.can_access_page("model", state) is True
    assert "Select a model" in app.get_workflow_step_message(state)

    state.model_type = "builtin"
    state.model_name = "linear"
    assert app.can_access_page("fitting", state) is True

    state.fit_result = object()
    assert app.can_access_page("results", state) is True
    assert "Fit complete" in app.get_workflow_step_message(state)


@pytest.mark.gui
@pytest.mark.unit
def test_configure_page_and_initialize_state(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = importlib.import_module("nlsq.gui.app")
    monkeypatch.setattr(app, "st", streamlit_stub)

    app.configure_page()
    assert streamlit_stub._set_page_config

    app.initialize_session_state()
    assert "nlsq_state" in streamlit_stub.session_state
    assert streamlit_stub.session_state.theme == "light"


@pytest.mark.gui
@pytest.mark.unit
def test_run_desktop_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("nlsq.gui.run_desktop")

    args = module.get_streamlit_args(1234)
    assert "--server.port" in args
    assert "1234" in args

    class DummyServer:
        def __init__(self, *_a, **_k):
            self.port = 1234
            self._running = False

        @property
        def url(self) -> str:
            return "http://localhost:1234"

        def start(self, *_a, **_k) -> bool:
            self._running = True
            return True

        def stop(self) -> None:
            self._running = False

        @property
        def is_running(self) -> bool:
            return False

    monkeypatch.setattr(module, "StreamlitServer", DummyServer)
    monkeypatch.setattr(module, "wait_for_server", lambda *_a, **_k: True)

    opened: dict[str, str] = {}
    monkeypatch.setitem(
        sys.modules,
        "webbrowser",
        types.SimpleNamespace(open=lambda url: opened.setdefault("url", url)),
    )
    monkeypatch.setattr(
        module, "time", types.SimpleNamespace(sleep=lambda *_a, **_k: None)
    )

    module.run_in_browser(port=1234)
    assert opened["url"] == "http://localhost:1234"

    dummy_webview = types.SimpleNamespace(
        create_window=lambda **_k: object(),
        start=lambda **_k: None,
    )
    monkeypatch.setitem(sys.modules, "webview", dummy_webview)
    monkeypatch.setitem(
        sys.modules,
        "nlsq.gui.desktop_config",
        types.SimpleNamespace(
            get_desktop_config=lambda: types.SimpleNamespace(
                icon_path=None, window_title="NLSQ"
            )
        ),
    )
    module.run_with_webview(width=800, height=600, debug=False)


@pytest.mark.gui
@pytest.mark.unit
def test_run_desktop_main_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("nlsq.gui.run_desktop")

    called = {"browser": 0, "webview": 0}

    def _run_browser(*_a, **_k):
        called["browser"] += 1

    def _run_webview(*_a, **_k):
        called["webview"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "run_in_browser", _run_browser)
    monkeypatch.setattr(module, "run_with_webview", _run_webview)
    monkeypatch.setattr(sys, "argv", ["nlsq-gui"])

    module.main()
    assert called["webview"] == 1
    assert called["browser"] == 1
