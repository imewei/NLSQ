"""Fast render-path tests for param_config using Streamlit stubs."""

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
    module._text_input_values: list[str] = []
    module._selectbox_values: list[str] = []
    module._metrics: list[tuple[str, str]] = []

    def _record(kind: str):
        def _fn(message: str | None = None, *_a: object, **_k: object) -> None:
            module._messages.append((kind, message))

        return _fn

    def _pop(values: list[str], default: str) -> str:
        return values.pop(0) if values else default

    module.subheader = lambda *_a, **_k: None
    module.markdown = lambda *_a, **_k: None
    module.divider = lambda *_a, **_k: None
    module.caption = _record("caption")
    module.info = _record("info")
    module.success = _record("success")
    module.warning = _record("warning")
    module.error = _record("error")
    module.metric = lambda label, value, **_k: module._metrics.append((label, value))
    module.columns = lambda n=1: tuple(
        _ContextManager(module) for _ in range(n if isinstance(n, int) else len(n))
    )
    module.container = lambda: _ContextManager(module)
    module.selectbox = lambda _label, options, **_k: _pop(
        module._selectbox_values, options[0]
    )
    module.text_input = lambda _label, **_k: _pop(
        module._text_input_values, _k.get("value", "")
    )
    module.sidebar = module
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    stub = _install_streamlit_stub()
    module_name = "nlsq.gui.components.param_config"
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
def test_render_param_config_warns_without_params(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.components.param_config")
    monkeypatch.setattr(module, "st", streamlit_stub)
    monkeypatch.setattr(module, "get_param_names_from_model", lambda *_a, **_k: [])

    from nlsq.gui.state import SessionState

    state = SessionState()
    module.render_param_config(state, model=object())
    assert (
        "warning",
        "Could not determine model parameters",
    ) in streamlit_stub._messages


@pytest.mark.gui
@pytest.mark.unit
def test_render_single_param_row_and_summary(
    streamlit_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("nlsq.gui.components.param_config")
    monkeypatch.setattr(module, "st", streamlit_stub)

    from nlsq.gui.state import SessionState

    state = SessionState()
    state.p0 = [None]
    state.bounds = ([None], [None])
    state.transforms = {}
    state.auto_p0 = True

    streamlit_stub._text_input_values = [
        "",  # p0 input (auto)
        "1.0",  # lower bound
        "0.0",  # upper bound (invalid lower>upper)
    ]
    streamlit_stub._selectbox_values = ["log"]

    module.render_single_param_row(
        param_name="a",
        index=0,
        state=state,
        estimated_p0=0.5,
        use_auto=True,
    )

    assert state.p0[0] is None
    assert state.transforms["a"] == "log"
    assert any("Lower bound" in (msg or "") for _, msg in streamlit_stub._messages)

    streamlit_stub._messages.clear()
    state.auto_p0 = False
    state.p0 = [1.0]
    module.render_param_summary(state, ["a"])
    assert streamlit_stub._metrics
