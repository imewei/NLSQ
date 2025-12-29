"""Fast tests for fit statistics component helpers and rendering."""

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
    module._metrics: list[tuple[str, str]] = []

    def _record(kind: str):
        def _fn(message: str | None = None, *_a: object, **_k: object) -> None:
            module._messages.append((kind, message))

        return _fn

    module.metric = lambda label, value, **_k: module._metrics.append((label, value))
    module.markdown = lambda *_a, **_k: None
    module.divider = lambda *_a, **_k: None
    module.success = _record("success")
    module.warning = _record("warning")
    module.info = _record("info")
    module.error = _record("error")
    module.caption = _record("caption")
    module.columns = lambda n=1: tuple(_ContextManager(module) for _ in range(n))
    module.__getattr__ = lambda _name: (lambda *_a, **_k: None)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    stub = _install_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    created_pkg = "nlsq.gui.components.fit_statistics" not in sys.modules

    def restore() -> None:
        if created_pkg:
            sys.modules.pop("nlsq.gui.components.fit_statistics", None)

    request.addfinalizer(restore)
    return stub


@pytest.mark.gui
@pytest.mark.unit
def test_format_statistics_handles_missing_fields() -> None:
    module = importlib.import_module("nlsq.gui.components.fit_statistics")
    stats = module.format_statistics(SimpleNamespace())
    assert stats["r_squared"] == "N/A"
    assert stats["aic"] == "N/A"


@pytest.mark.gui
@pytest.mark.unit
def test_fit_quality_label_thresholds() -> None:
    module = importlib.import_module("nlsq.gui.components.fit_statistics")
    label, color = module.get_fit_quality_label(0.97)
    assert label == "Very Good"
    assert color == "green"


@pytest.mark.gui
@pytest.mark.unit
def test_render_fit_statistics_and_summary(streamlit_stub: types.ModuleType) -> None:
    module = importlib.import_module("nlsq.gui.components.fit_statistics")

    result = SimpleNamespace(
        r_squared=0.96,
        adj_r_squared=0.95,
        rmse=0.1,
        mae=0.05,
        aic=12.34,
        bic=23.45,
        success=True,
        nfev=10,
        cost=0.01,
        message="ok",
        optimality=1e-6,
    )

    module.render_fit_statistics(
        result, container=streamlit_stub, show_convergence=True
    )
    assert any("Fit Quality" in (msg or "") for kind, msg in streamlit_stub._messages)
    assert ("success", "Fit Quality: Very Good") in streamlit_stub._messages

    streamlit_stub._messages.clear()
    module.render_statistics_summary(result, container=streamlit_stub)
    assert ("success", "Converged") in streamlit_stub._messages
