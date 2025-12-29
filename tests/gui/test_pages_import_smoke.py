"""Lightweight GUI page import tests with a Streamlit stub."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def _install_streamlit_stub() -> types.ModuleType:
    module = types.ModuleType("streamlit")

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    def _getattr(name: str) -> object:
        if name == "sidebar":
            return module
        return _noop

    module.session_state = {}
    module.set_page_config = _noop
    module.__getattr__ = _getattr  # type: ignore[attr-defined]
    return module


@pytest.mark.gui
def test_gui_pages_import_with_stubbed_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUI page modules should import without real Streamlit present."""
    stub = _install_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    page_modules = [
        "nlsq.gui.pages.1_Data_Loading",
        "nlsq.gui.pages.2_Model_Selection",
        "nlsq.gui.pages.3_Fitting_Options",
        "nlsq.gui.pages.4_Results",
        "nlsq.gui.pages.5_Export",
    ]

    for module_name in page_modules:
        importlib.import_module(module_name)
