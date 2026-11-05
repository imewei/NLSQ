"""Regression tests: sidebar navigation, theme sync and Results page (GUI1/4/5/6/8/10)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.serial

CUSTOM_CODE = "import jax.numpy as jnp\ndef m(x, a, b):\n    return a*jnp.exp(-b*x)\n"


def _item_enabled(window, row: int) -> bool:
    from PySide6.QtCore import Qt

    return bool(window._nav_list.item(row).flags() & Qt.ItemFlag.ItemIsEnabled)


def _loaded(app_state, window):
    x = np.linspace(0, 5, 50)
    app_state.set_data(x, 2 * np.exp(-0.7 * x) + 0.1)
    return x


def _fake_fit(app_state):
    from nlsq.core.functions import exponential_decay
    from nlsq.core.minpack import fit

    state = app_state.state
    app_state.set_model("builtin", {"name": "exponential_decay"}, exponential_decay)
    res = fit(exponential_decay, state.xdata, state.ydata, p0=[1.0, 1.0, 0.0])
    app_state.set_fit_result(res)
    return res


def test_custom_model_unlocks_fitting_options(qapp, app_state, main_window):
    from nlsq.gui_qt.adapters.model_adapter import parse_custom_model_string

    _loaded(app_state, main_window)
    f, _ = parse_custom_model_string(CUSTOM_CODE, "m")
    app_state.set_model("custom", {"code": CUSTOM_CODE, "function": "m"}, f)
    assert _item_enabled(main_window, 2)


def test_clearing_result_does_not_jump_to_results(qapp, app_state, main_window):
    _loaded(app_state, main_window)
    _fake_fit(app_state)
    main_window._nav_list.setCurrentRow(2)
    app_state.set_fit_running(True)  # clears the old result
    app_state.set_fit_running(False)  # else closeEvent shows a modal "fit running" box
    assert main_window._nav_list.currentRow() == 2
    assert "completed" not in main_window._status_label.text()


def test_results_page_cleared_when_result_invalidated(qapp, app_state, main_window):
    _loaded(app_state, main_window)
    _fake_fit(app_state)
    rp = main_window._pages["results"]
    assert rp.has_results() and rp._export_btn.isEnabled()
    app_state.set_fit_running(True)
    app_state.set_fit_running(False)
    assert not rp.has_results()
    assert not rp._export_btn.isEnabled()
    assert rp._last_result is None


def test_shortcut_navigation_respects_guards(qapp, app_state, main_window):
    _loaded(app_state, main_window)
    main_window._nav_list.setCurrentRow(0)
    for _ in range(4):
        main_window._navigate_next()  # Results/Export need a fit result
    assert main_window._nav_list.currentRow() == main_window._page_stack.currentIndex()
    assert main_window._nav_list.currentRow() == 2
    main_window._nav_list.setCurrentRow(4)  # Export: no fit result
    assert main_window._nav_list.currentRow() != 4
    assert main_window._page_stack.currentIndex() != 4


def test_ctrl_t_syncs_checkbox_and_state(qapp, app_state, main_window):
    mw = main_window
    before = mw._theme_manager.current_theme.name
    mw._theme_manager.toggle()
    after = mw._theme_manager.current_theme.name
    assert after != before
    assert mw._theme_toggle.isChecked() == (after == "dark")
    assert app_state.theme == after


def test_confidence_band_large_parameters(qapp, app_state):
    from nlsq.core.functions import exponential_decay as m
    from nlsq.gui_qt.pages.results import ResultsPage

    rp = ResultsPage(app_state)
    x = np.linspace(0, 5, 50)
    popt = np.array([1e9, 0.7, 0.0])
    pcov = np.diag([1e14, 1e-4, 1.0])  # sd(a) = 1e7
    lo, hi = rp._compute_confidence_bands(x, popt, pcov, m, n_data=50)
    half = (hi[0] - lo[0]) / 2
    expected = 2.0106 * np.sqrt(1e14 + 1.0)
    assert half == pytest.approx(expected, rel=1e-2)
