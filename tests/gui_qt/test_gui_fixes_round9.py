"""Regression tests: guided absolute_sigma, --theme, session-restore previews, stale results."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.serial


def _poly_state(app_state):
    from nlsq.gui_qt.adapters.model_adapter import get_model

    x = np.linspace(0, 1, 50)
    app_state.set_data(x, 2 * x + 1, np.full(50, 0.1))
    app_state.set_model(
        "polynomial", {"degree": 1}, get_model("polynomial", {"degree": 1})
    )


def test_guided_run_fit_keeps_absolute_sigma(app_state, qtbot, monkeypatch):
    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    _poly_state(app_state)
    page._mode_button_group.button(1).click()
    page._advanced_options._abs_sigma_check.setChecked(True)
    page._mode_button_group.button(0).click()
    # stop right after the option sync; the fit itself is irrelevant here
    monkeypatch.setattr(page, "_sync_parameters_to_state", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        page.run_fit()
    assert app_state.state.absolute_sigma is True


def test_cli_theme_decides_initial_theme(qtbot, monkeypatch):
    from nlsq.gui_qt import _parse_args
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.main_window import MainWindow

    assert _parse_args([]).theme is None
    assert _parse_args(["--theme", "light"]).theme == "light"

    window = MainWindow(AppState())
    qtbot.addWidget(window)
    window.set_theme("light")  # what run_desktop does for --theme light
    assert window._theme_manager.current_theme.name == "light"
    assert window._app_state.theme == "light"
    assert not window._theme_toggle.isChecked()


def test_recovered_builtin_and_polynomial_refresh_preview(qtbot):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.model_selection import ModelSelectionPage

    state = AppState()
    page = ModelSelectionPage(state)
    qtbot.addWidget(page)
    before = page._param_list.text()

    state.state.model_type = "builtin"
    state.state.model_config = {"name": "gaussian"}
    page._restore_session_controls()
    assert page._param_list.text() != before
    assert "mu" in page._param_list.text() or "sigma" in page._param_list.text()
    assert page._builtin_desc.text().startswith("y = A * exp")

    state.state.model_type = "polynomial"
    state.state.model_config = {"degree": 4}
    page._restore_session_controls()
    assert "c4" in page._param_list.text()
    assert page._degree_value.text() == "4"
    assert page._current_config == {"degree": 4}


def test_results_cleared_when_display_fails(app_state, qtbot, monkeypatch):
    from types import SimpleNamespace

    from nlsq.gui_qt.pages.results import ResultsPage

    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    _poly_state(app_state)
    page._param_results.set_results(
        names=["c0", "c1"], values=[1.0, 2.0], uncertainties=[0.1, 0.1]
    )
    assert page._param_results._table.rowCount() == 2

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(page, "_update_statistics", boom)
    page.update_results(SimpleNamespace(x=np.array([1.0, 2.0]), success=True, nfev=1))
    assert page._param_results._table.rowCount() == 0


def test_results_param_table_cleared_without_popt(app_state, qtbot):
    from types import SimpleNamespace

    from nlsq.gui_qt.pages.results import ResultsPage

    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    _poly_state(app_state)
    page._param_results.set_results(
        names=["c0", "c1"], values=[1.0, 2.0], uncertainties=[0.1, 0.1]
    )
    page.update_results(SimpleNamespace(success=False, nfev=0))
    assert page._param_results._table.rowCount() == 0
