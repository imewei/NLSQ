"""Results-page fixes found by the Codex review of results.py (2026-11-13)."""

from __future__ import annotations

import numpy as np
import pytest

from nlsq.gui_qt.pages.results import ResultsPage
from nlsq.result import CurveFitResult

N = 21


def _setup(app_state):
    x = np.linspace(1, 5, N)
    app_state.set_data(x, 2 * x + 0.5, sigma=np.full(N, 0.1))
    app_state.set_model("builtin", {"name": "linear"})
    app_state.state.absolute_sigma = False
    return CurveFitResult(
        popt=np.array([2.0, 0.5]), pcov=np.eye(2) * 1e-4, success=True
    )


def _page(app_state, qtbot):
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    return page


@pytest.mark.serial
def test_status_reports_the_fit_time_sigma_convention(app_state, qtbot):
    result = _setup(app_state)
    app_state.set_fit_result(result, app_state.state.snapshot_for_fit())
    app_state.state.absolute_sigma = True  # edited after the fit ran

    page = _page(app_state, qtbot)
    page.update_results(result)

    text = page._status_label.text()
    assert "sigma treated as relative" in text
    assert "sigma treated as absolute" not in text


@pytest.mark.serial
def test_failed_model_evaluation_does_not_keep_the_previous_curve(app_state, qtbot):
    result = _setup(app_state)
    app_state.state.model_func = lambda x, a, b: a * x + b
    page = _page(app_state, qtbot)
    page.update_results(result)
    assert page._fit_plot._fit_curve is not None  # first fit drew its curve

    def broken(x, *p):
        raise RuntimeError("model cannot be evaluated")

    app_state.state.model_func = broken
    page.update_results(result)

    assert page._fit_plot._fit_curve is None
    assert page._fit_plot._conf_band_upper is None
    assert "model could not be evaluated" in page._status_label.text()


@pytest.mark.serial
def test_dense_grid_failure_keeps_results_and_draws_curve_at_data(app_state, qtbot):
    result = _setup(app_state)

    def tied_to_data_length(x, a, b):
        if len(x) != N:
            raise ValueError("model only accepts the original data length")
        return a * x + b

    app_state.state.model_func = tied_to_data_length
    page = _page(app_state, qtbot)
    page.update_results(result)

    assert page._has_results
    assert "Could not display" not in page._status_label.text()
    assert len(page._fit_plot._fit_x) == N
    assert np.all(np.diff(page._fit_plot._fit_x) >= 0)  # sorted: no zig-zag curve


def test_confidence_band_is_not_zero_for_float32_parameters():
    x = np.linspace(1.0, 5.0, 11)
    popt = np.array([1.0], dtype=np.float32)
    pcov = np.array([[0.04]])

    lower, upper = ResultsPage._compute_confidence_bands(
        object(), x, popt, pcov, lambda x, a: a * x
    )

    assert np.all(upper - lower > 0)
