"""Tests for ExportPage._generate_json() -- NaN/Infinity JSON compliance."""

from __future__ import annotations

import json

import numpy as np
import pytest

# GUI tests crash xdist workers on headless CI (see test_gui_startup.py) --
# run on a single dedicated worker to avoid taking down the shared pool.
pytestmark = pytest.mark.serial


class FakeFitResult:
    def __init__(self, x):
        self.x = np.asarray(x)
        self.pcov = None
        self.success = True
        self.nfev = 1


def _nan_model(x, a):
    return np.full_like(x, np.nan)


class TestGenerateJsonNonFiniteStatistics:
    """_generate_json() must normalize non-finite r_squared/rmse to `null`
    instead of emitting non-standard `NaN`/`Infinity` JSON tokens, which a
    diverged fit (model producing NaN residuals) would otherwise trigger."""

    def test_diverged_fit_emits_null_not_nan_token(self, app_state, qtbot):
        from nlsq.gui_qt.pages.export import ExportPage

        app_state.set_data(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        app_state.set_model("custom", model_func=_nan_model)
        app_state.set_fit_result(FakeFitResult([1.0]))

        page = ExportPage(app_state)
        qtbot.addWidget(page)

        json_str = page._generate_json()

        assert "NaN" not in json_str
        assert "Infinity" not in json_str
        data = json.loads(json_str)
        assert data["statistics"]["r_squared"] is None
        assert data["statistics"]["rmse"] is None

    def test_normal_fit_emits_finite_statistics(self, app_state, qtbot):
        from nlsq.gui_qt.pages.export import ExportPage

        xdata = np.array([1.0, 2.0, 3.0, 4.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0])
        app_state.set_data(xdata, ydata)
        app_state.set_model("custom", model_func=lambda x, a: a * x)
        app_state.set_fit_result(FakeFitResult([2.0]))

        page = ExportPage(app_state)
        qtbot.addWidget(page)

        data = json.loads(page._generate_json())

        assert data["statistics"]["r_squared"] == 1.0
        assert data["statistics"]["rmse"] == 0.0
