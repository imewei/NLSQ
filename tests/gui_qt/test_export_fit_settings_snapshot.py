"""Exports describe the fit that produced the displayed result (CG1), and the
CLI-schema flat ``fitting.absolute_sigma`` imports correctly (CG3)."""

import json

import numpy as np
import pytest
import yaml

from nlsq.gui_qt.adapters.config_adapter import export_yaml_config, load_yaml_config
from nlsq.gui_qt.pages.export import ExportPage
from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage
from nlsq.result import CurveFitResult


def _fitted_state(app_state):
    x = np.linspace(1, 5, 21)
    app_state.set_data(x, 2 * x + 0.5, sigma=np.full(21, 0.1))
    app_state.set_model("builtin", {"name": "linear"})
    app_state.state.absolute_sigma = False
    return CurveFitResult(popt=np.array([2.0, 0.5]), pcov=np.eye(2), success=True)


@pytest.mark.serial
def test_export_uses_settings_snapshot_not_live_state(app_state, qtbot):
    result = _fitted_state(app_state)
    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    page._fit_settings = app_state.state.snapshot_for_fit()
    page._on_fit_finished(result)

    # Edit options after the fit, as the widgets' slots do
    app_state.state.absolute_sigma = True
    app_state.state.ftol = 1e-3
    app_state.state.p0 = [9.0, 9.0]

    export = ExportPage(app_state)
    qtbot.addWidget(export)
    assert json.loads(export._generate_json())["statistics"]["absolute_sigma"] is False
    code = export.generate_python_code()
    assert "absolute_sigma=True" not in code
    assert "ftol=0.001" not in code
    assert "9.0" not in code
    assert (
        yaml.safe_load(export._generate_config_yaml())["fitting"]["objective"][
            "absolute_sigma"
        ]
        is False
    )


@pytest.mark.serial
def test_export_falls_back_to_live_state_without_snapshot(app_state, qtbot):
    app_state.set_fit_result(_fitted_state(app_state))
    app_state.state.absolute_sigma = True
    export = ExportPage(app_state)
    qtbot.addWidget(export)
    assert json.loads(export._generate_json())["statistics"]["absolute_sigma"] is True


@pytest.mark.serial
def test_snapshot_dropped_when_result_cleared(app_state):
    result = _fitted_state(app_state)
    app_state.set_fit_result(result, app_state.state.snapshot_for_fit())
    assert app_state.fit_settings is not app_state.state
    app_state.set_model("builtin", {"name": "linear"})  # clears the fit result
    assert app_state.fit_settings is app_state.state


_BASE = "model: {type: builtin, name: linear}\nfitting:\n  p0: [1.0, 0.0]\n"


def test_flat_absolute_sigma_is_imported():
    assert load_yaml_config(_BASE + "  absolute_sigma: true\n").absolute_sigma is True
    state = load_yaml_config(_BASE + "  absolute_sigma: true\n")
    assert yaml.safe_load(export_yaml_config(state))  # round-trips without error


@pytest.mark.parametrize(
    "extra",
    [
        "  absolute_sigma: 'false'\n",
        "  objective: {absolute_sigma: 'false'}\n",
        "  check_finite: 'no'\n",
    ],
)
def test_non_bool_options_rejected(extra):
    with pytest.raises(ValueError, match=r"absolute_sigma|check_finite"):
        load_yaml_config(_BASE + extra)


def test_flat_check_finite_bool_accepted():
    load_yaml_config(_BASE + "  check_finite: false\n")
