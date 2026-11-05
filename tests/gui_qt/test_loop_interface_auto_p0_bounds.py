"""Automatic initial guesses must preserve explicit parameter constraints."""

import numpy as np
import pytest
from PySide6.QtCore import QThread

from nlsq.gui_qt.adapters.config_adapter import load_yaml_config
from nlsq.gui_qt.adapters.model_adapter import get_model
from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage
from nlsq.gui_qt.widgets.param_config import ParamConfigWidget

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("bounds", ["[0, 5]", "{lower: [0, 1], upper: [4, 5]}"])
def test_imported_bounds_reach_worker_with_automatic_guess(
    app_state, qtbot, monkeypatch, bounds
):
    app_state._state = load_yaml_config(
        f"model: {{type: builtin, name: linear}}\nfitting: {{bounds: {bounds}}}"
    )
    app_state.set_data(np.arange(6.0), 2 * np.arange(6.0) + 3)
    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    app_state.set_model(
        "builtin", {"name": "linear"}, get_model("builtin", {"name": "linear"})
    )
    app_state.session_restored.emit()
    monkeypatch.setattr(QThread, "start", lambda self: None)
    page.run_fit()
    worker_state = page._fit_worker._state
    assert worker_state.p0 is None
    page._cleanup_thread()
    app_state.set_fit_aborted()
    expected = ([0, 0], [5, 5]) if bounds.startswith("[") else ([0, 1], [4, 5])
    assert worker_state.bounds == expected


def test_automatic_guess_allows_manual_bound_edits(app_state, qtbot):
    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    app_state.set_model(
        "builtin", {"name": "linear"}, get_model("builtin", {"name": "linear"})
    )
    params = page._param_config
    assert not params._table.cellWidget(0, 1).isEnabled()
    lower = params._table.cellWidget(0, 2)
    assert lower.isEnabled()
    lower.setValue(0)
    assert app_state.state.p0 is None
    assert app_state.state.bounds == ([0, None], [None, None])


def test_default_automatic_guess_remains_unbounded(qtbot):
    params = ParamConfigWidget()
    qtbot.addWidget(params)
    params.set_param_names(["a", "b"])
    assert params.get_values() == ([], None)
