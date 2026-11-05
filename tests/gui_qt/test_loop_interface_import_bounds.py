"""Imported CLI bounds must remain usable by GUI background fits."""

import numpy as np
import pytest

from nlsq.gui_qt.adapters.config_adapter import load_yaml_config
from nlsq.gui_qt.adapters.fit_adapter import run_fit
from nlsq.gui_qt.adapters.model_adapter import get_model


@pytest.mark.parametrize("bounds", ["[0, 5]", "[0, [4, 5]]"])
def test_cli_bound_pair_broadcasts_before_gui_snapshot(bounds):
    state = load_yaml_config(
        "model: {type: builtin, name: linear}\n"
        f"fitting: {{p0: [1, 1], bounds: {bounds}}}\n"
    )
    assert state.bounds == ([0, 0], [4, 5] if "4" in bounds else [5, 5])
    state.xdata = np.arange(6.0)
    state.ydata = 2 * state.xdata + 3
    state.model_func = get_model("builtin", {"name": "linear"})
    result = run_fit(state.snapshot_for_fit())
    np.testing.assert_allclose(result.popt, [2, 3], rtol=1e-5)


def test_dict_scalar_bounds_keep_broadcasting():
    state = load_yaml_config("fitting: {p0: [1, 1], bounds: {lower: 0, upper: 5}}")
    assert state.snapshot_for_fit().bounds == ([0, 0], [5, 5])
