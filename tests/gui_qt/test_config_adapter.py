"""Workflow import/export must preserve the settings the user selected."""

import numpy as np
import pytest

from nlsq.gui_qt.adapters.config_adapter import export_yaml_config, load_yaml_config
from nlsq.gui_qt.session_state import SessionState


@pytest.mark.parametrize("workflow", ["auto_global", "hpc"])
def test_yaml_round_trip_preserves_workflow_and_global_options(workflow):
    original = SessionState(
        workflow=workflow,
        n_starts=23,
        sampler="sobol",
        center_on_p0=False,
        scale_factor=3.0,
    )

    restored = load_yaml_config(export_yaml_config(original))

    assert restored.workflow == original.workflow
    assert restored.n_starts == original.n_starts
    assert restored.sampler == original.sampler
    assert restored.center_on_p0 == original.center_on_p0
    assert restored.scale_factor == original.scale_factor


def test_import_current_cli_workflow_preserves_model_and_fit_settings():
    restored = load_yaml_config("""
model:
  type: polynomial
  degree: 5
fitting:
  p0: [2.0, 1.0]
  bounds: {lower: [0.0, 0.0], upper: [4.0, 3.0]}
  gtol: 1.0e-6
  ftol: 2.0e-6
  xtol: 3.0e-6
  max_nfev: 345
  loss: huber
""")

    assert restored.polynomial_degree == 5
    assert restored.p0 == [2.0, 1.0]
    assert restored.bounds == ([0.0, 0.0], [4.0, 3.0])
    assert restored.gtol == 1e-6
    assert restored.ftol == 2e-6
    assert restored.xtol == 3e-6
    assert restored.max_function_evals == 345
    assert restored.loss == "huber"


def test_partial_parameter_bounds_remain_aligned_with_parameters():
    restored = load_yaml_config("""
model:
  parameters:
    - {name: slope, initial: 1.0}
    - {name: offset, initial: 2.0, bounds: [0.0, 5.0]}
""")

    assert restored.p0 == [1.0, 2.0]
    np.testing.assert_array_equal(restored.bounds[0], [-np.inf, 0.0])
    np.testing.assert_array_equal(restored.bounds[1], [np.inf, 5.0])


def test_import_cli_one_sided_bounds_expands_unbounded_side():
    restored = load_yaml_config("""
fitting:
  p0: [2.0, 1.0]
  bounds: {lower: [0.0, 0.0]}
""")

    assert restored.bounds[0] == [0.0, 0.0]
    assert restored.bounds[1] == [np.inf, np.inf]


def test_null_parameter_bounds_remain_aligned():
    restored = load_yaml_config("""
model:
  parameters:
    - {name: slope, initial: 1.0, bounds: null}
    - {name: offset, initial: 2.0, bounds: [0.0, 5.0]}
""")

    np.testing.assert_array_equal(restored.bounds[0], [-np.inf, 0.0])
    np.testing.assert_array_equal(restored.bounds[1], [np.inf, 5.0])


def test_incomplete_initial_parameters_are_rejected():
    with pytest.raises(ValueError, match="initial"):
        load_yaml_config("""
model:
  parameters:
    - {name: slope}
    - {name: offset, initial: 2.0}
""")
