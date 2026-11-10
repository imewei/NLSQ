"""R7: from_dict drops unknown keys with a warning; n_starts coercion/validation."""

import warnings

import numpy as np
import pytest

from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.result import OptimizeWarning


def test_from_dict_warns_and_drops_unknown_keys():
    with pytest.warns(OptimizeWarning, match="legacy_key"):
        cfg = GlobalOptimizationConfig.from_dict(
            {"n_starts": 7, "legacy_key": 1, "sampler": "sobol"}
        )
    assert cfg.n_starts == 7 and cfg.sampler == "sobol"


def test_from_dict_known_keys_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GlobalOptimizationConfig.from_dict(GlobalOptimizationConfig().to_dict())


def test_with_overrides_still_raises_on_unknown_field():
    with pytest.raises(TypeError, match="n_start"):
        GlobalOptimizationConfig().with_overrides(n_start=5)


@pytest.mark.parametrize("value", [5.0, np.int64(5), np.float64(5.0)])
def test_n_starts_integral_values_coerced_to_int(value):
    cfg = GlobalOptimizationConfig(n_starts=value)
    assert cfg.n_starts == 5 and type(cfg.n_starts) is int


@pytest.mark.parametrize("value", [-1, -3.0, float("nan"), float("inf"), 2.5])
def test_n_starts_bad_values_are_value_errors(value):
    with pytest.raises(ValueError):
        GlobalOptimizationConfig(n_starts=value)


@pytest.mark.parametrize("value", [True, np.bool_(False), "5", None, [3]])
def test_n_starts_wrong_types_are_type_errors(value):
    with pytest.raises(TypeError):
        GlobalOptimizationConfig(n_starts=value)


def test_n_starts_zero_ok():
    assert GlobalOptimizationConfig(n_starts=0).n_starts == 0
