"""PR #63 review follow-ups: option validators, result indexing, CLI/curve_fit parity."""

from __future__ import annotations

import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.core._option_checks import (
    validate_f_scale,
    validate_tolerance,
    validate_tr_solver,
)


def _lin(x, a, b):
    return a * x + b


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_validate_f_scale_rejects_non_positive_or_non_finite(bad):
    with pytest.raises(ValueError, match="f_scale"):
        validate_f_scale(bad)


def test_validate_tr_solver_rejects_unknown_name():
    validate_tr_solver("exact")
    with pytest.raises(ValueError, match="tr_solver"):
        validate_tr_solver("bogus")


@pytest.mark.parametrize("bad", [np.inf, -1.0])
def test_validate_tolerance_strict_rejects_inf_and_negative(bad):
    with pytest.raises(ValueError, match="ftol"):
        validate_tolerance("ftol", bad, strict=True)


def test_result_indexes_with_numpy_integer():
    x = np.linspace(0, 1, 20)
    res = curve_fit(_lin, x, 2 * x + 1, p0=[1.0, 0.0])
    np.testing.assert_array_equal(res[np.int64(0)], res.popt)
    np.testing.assert_array_equal(res[np.int64(1)], res.pcov)


def test_full_output_ier_tolerates_missing_status():
    """`status=None` on a result must not raise TypeError when building ier."""
    from nlsq.core.minpack import CurveFit

    x = np.linspace(0, 1, 20)
    out = CurveFit().curve_fit(_lin, x, 2 * x + 1, p0=[1.0, 0.0], full_output=True)
    assert out[-1] in (1, 2, 3, 4)


def test_cli_absolute_sigma_matches_curve_fit_rule():
    from nlsq.cli.errors import ConfigError
    from nlsq.core._option_checks import coerce_absolute_sigma

    assert coerce_absolute_sigma(1) is True
    assert coerce_absolute_sigma(np.bool_(False)) is False
    for bad in ("yes", 2, 0.5):
        with pytest.raises(TypeError):
            coerce_absolute_sigma(bad)
    assert issubclass(ConfigError, Exception)


def test_prediction_interval_keeps_sigma_weighting_with_data_mask():
    x = np.arange(5.0)
    sigma = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([True, False, True, True, True])
    res = curve_fit(
        _lin,
        x,
        2 * x + 1,
        p0=[1.0, 0.0],
        sigma=sigma,
        absolute_sigma=True,
        data_mask=mask,
    )
    assert np.asarray(res.sigma).shape == (4,)
    np.testing.assert_allclose(res.sigma, sigma[mask])
    width = np.diff(res.prediction_interval(), axis=1).ravel()
    # exact fit: the width is driven by sigma(x), which grows along the kept points
    assert np.all(np.diff(width) > 0)


def test_prediction_interval_masks_covariance_sigma():
    from nlsq.core.minpack import CurveFit

    sig = np.diag([1.0, 4.0, 9.0, 16.0])
    mask = np.array([True, False, True, True])
    out = CurveFit._drop_unobserved_sigma(sig, mask)
    np.testing.assert_array_equal(out, np.diag([1.0, 9.0, 16.0]))
    assert CurveFit._drop_unobserved_sigma(None, mask) is None
    too_long = np.ones(5)
    assert CurveFit._drop_unobserved_sigma(too_long, mask) is too_long


def test_exact_streaming_and_trf_share_the_tolerance_rule():
    from nlsq.core.least_squares import check_tolerance
    from nlsq.streaming.exact_lm import _check_tolerances

    msg = "higher than machine epsilon"
    with pytest.raises(ValueError, match=msg):
        check_tolerance(0.0, 0.0, 0.0, "trf")
    with pytest.raises(ValueError, match=msg):
        _check_tolerances(0.0, 0.0, 0.0)


def test_sigma_value_check_shared_messages():
    from nlsq.core._option_checks import check_sigma_values

    with pytest.raises(ValueError, match="finite"):
        check_sigma_values(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="non-zero"):
        check_sigma_values(np.array([1.0, 0.0]))
    check_sigma_values(np.array([1.0, np.nan]), require_finite=False)


def test_cmaes_config_integral_fields():
    from nlsq.global_optimization.cmaes_config import CMAESConfig

    cfg = CMAESConfig(popsize=8.0, seed=np.int64(3))
    assert type(cfg.popsize) is int and type(cfg.seed) is int
    for bad in (True, 2.5, float("nan")):
        with pytest.raises(ValueError, match="popsize must be an integer"):
            CMAESConfig(popsize=bad)  # type: ignore[arg-type]


def test_ld_memory_config_min_chunk_default_follows_max_but_explicit_does_not():
    from nlsq.streaming.large_dataset import LDMemoryConfig

    assert LDMemoryConfig(max_chunk_size=500).min_chunk_size == 500
    assert LDMemoryConfig().min_chunk_size == 1000
    with pytest.raises(ValueError, match="min_chunk_size"):
        LDMemoryConfig(min_chunk_size=1000, max_chunk_size=500)


def test_padded_masked_fit_keeps_sigma_weighting():
    from nlsq.core.minpack import CurveFit

    x = np.arange(5.0)
    sigma = np.arange(1.0, 6.0)
    mask = np.array([True, False, True, True, True])
    res = CurveFit(flength=8).curve_fit(
        _lin,
        x,
        2 * x + 1,
        p0=[1.0, 0.0],
        sigma=sigma,
        absolute_sigma=True,
        data_mask=mask,
    )
    np.testing.assert_allclose(res.sigma, sigma[mask])


def test_ld_memory_config_default_provenance_survives_replace_copy_pickle():
    import copy
    import dataclasses
    import pickle

    from nlsq.streaming.large_dataset import LDMemoryConfig

    base = LDMemoryConfig()
    assert dataclasses.replace(base, max_chunk_size=500).min_chunk_size == 500
    assert copy.deepcopy(base).min_chunk_size == 1000
    assert dataclasses.replace(pickle.loads(pickle.dumps(base)), max_chunk_size=500)
    explicit = dataclasses.replace(base, min_chunk_size=1000)
    with pytest.raises(ValueError, match="min_chunk_size"):
        dataclasses.replace(explicit, max_chunk_size=500)


def test_integral_check_uses_exact_value_and_accepts_decimal():
    from decimal import Decimal
    from fractions import Fraction

    from nlsq.global_optimization.cmaes_config import CMAESConfig

    with pytest.raises(ValueError, match="seed must be an integer"):
        CMAESConfig(seed=Fraction(18014398509481985, 2))  # type: ignore[arg-type]
    cfg = CMAESConfig(popsize=Decimal(20))  # type: ignore[arg-type]
    assert cfg.popsize == 20 and type(cfg.popsize) is int


# --- review-pr pass 2 -------------------------------------------------------


def test_validate_tolerance_returns_float_not_raw_object():
    out = validate_tolerance("ftol", np.array(1e-8))
    assert type(out) is float and out == 1e-8
    assert type(validate_tolerance("ftol", None)) is float


def test_check_optimizer_kwargs_rejects_0d_nan_f_scale():
    from nlsq.core.minpack import _check_optimizer_kwargs

    with pytest.raises(ValueError, match="f_scale"):
        _check_optimizer_kwargs({"f_scale": np.array(np.nan)})


def test_diagnostics_config_window_sizes_coerced_to_int():
    from nlsq.diagnostics.types import DiagnosticsConfig

    cfg = DiagnosticsConfig(gradient_window_size=50.0, stagnation_window=np.int64(4))
    assert type(cfg.gradient_window_size) is int and cfg.gradient_window_size == 50
    assert type(cfg.stagnation_window) is int
    with pytest.raises(ValueError, match="integer"):
        DiagnosticsConfig(gradient_window_size="x")


def test_hybrid_config_rejects_numpy_bool_verbose():
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.validators import ConfigValidationError

    with pytest.raises((ConfigValidationError, ValueError), match="verbose"):
        HybridStreamingConfig(verbose=np.True_)
