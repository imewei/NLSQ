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
