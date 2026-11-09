"""Shared option validators for the fitting entry points.

One rule per option, used by ``curve_fit``, ``fit``, ``least_squares``,
``curve_fit_large`` (including ``exact_streaming``) and the hybrid optimizer:

- ``absolute_sigma``: ``bool``, ``np.bool_``, 0-d arrays of those, or 0/1, coerced to ``bool``.
- ``ftol``/``xtol``/``gtol``: NaN raises; ``+inf`` or below machine epsilon warns;
  all three below epsilon raises; ``strict`` (exact solver) also raises on inf/negative.
- ``max_nfev``: ``None`` or a positive integer (integral floats coerced; bool rejected).
- integer counts: :func:`is_integral_count` backs ``streaming.validators``.
"""

from __future__ import annotations

import decimal
import math
import numbers
import warnings
from typing import Any

import numpy as np

from nlsq.core.covariance import _NLSQ_DIR
from nlsq.result.optimize_warning import OptimizeWarning

__all__ = [
    "check_not_nan",
    "check_sigma_values",
    "coerce_absolute_sigma",
    "is_integral",
    "is_integral_count",
    "require_usable_tolerances",
    "validate_f_scale",
    "validate_max_nfev",
    "validate_tolerance",
    "validate_tr_solver",
]

_EPS = float(np.finfo(float).eps)


def _unwrap_scalar(value: Any) -> Any:
    """0-d numpy/JAX arrays (HEAD and scipy accept them) become Python scalars."""
    if not isinstance(value, (bool, int, float)) and getattr(value, "ndim", None) == 0:
        return value.item()
    return value


def coerce_absolute_sigma(value: Any) -> bool:
    """Return ``value`` as ``bool``; ``TypeError`` for anything but bool-likes."""
    value = _unwrap_scalar(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return bool(value)
    raise TypeError(
        f"absolute_sigma must be a bool (or 0/1), got {type(value).__name__} "
        f"({value!r})"
    )


def is_integral(value: Any) -> bool:
    """True for a real, finite, integral, non-bool number (``3``, ``np.int64(3)``, ``3.0``)."""
    value = _unwrap_scalar(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Real, decimal.Decimal)
    ):
        return False
    if isinstance(value, numbers.Integral):
        return True
    # compare the original value, not its float rounding (Fraction, Decimal)
    return math.isfinite(value) and value == int(value)  # type: ignore[call-overload,arg-type]


def is_integral_count(value: Any, minimum: int) -> bool:
    """True for a real, finite, integral, non-bool number ``>= minimum``."""
    return is_integral(value) and int(_unwrap_scalar(value)) >= minimum


def require_usable_tolerances(ftol: float, xtol: float, gtol: float) -> None:
    """``ValueError`` when all three (already validated) tolerances are below eps."""
    if ftol < _EPS and xtol < _EPS and gtol < _EPS:
        raise ValueError(
            "At least one of the tolerances must be higher than "
            f"machine epsilon ({_EPS:.2e})."
        )


def check_sigma_values(sigma: Any, *, require_finite: bool = True) -> None:
    """``ValueError`` for a zero (and, by default, non-finite) entry of a 1-D ``sigma``."""
    s = np.asarray(sigma, dtype=float)
    if require_finite and not np.isfinite(s).all():
        raise ValueError("sigma must be finite (found NaN or inf)")
    if np.any(s == 0):
        raise ValueError("sigma values must be non-zero")


def validate_max_nfev(
    value: Any, *, allow_fractional: bool = False
) -> int | float | None:
    """Return ``max_nfev`` (``int`` or ``None``); ``ValueError`` otherwise.

    ``allow_fractional`` keeps a finite positive non-integral real as given: the
    TRF outer loop has always accepted e.g. ``5.5`` (terminates by ``>=``).
    """
    value = _unwrap_scalar(value)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("`max_nfev` must be None or positive integer, not bool.")
    if allow_fractional and isinstance(value, numbers.Real):
        num = float(value)
        if math.isfinite(num) and num > 0:
            return num if num != int(num) else int(num)
    elif is_integral_count(value, 1):
        return int(value)
    raise ValueError(f"`max_nfev` must be None or positive integer, got {value!r}.")


def check_not_nan(name: str, tol: Any) -> float:
    """Return ``tol`` as ``float``; NaN (also as a 0-d array) raises ``ValueError``."""
    value = float(np.asarray(tol, dtype=float))
    if math.isnan(value):
        raise ValueError(f"`{name}` must not be NaN.")
    return value


def validate_tr_solver(tr_solver: Any) -> None:
    # "sparse" is set internally after validation; "cg" is a trf.py branch.
    if tr_solver not in (None, "exact", "lsmr", "sparse", "cg"):
        raise ValueError(
            "`tr_solver` must be None, 'exact', 'lsmr', 'sparse' or 'cg', "
            f"got {tr_solver!r}."
        )


def validate_f_scale(f_scale: Any) -> None:
    if not (np.isfinite(f_scale) and f_scale > 0):
        raise ValueError(f"`f_scale` must be positive and finite, got {f_scale!r}.")


def validate_tolerance(name: str, tol: Any, *, strict: bool = False) -> float:
    """Validate one termination tolerance; ``None`` means 0 (disabled).

    NaN raises (it compares False to everything and would silently disable the
    test). ``+inf`` and values below machine epsilon warn. ``strict`` (the exact
    solver, where a no-op tolerance wastes full data passes) raises on inf and
    negative values instead.
    """
    if tol is None:
        return 0.0
    value = check_not_nan(name, tol)
    if strict and not (math.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be finite and non-negative, got {tol}")
    if value == math.inf:
        msg = (
            f"`{name}=inf` makes the corresponding termination condition "
            "hold immediately, so the solver stops at the initial guess."
        )
    elif value < _EPS:
        msg = (
            f"Setting `{name}` below the machine epsilon ({_EPS:.2e}) effectively "
            "disables the corresponding termination condition."
        )
    else:
        return value
    warnings.warn(msg, OptimizeWarning, skip_file_prefixes=_NLSQ_DIR)
    return value
