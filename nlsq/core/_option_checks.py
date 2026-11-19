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
    """Return ``value`` as ``bool``; ``TypeError`` for anything but bool-likes.

    Parameters
    ----------
    value : Any
        A ``bool``, ``np.bool_``, integer ``0`` or ``1``, or a 0-d array
        holding one of these.

    Returns
    -------
    bool
        ``value`` as a Python ``bool``.

    Raises
    ------
    TypeError
        If ``value`` is not one of the accepted forms.
    """
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
    """True for a real, finite, integral, non-bool number (``3``, ``np.int64(3)``, ``3.0``).

    Parameters
    ----------
    value : Any
        Value to test; a 0-d array is unwrapped to a scalar first.

    Returns
    -------
    bool
        ``True`` if ``value`` is such a number, ``False`` otherwise.
    """
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
    """True for a real, finite, integral, non-bool number ``>= minimum``.

    Parameters
    ----------
    value : Any
        Value to test (see :func:`is_integral`).
    minimum : int
        Smallest accepted value.

    Returns
    -------
    bool
        ``True`` if ``value`` is integral and ``>= minimum``.
    """
    return is_integral(value) and int(_unwrap_scalar(value)) >= minimum


def require_usable_tolerances(ftol: float, xtol: float, gtol: float) -> None:
    """``ValueError`` when all three (already validated) tolerances are below eps.

    Parameters
    ----------
    ftol, xtol, gtol : float
        Termination tolerances.

    Raises
    ------
    ValueError
        If ``ftol``, ``xtol`` and ``gtol`` are all below machine epsilon.
    """
    if ftol < _EPS and xtol < _EPS and gtol < _EPS:
        raise ValueError(
            "At least one of the tolerances must be higher than "
            f"machine epsilon ({_EPS:.2e})."
        )


def check_sigma_values(sigma: Any, *, require_finite: bool = True) -> None:
    """``ValueError`` for a zero (and, by default, non-finite) entry of a 1-D ``sigma``.

    Parameters
    ----------
    sigma : array_like
        Uncertainties, converted with ``np.asarray(sigma, dtype=float)``.
    require_finite : bool, optional
        If True (default), NaN and inf entries are rejected.

    Raises
    ------
    ValueError
        If ``require_finite`` is True and an entry is NaN or inf, or if any
        entry equals zero.
    """
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

    Parameters
    ----------
    value : Any
        ``None`` or a positive integer (an integral float is converted to
        ``int``); a 0-d array is unwrapped first. ``bool`` is rejected.
    allow_fractional : bool, optional
        If True, a finite positive real that is not integral is returned as a
        float instead of raising. Default is False.

    Returns
    -------
    int, float or None
        ``None`` if ``value`` is ``None``; otherwise the validated count.

    Raises
    ------
    ValueError
        If ``value`` is a ``bool`` or is not ``None`` or an accepted positive
        number.
    """
    value = _unwrap_scalar(value)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("`max_nfev` must be None or positive integer, not bool.")
    if allow_fractional and isinstance(value, numbers.Integral):
        # float() would round above 2**53 and overflow for huge ints
        if int(value) > 0:
            return int(value)
    elif allow_fractional and isinstance(value, numbers.Real):
        num = float(value)
        if math.isfinite(num) and num > 0:
            return num if num != int(num) else int(num)
    elif is_integral_count(value, 1):
        return int(value)
    raise ValueError(f"`max_nfev` must be None or positive integer, got {value!r}.")


def has_complex(a: Any) -> bool:
    """True when ``a`` is complex, including an object array that boxes a complex scalar.

    ``np.iscomplexobj`` looks only at the dtype, so ``dtype=object`` holding
    ``1+2j`` slips through and the later float cast drops the imaginary part.
    Not array-like input is False; an out-of-core source keeps its own dtype test.

    Parameters
    ----------
    a : Any
        Array-like or ``None``.

    Returns
    -------
    bool
        ``True`` if ``a`` has a complex dtype or is an object array containing a
        complex scalar; ``False`` otherwise, including ``None`` and input that
        cannot be converted.
    """
    if a is None:
        return False
    try:
        if np.iscomplexobj(a):
            return True
        if getattr(a, "dtype", None) is not None and a.dtype != object:
            return False
        arr = np.asarray(a)
        if arr.dtype != object:
            return False
        return any(isinstance(v, complex | np.complexfloating) for v in arr.ravel())
    except (TypeError, ValueError):
        return False


def check_not_nan(name: str, tol: Any) -> float:
    """Return ``tol`` as ``float``; NaN (also as a 0-d array) raises ``ValueError``.

    Parameters
    ----------
    name : str
        Option name used in the error message.
    tol : Any
        Tolerance value, converted with ``np.asarray``.

    Returns
    -------
    float
        ``tol`` as a Python float.

    Raises
    ------
    ValueError
        If ``tol`` is complex or NaN.
    """
    arr = np.asarray(tol)
    if has_complex(arr):  # the float cast would drop the imaginary part silently
        raise ValueError(f"`{name}` must be real, got {tol!r}.")
    value = float(arr.astype(float))
    if math.isnan(value):
        raise ValueError(f"`{name}` must not be NaN.")
    return value


def validate_tr_solver(tr_solver: Any) -> None:
    """Raise ``ValueError`` unless ``tr_solver`` is None, 'exact', 'lsmr', 'sparse' or 'cg'.

    Parameters
    ----------
    tr_solver : Any
        Trust-region solver option.

    Raises
    ------
    ValueError
        If ``tr_solver`` is not one of the accepted values.
    """
    # "sparse" is set internally after validation; "cg" is a trf.py branch.
    if tr_solver not in (None, "exact", "lsmr", "sparse", "cg"):
        raise ValueError(
            "`tr_solver` must be None, 'exact', 'lsmr', 'sparse' or 'cg', "
            f"got {tr_solver!r}."
        )


def validate_f_scale(f_scale: Any) -> None:
    """Raise ``ValueError`` unless ``f_scale`` is positive and finite.

    Parameters
    ----------
    f_scale : Any
        Soft margin between inlier and outlier residuals.

    Raises
    ------
    ValueError
        If ``f_scale`` is not finite or is not greater than zero.
    TypeError
        If ``f_scale`` is not numeric.
    """
    if not (np.isfinite(f_scale) and f_scale > 0):
        raise ValueError(f"`f_scale` must be positive and finite, got {f_scale!r}.")


def validate_tolerance(name: str, tol: Any, *, strict: bool = False) -> float:
    """Validate one termination tolerance; ``None`` means 0 (disabled).

    NaN raises (it compares False to everything and would silently disable the
    test). ``+inf`` and values below machine epsilon warn. ``strict`` (the exact
    solver, where a no-op tolerance wastes full data passes) raises on inf and
    negative values instead.

    Parameters
    ----------
    name : str
        Option name used in messages.
    tol : Any
        Tolerance value, or ``None``.
    strict : bool, optional
        If True, raise on non-finite or negative values. Default is False.

    Returns
    -------
    float
        The tolerance as a float; ``0.0`` if ``tol`` is ``None``.

    Raises
    ------
    ValueError
        If ``tol`` is NaN or complex, or if ``strict`` is True and ``tol`` is
        not finite and non-negative.
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
