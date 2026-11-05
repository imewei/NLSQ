"""Input preparation for ``curve_fit``: ``prepare_problem``.

One pipeline turns the raw ``curve_fit`` arguments into a ``PreparedProblem``.
The step order is fixed and is the contract of this module:

1. parameter count (and ``p0='auto'`` estimation), solver validation, bounds/``p0``;
2. validation, array conversion and length check, then NaN removal when
   ``nan_policy='omit'`` (``sigma`` and a caller ``data_mask`` are realigned to
   the surviving rows) -- before method selection, so the selector's
   data-driven heuristics never see NaN/Inf rows that ``omit`` asked to remove;
3. method selection (on the cleaned, unpadded data);
4. mask construction and padding to ``flength``.

Downstream the effective point count is ``PreparedProblem.n_effective`` (the
number of unmasked rows), not ``m_real``: it drives the covariance
degrees of freedom. ``len_diff`` is the padding length and feeds the sigma
transform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import signature
from typing import Any

import jax.numpy as jnp
import numpy as np

from nlsq.common_scipy import EPS
from nlsq.core.least_squares import prepare_bounds
from nlsq.utils.logging import NLSQLogger, get_logger
from nlsq.utils.validators import InputValidator

__all__ = ["PrepareContext", "PreparedProblem", "prepare_problem"]


@dataclass(frozen=True)
class PrepareContext:
    """``CurveFit`` state that input preparation reads.

    Callers that do not pad and do not use stability checks pass nothing.
    ``logger`` and ``validator`` default to a ``curve_fit`` logger and a
    fast-mode ``InputValidator``.
    """

    flength: float | None = None  # integral length; a float such as 1e6 is accepted
    use_dynamic_sizing: bool = False
    enable_stability: bool = False
    # Layout of multi-dimensional x for the x/y length check only: True is
    # (n_features, n_points) (standard path); hybrid streaming passes False for
    # (n_points, n_features). NaN omission (nan_policy="omit") stays
    # feature-major either way, so multi-dimensional hybrid x with
    # nan_policy="omit" is unsupported, exactly as before this field existed.
    x_points_axis_last: bool = True
    validator: InputValidator = field(
        default_factory=lambda: InputValidator(fast_mode=True)
    )
    logger: NLSQLogger = field(default_factory=lambda: get_logger("curve_fit"))


@dataclass(frozen=True)
class PreparedProblem:
    """Validated, cleaned and (optionally) padded ``curve_fit`` inputs.

    Field mapping from the legacy 13-tuple ``(n, p0, xdata, ydata, data_mask,
    method, lb, ub, m, len_diff, should_pad, none_mask, sigma)``:
    ``xdata -> x``, ``ydata -> y``, ``data_mask -> mask``, ``m -> m_real``;
    the rest keep their names. ``n_effective`` is derived.
    """

    n: int
    p0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    x: Any
    y: Any
    sigma: Any | None
    mask: np.ndarray
    method: str
    m_real: int
    n_effective: int
    len_diff: int
    should_pad: bool
    none_mask: bool


def _is_auto_p0(p0: object) -> bool:
    """True if p0 is the "auto" sentinel requesting parameter estimation.

    p0="auto" is a control-flow sentinel, not literal parameter data --
    np.atleast_1d("auto")/np.asarray("auto") would otherwise treat it as a
    1-element string array, miscounting n_params as 1 regardless of the
    model's real parameter count.
    """
    return isinstance(p0, str) and p0 == "auto"


def _initialize_feasible(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Initialize feasible parameters for optimization.

    This function initializes feasible parameters for optimization based on the
    lower and upper bounds of the variables. If both bounds are finite, the
    feasible parameters are set to the midpoint between the bounds. If only the
    lower bound is finite, the feasible parameters are set to the lower bound
    plus 1. If only the upper bound is finite, the feasible parameters are set
    to the upper bound minus 1. If neither bound is finite, the feasible
    parameters are set to 1.

    Parameters
    ----------
    lb : np.ndarray
        The lower bounds of the variables.
    ub : np.ndarray
        The upper bounds of the variables.

    Returns
    -------
    np.ndarray
        The initialized feasible parameters.
    """

    p0 = np.ones_like(lb)
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])

    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1

    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1

    return p0


def pad_fit_data(
    xdata: np.ndarray,
    ydata: np.ndarray,
    xdims: int,
    len_diff: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad fit data to match the fixed input data length.

    This function pads the input data arrays with small values to match the
    fixed input data length to avoid JAX retracing the JITted functions.
    The padding is added along the second dimension of the `xdata` array
    if it's multidimensional data otherwise along the first dimension. The
    small values are chosen to be `EPS`, a global constant defined as a
    very small positive value which avoids numerical issues.

    Parameters
    ----------
    xdata : np.ndarray
        The independent variables of the data.
    ydata : np.ndarray
        The dependent variables of the data.
    xdims : int
        The number of dimensions in the `xdata` array.
    len_diff : int
        The difference in length between the data arrays and the fixed input data length.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The padded `xdata` and `ydata` arrays.
    """

    if xdims > 1:
        xpad = EPS * np.ones([xdata.shape[0], len_diff])
        xdata = np.concatenate([xdata, xpad], axis=1)
    else:
        xpad = EPS * np.ones([len_diff])
        xdata = np.concatenate([xdata, xpad])
    ypad = EPS * np.ones([len_diff])
    ydata = np.concatenate([ydata, ypad])
    return xdata, ydata


def _determine_parameter_count(
    f: Callable,
    p0: np.ndarray | str | None,
    xdata: np.ndarray | None = None,
    ydata: np.ndarray | None = None,
    *,
    ctx: PrepareContext,
) -> tuple[int, np.ndarray | None]:
    """Determine number of fit parameters from p0 or function signature.

    Parameters
    ----------
    f : Callable
        The fit function
    p0 : np.ndarray | None | 'auto'
        Initial parameter guess. If 'auto', will estimate from data
        if xdata and ydata are provided. If None, uses default behavior
        (determined by bounds in _prepare_bounds_and_initial_guess).
    xdata : np.ndarray, optional
        Independent variable data (for auto p0 estimation)
    ydata : np.ndarray, optional
        Dependent variable data (for auto p0 estimation)

    Returns
    -------
    n : int
        Number of parameters
    p0 : np.ndarray | None
        Validated p0 array (or None if auto-estimation not requested)
    """
    # If p0 is explicitly provided (not None or 'auto'), use it
    if p0 is not None and not _is_auto_p0(p0):
        p0 = np.atleast_1d(p0)
        n = p0.size
        return n, p0

    # Only auto-estimate if p0='auto' is explicitly requested
    # (not when p0=None, to preserve backward compatibility)
    if _is_auto_p0(p0) and xdata is not None and ydata is not None:
        try:
            # Lazy import to reduce module dependencies
            from nlsq.precision.parameter_estimation import (
                estimate_initial_parameters,
            )

            p0_estimated = estimate_initial_parameters(f, xdata, ydata, p0)
            p0 = np.atleast_1d(p0_estimated)
            n = p0.size
            ctx.logger.debug(
                "Auto-estimated initial parameters",
                p0=p0.tolist(),
                n_params=n,
            )
            return n, p0
        except Exception as e:
            # If auto-estimation fails, fall back to default behavior
            ctx.logger.warning(
                "Auto p0 estimation failed, using defaults",
                error=str(e),
            )
            if any(
                a.kind in (a.VAR_POSITIONAL, a.VAR_KEYWORD)
                for a in signature(f).parameters.values()
            ):
                raise ValueError(
                    "p0='auto' failed and the number of parameters of a model "
                    "with *args/**kwargs cannot be inferred; provide p0 explicitly."
                ) from e
            p0 = None  # drop the "auto" sentinel so bounds-derived defaults apply

    # Fall back: determine n from function signature, p0 stays None
    # (will be initialized to defaults in _prepare_bounds_and_initial_guess)
    sig = signature(f)
    args = sig.parameters
    if len(args) < 2:
        raise ValueError("Unable to determine number of fit parameters.")
    n = len(args) - 1

    return n, p0  # type: ignore[return-value]  # p0 is only a str for the 'auto' sentinel, which is resolved above


def _validate_solver_config(solver: str, batch_size: int | None) -> None:
    """Validate solver and batch_size configuration.

    Parameters
    ----------
    solver : str
        Solver type
    batch_size : int | None
        Batch size for minibatch solver

    Raises
    ------
    ValueError
        If solver or batch_size is invalid
    """
    valid_solvers = {"auto", "svd", "cg", "lsqr", "minibatch"}
    if solver not in valid_solvers:
        raise ValueError(
            f"Invalid solver '{solver}'. Must be one of {valid_solvers}.",
        )

    if solver == "minibatch" and batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when using minibatch solver.")


def _prepare_bounds_and_initial_guess(
    bounds: tuple,
    n: int,
    p0: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare bounds and initialize p0 if needed.

    Parameters
    ----------
    bounds : tuple
        Bounds tuple (lower, upper)
    n : int
        Number of parameters
    p0 : np.ndarray | None
        Initial parameter guess

    Returns
    -------
    lb : np.ndarray
        Lower bounds
    ub : np.ndarray
        Upper bounds
    p0 : np.ndarray
        Initial parameter guess (clipped to bounds if necessary)
    """
    lb, ub = prepare_bounds(bounds, n)
    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError(
            f"`bounds` must have length {n} (number of parameters); got lower "
            f"{lb.shape}, upper {ub.shape}."
        )
    if p0 is None:
        p0 = _initialize_feasible(lb, ub)
    else:
        # Clip p0 to bounds to ensure feasibility. This applies to any
        # p0 (user-supplied or auto-estimated), and is why NLSQ diverges
        # from scipy here: scipy.optimize.curve_fit raises
        # ValueError("x0 is infeasible") for an out-of-bounds p0 instead.
        p0 = np.clip(p0, lb, ub)

    return lb, ub, p0


def _select_optimization_method(
    method: str | None,
    f: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    bounds: tuple,
    kwargs: dict,
    *,
    ctx: PrepareContext,
) -> str:
    """Select optimization method, auto-selecting if needed.

    Parameters
    ----------
    method : str | None
        Optimization method (None for auto-selection)
    f : Callable
        Fit function
    xdata : np.ndarray
        X data
    ydata : np.ndarray
        Y data
    p0 : np.ndarray
        Initial parameter guess
    bounds : tuple
        Bounds tuple
    kwargs : dict
        Additional optimization parameters

    Returns
    -------
    method : str
        Selected optimization method
    """
    if method is None:
        if ctx.enable_stability:
            # Lazy import to reduce module dependencies
            from nlsq.precision.algorithm_selector import auto_select_algorithm

            recommendations = auto_select_algorithm(f, xdata, ydata, p0, bounds)
            method = recommendations["algorithm"]
            ctx.logger.info(
                "Auto-selected algorithm",
                method=method,
                loss=recommendations.get("loss", "linear"),
            )
            # Apply recommended parameters to kwargs
            for key in ["ftol", "xtol", "gtol", "max_nfev", "x_scale"]:
                if key in recommendations and key not in kwargs:
                    kwargs[key] = recommendations[key]
        else:
            method = "trf"

    return method


def _validate_and_sanitize_inputs(
    f: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    bounds: tuple | None = None,
    check_finite: bool = True,
    *,
    ctx: PrepareContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and sanitize curve fit inputs if stability is enabled.

    Parameters
    ----------
    f : Callable
        Fit function
    xdata : np.ndarray
        X data
    ydata : np.ndarray
        Y data
    p0 : np.ndarray
        Initial parameter guess

    Returns
    -------
    xdata : np.ndarray
        Cleaned X data
    ydata : np.ndarray
        Cleaned Y data
    """
    # Security constraints (array-size/Jacobian-overflow DoS limits,
    # bounds/parameter range checks) must run regardless of
    # enable_stability -- ctx.enable_stability defaults to False on
    # both CurveFit and the public curve_fit() wrapper, and these
    # checks were previously only reachable through the full
    # validate_curve_fit_inputs() pipeline gated behind it, making them
    # dead code on the default "drop-in scipy.optimize.curve_fit
    # replacement" path.
    security_validator = ctx.validator
    n_params = security_validator._estimate_n_params(f, p0)
    security_errors, security_warnings = (
        security_validator.validate_security_constraints(
            len(ydata),
            n_params,
            bounds,
            p0,
        )
    )
    if security_errors:
        error_msg = f"Input validation failed: {'; '.join(security_errors)}"
        ctx.logger.error("Input validation failed", error=error_msg)
        raise ValueError(error_msg)
    for warning in security_warnings:
        ctx.logger.warning("Input validation warning", warning=warning)

    if ctx.enable_stability:
        try:
            errors, warnings_list, xdata_clean, ydata_clean = (
                ctx.validator.validate_curve_fit_inputs(
                    f,
                    xdata,
                    ydata,
                    p0,
                    bounds=bounds,
                    check_finite=check_finite,
                )
            )

            if errors:
                error_msg = f"Input validation failed: {'; '.join(errors)}"
                ctx.logger.error("Input validation failed", error=error_msg)
                raise ValueError(error_msg)

            for warning in warnings_list:
                ctx.logger.warning("Input validation warning", warning=warning)

            xdata = xdata_clean
            ydata = ydata_clean

        except ValueError as e:
            if "too many values to unpack" not in str(e):
                ctx.logger.error("Input validation failed", error=str(e))
            raise

    return xdata, ydata


def _convert_and_validate_arrays(
    xdata: np.ndarray | tuple | list,
    ydata: np.ndarray,
    check_finite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to arrays and validate finiteness.

    Parameters
    ----------
    xdata : array-like
        X data
    ydata : array-like
        Y data
    check_finite : bool
        Whether to check for finite values

    Returns
    -------
    xdata : np.ndarray
        X data as array
    ydata : np.ndarray
        Y data as array
    """
    # Convert ydata
    if check_finite:
        ydata = np.asarray_chkfinite(ydata, float)
    else:
        ydata = np.asarray(ydata, float)

    # Convert xdata
    if hasattr(xdata, "__array__") or isinstance(
        xdata,
        (list, tuple, np.ndarray, jnp.ndarray),
    ):
        if check_finite:
            xdata = np.asarray_chkfinite(xdata, float)
        else:
            xdata = np.asarray(xdata, float)
    else:
        raise ValueError("X needs arrays")

    if ydata.size == 0:
        raise ValueError("`ydata` must not be empty!")

    return xdata, ydata


def _validate_data_lengths(
    xdata: np.ndarray,
    ydata: np.ndarray,
    points_axis_last: bool = True,
) -> tuple[int, int]:
    """Validate that X and Y data lengths match.

    Parameters
    ----------
    xdata : np.ndarray
        X data
    ydata : np.ndarray
        Y data
    points_axis_last : bool, default True
        For multi-dimensional ``xdata``: points on the last axis
        ``(n_features, n_points)`` (standard path) or the first axis
        ``(n_points, n_features)`` (hybrid streaming).

    Returns
    -------
    m : int
        Data length
    xdims : int
        X data dimensionality

    Raises
    ------
    ValueError
        If X and Y lengths don't match
    """
    m = len(ydata)
    xdims = xdata.ndim
    xlen = len(xdata) if xdims == 1 or not points_axis_last else len(xdata[0])
    if xlen != m:
        raise ValueError("X and Y data lengths dont match")

    return m, xdims


def _setup_data_mask_and_padding(
    data_mask: np.ndarray | None,
    m: int,
    *,
    ctx: PrepareContext,
) -> tuple[np.ndarray, bool, int, bool]:
    """Setup data mask and compute padding parameters.

    Parameters
    ----------
    data_mask : np.ndarray | None
        Optional data mask
    m : int
        Data length

    Returns
    -------
    data_mask : np.ndarray
        Data mask array
    none_mask : bool
        Whether data_mask was None on input
    len_diff : int
        Length difference for padding
    should_pad : bool
        Whether padding is needed
    """
    none_mask = data_mask is None
    should_pad = False
    len_diff = 0

    if data_mask is not None:
        data_mask = np.asarray(data_mask)  # PreparedProblem.mask is an ndarray
        if len(data_mask) != m:
            raise ValueError("Data mask doesn't match data lengths.")

    if ctx.flength is not None:
        len_diff = int(ctx.flength) - m
        if ctx.use_dynamic_sizing:
            should_pad = len_diff > 0
        else:
            should_pad = len_diff >= 0

        if data_mask is None:
            data_mask = np.ones(m, dtype=bool)
        if should_pad and len_diff > 0:
            data_mask = np.concatenate(
                [data_mask, np.zeros(len_diff, dtype=bool)],
            )
    elif data_mask is None:
        data_mask = np.ones(m, dtype=bool)

    return data_mask, none_mask, len_diff, should_pad


def _preprocess_input_data(
    f: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    check_finite: bool,
    nan_policy: str = "raise",
    bounds: tuple | None = None,
    sigma: np.ndarray | None = None,
    *,
    ctx: PrepareContext,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray | None, np.ndarray | None]:
    """Preprocess input data.

    Parameters
    ----------
    f : Callable
        Model function
    xdata : np.ndarray
        X data
    ydata : np.ndarray
        Y data
    p0 : np.ndarray
        Initial parameter guess
    check_finite : bool
        Whether to check for finite values
    sigma : np.ndarray, optional
        Uncertainties/weights. When nan_policy='omit', rows dropped for
        non-finite x/y (or a non-finite 1D sigma entry) are also dropped
        from sigma, so the returned sigma stays aligned with xdata/ydata.

    Returns
    -------
    xdata : np.ndarray
        Validated X data
    ydata : np.ndarray
        Validated Y data
    m : int
        Data length
    xdims : int
        X data dimensionality
    sigma : np.ndarray | None
        Sigma, filtered consistently with xdata/ydata under nan_policy='omit'
    omit_mask : np.ndarray | None
        Boolean mask (over the ORIGINAL pre-omit length) used to filter
        rows, so a caller-supplied data_mask can be realigned the same
        way. None when nan_policy != 'omit'.
    """
    # Defer finiteness raising when nan_policy handles it instead:
    # 'omit' filters invalid rows below, 'propagate' intentionally keeps NaN/Inf.
    effective_check_finite = check_finite and nan_policy not in (
        "omit",
        "propagate",
    )

    # Step 6: Validate and sanitize inputs (if stability enabled).
    # Forward bounds + check_finite so bounds-consistency and finiteness
    # checks are actually exercised (previously they were silently dropped).
    xdata, ydata = _validate_and_sanitize_inputs(
        f,
        xdata,
        ydata,
        p0,
        bounds=bounds,
        check_finite=effective_check_finite,
        ctx=ctx,
    )

    # Step 7: Convert to arrays and validate finiteness
    xdata, ydata = _convert_and_validate_arrays(
        xdata,
        ydata,
        effective_check_finite,
    )

    # Step 8: Validate data lengths
    m, xdims = _validate_data_lengths(xdata, ydata, ctx.x_points_axis_last)

    omit_mask = None
    if nan_policy == "omit":
        xdata, ydata, sigma, m, omit_mask = _omit_nonfinite_rows(
            xdata,
            ydata,
            sigma,
            xdims,
        )

    return xdata, ydata, m, xdims, sigma, omit_mask


def _omit_nonfinite_rows(
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray | None,
    xdims: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, np.ndarray]:
    """Drop rows with non-finite x/y (or a non-finite 1D sigma entry).

    Parameters
    ----------
    xdata, ydata : np.ndarray
        Data to filter.
    sigma : np.ndarray | None
        1D sigma is filtered alongside xdata/ydata (rows with a
        non-finite sigma entry are also dropped). A 2D covariance-matrix
        sigma is row/column-extracted to match the same mask.
    xdims : int
        xdata.ndim, to select row-wise vs column-wise masking.

    Returns
    -------
    xdata, ydata, sigma, m, omit_mask : filtered data, new data length, and
        the boolean mask (over the ORIGINAL, pre-omit length) used to
        filter — so a caller-supplied data_mask can be filtered the same
        way to stay aligned.
    """
    y_valid = np.isfinite(ydata)
    x_valid = np.isfinite(xdata) if xdims == 1 else np.all(np.isfinite(xdata), axis=0)
    valid_mask = y_valid & x_valid

    sigma_arr = np.asarray(sigma) if sigma is not None else None
    sigma_1d = sigma_arr is not None and sigma_arr.ndim == 1
    if sigma_1d:
        valid_mask = valid_mask & np.isfinite(sigma_arr)  # type: ignore[arg-type]  # sigma_arr is not None whenever sigma_1d is True

    if not np.any(valid_mask):
        msg = "nan_policy='omit' removed all data points (every row contained NaN/Inf)."
        raise ValueError(msg)

    omit_mask = valid_mask
    ydata = ydata[valid_mask]
    xdata = xdata[valid_mask] if xdims == 1 else xdata[:, valid_mask]
    if sigma_1d:
        sigma = sigma_arr[valid_mask]  # type: ignore[index]  # sigma_arr is not None whenever sigma_1d is True
    elif sigma_arr is not None and sigma_arr.ndim == 2:
        sigma = sigma_arr[np.ix_(valid_mask, valid_mask)]
        if not np.all(np.isfinite(sigma)):
            raise ValueError(
                "Sigma covariance matrix contains non-finite values in "
                "valid data rows/columns after NaN filtering",
            )

    return xdata, ydata, sigma, len(ydata), omit_mask


def _apply_padding_if_needed(
    xdata: np.ndarray,
    ydata: np.ndarray,
    xdims: int,
    m: int,
    len_diff: int,
    should_pad: bool,
    *,
    ctx: PrepareContext,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply padding to data if needed.

    Parameters
    ----------
    xdata : np.ndarray
        X data
    ydata : np.ndarray
        Y data
    xdims : int
        X data dimensionality
    m : int
        Data length
    len_diff : int
        Length difference
    should_pad : bool
        Whether padding is needed

    Returns
    -------
    xdata : np.ndarray
        Possibly padded X data
    ydata : np.ndarray
        Possibly padded Y data
    len_diff : int
        Updated length difference
    """
    if ctx.flength is not None and should_pad:
        if len_diff > 0:
            xdata, ydata = pad_fit_data(xdata, ydata, xdims, len_diff)
        elif len_diff < 0 and not ctx.use_dynamic_sizing:
            ctx.logger.debug(
                "Data size exceeds fixed length, JIT retracing may occur",
                data_size=m,
                flength=ctx.flength,
            )
    elif ctx.use_dynamic_sizing and ctx.flength is not None and len_diff < 0:
        len_diff = 0

    return xdata, ydata, len_diff


def prepare_problem(
    f: Callable,
    x: Any,
    y: Any,
    p0: np.ndarray | str | None = None,
    bounds: tuple = (-np.inf, np.inf),
    sigma: np.ndarray | None = None,
    data_mask: np.ndarray | None = None,
    check_finite: bool = True,
    nan_policy: str = "raise",
    method: str | None = None,
    solver: str = "auto",
    batch_size: int | None = None,
    kwargs: dict | None = None,
    ctx: PrepareContext = PrepareContext(),  # noqa: B008 (frozen, immutable default)
) -> PreparedProblem:
    """Prepare and validate inputs for curve fitting.

    ``kwargs`` is the caller's dict and is mutated in place: nothing is
    consumed, but method auto-selection (``method=None`` with
    ``ctx.enable_stability``) adds ``ftol``/``xtol``/``gtol``/``max_nfev``/
    ``x_scale`` defaults the caller did not set. ``nan_policy`` is passed
    explicitly; the caller reads it from its own ``kwargs``.

    Returns
    -------
    PreparedProblem
        See the module docstring for the step order and the field mapping
        from the legacy 13-tuple.
    """
    if kwargs is None:
        kwargs = {}
    xdata, ydata = x, y

    # Step 1: Determine parameter count and auto-estimate p0 if needed
    n, p0 = _determine_parameter_count(f, p0, xdata, ydata, ctx=ctx)

    # Step 2: Validate solver configuration
    _validate_solver_config(solver, batch_size)

    # Step 3: Log curve fit start
    # Check if bounds are provided (not infinite)
    has_bounds = False
    if isinstance(bounds, tuple) and len(bounds) == 2:
        lower_b, upper_b = bounds
        lower_arr = np.atleast_1d(lower_b)
        upper_arr = np.atleast_1d(upper_b)
        has_bounds = not (
            np.all(np.isneginf(lower_arr)) and np.all(np.isposinf(upper_arr))
        )

    ctx.logger.info(
        "Starting curve fit",
        n_params=n,
        n_data_points=len(ydata),
        method=method or "trf",
        solver=solver,
        batch_size=batch_size if solver == "minibatch" else None,
        has_bounds=has_bounds,
        dynamic_sizing=ctx.use_dynamic_sizing,
    )

    # Step 4: Prepare bounds and initial guess
    lb, ub, p0 = _prepare_bounds_and_initial_guess(bounds, n, p0)

    # Steps 5-7: Preprocess data (validation, conversion, length check,
    # NaN handling) BEFORE method/algorithm auto-selection below — auto
    # selection's data-driven heuristics (percentile/diff/ptp) must not
    # run against NaN/Inf-contaminated data that nan_policy='omit' asked
    # to have removed.
    xdata, ydata, m, xdims, sigma, omit_mask = _preprocess_input_data(
        f,
        xdata,
        ydata,
        p0,
        check_finite,
        nan_policy=nan_policy,
        # Pass the *normalized* (lb, ub) arrays (not the raw bounds tuple,
        # which may be scalar (-inf, inf)) so the validator's bounds checks
        # receive per-parameter arrays of the expected shape.
        bounds=(lb, ub),
        sigma=sigma,
        ctx=ctx,
    )

    # A user-supplied data_mask is sized to the ORIGINAL data length; if
    # nan_policy='omit' dropped rows, realign it the same way so it isn't
    # rejected by _setup_data_mask_and_padding's length check below.
    if data_mask is not None and omit_mask is not None:
        if len(data_mask) == len(omit_mask) and len(data_mask) != m:
            data_mask = np.asarray(data_mask)[omit_mask]

    # Step 8: Select optimization method (on cleaned data)
    method = _select_optimization_method(
        method,
        f,
        xdata,
        ydata,
        p0,
        bounds,
        kwargs,
        ctx=ctx,
    )

    # Step 9: Setup data mask and padding parameters
    data_mask, none_mask, len_diff, should_pad = _setup_data_mask_and_padding(
        data_mask,
        m,
        ctx=ctx,
    )

    # Step 10: Apply padding if needed
    xdata, ydata, len_diff = _apply_padding_if_needed(
        xdata,
        ydata,
        xdims,
        m,
        len_diff,
        should_pad,
        ctx=ctx,
    )

    return PreparedProblem(
        n=n,
        p0=p0,
        lb=lb,
        ub=ub,
        x=xdata,
        y=ydata,
        sigma=sigma,
        mask=data_mask,
        method=method,
        m_real=m,
        n_effective=int(np.sum(data_mask)),
        len_diff=len_diff,
        should_pad=should_pad,
        none_mask=none_mask,
    )
