"""Input validation for NLSQ optimization functions.

This module provides comprehensive input validation to catch errors early
and provide helpful error messages to users.
"""

import warnings
from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any

import numpy as np

from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax.numpy as jnp


class InputValidator:
    """Comprehensive input validation for curve fitting functions."""

    def __init__(self, fast_mode: bool = True):
        """Initialize the input validator.

        Parameters
        ----------
        fast_mode : bool, default True
            If True, skip expensive validation checks for better performance.
            If False, perform all validation checks.
        """
        self.fast_mode = fast_mode
        self._function_cache = {}  # Cache function test results

    def validate_curve_fit_inputs(
        self,
        f: Callable,
        xdata: Any,
        ydata: Any,
        p0: Any | None = None,
        bounds: tuple | None = None,
        sigma: Any | None = None,
        absolute_sigma: bool = True,
        check_finite: bool = True,
    ) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
        """Validate inputs for curve_fit function.

        Parameters
        ----------
        f : callable
            Model function to fit
        xdata : array_like
            Independent variable data
        ydata : array_like
            Dependent variable data
        p0 : array_like, optional
            Initial parameter guess
        bounds : tuple, optional
            Parameter bounds
        sigma : array_like, optional
            Uncertainties in ydata
        absolute_sigma : bool
            Whether sigma is absolute or relative
        check_finite : bool
            Whether to check for finite values

        Returns
        -------
        errors : list
            List of error messages (empty if no errors)
        warnings : list
            List of warning messages
        xdata_clean : np.ndarray
            Cleaned and validated xdata
        ydata_clean : np.ndarray
            Cleaned and validated ydata
        """
        errors = []
        warnings_list = []

        # 1. Handle tuple xdata (for multi-dimensional fitting)
        if isinstance(xdata, tuple):
            # xdata is a tuple of arrays (e.g., for 2D fitting)
            # Don't convert to a single array - keep as tuple
            try:
                n_points = len(xdata[0]) if len(xdata) > 0 else 0
                # Check all arrays in tuple have same length
                for i, x_arr in enumerate(xdata):
                    if len(x_arr) != n_points:
                        errors.append("All arrays in xdata tuple must have same length")
                        break
                warnings_list.append(f"xdata is tuple with {len(xdata)} arrays")
            except Exception as e:
                errors.append(f"Invalid xdata tuple: {e}")
                return errors, warnings_list, xdata, ydata
        else:
            # 1. Convert to numpy arrays and check types
            try:
                if not isinstance(xdata, (np.ndarray, jnp.ndarray)):
                    xdata = np.asarray(xdata)
                    warnings_list.append("xdata converted to numpy array")
            except Exception as e:
                errors.append(f"Cannot convert xdata to array: {e}")
                return errors, warnings_list, xdata, ydata

            # 2. Check dimensions
            if xdata.ndim == 0:
                errors.append("xdata must be at least 1-dimensional")

            # Handle 2D xdata (multiple independent variables)
            if xdata.ndim == 2:
                n_points = xdata.shape[0]
                n_vars = xdata.shape[1]
                warnings_list.append(f"xdata has {n_vars} independent variables")
            else:
                n_points = len(xdata) if hasattr(xdata, "__len__") else 1

        try:
            if not isinstance(ydata, (np.ndarray, jnp.ndarray)):
                ydata = np.asarray(ydata)
                warnings_list.append("ydata converted to numpy array")
        except Exception as e:
            errors.append(f"Cannot convert ydata to array: {e}")
            return errors, warnings_list, xdata, ydata

        if ydata.ndim == 0:
            errors.append("ydata must be at least 1-dimensional")

        # 3. Check shapes match
        if len(ydata) != n_points:
            errors.append(
                f"xdata ({n_points} points) and ydata ({len(ydata)} points) must have same length"
            )

        # 4. Check for minimum data points
        if n_points < 2:
            errors.append("Need at least 2 data points for fitting")

        # Estimate number of parameters
        n_params = 2  # Default estimate
        try:
            sig = signature(f)
            # Count parameters excluding x
            params = list(sig.parameters.keys())
            if params:
                n_params = len(params) - 1
        except Exception:
            if p0 is not None:
                n_params = len(p0)

        if n_points <= n_params:
            errors.append(
                f"Need more data points ({n_points}) than parameters ({n_params}) for fitting"
            )

        # 5. Check for degenerate cases
        if not isinstance(xdata, tuple):  # Only check for non-tuple xdata
            if hasattr(xdata, "ndim") and xdata.ndim == 1 and len(xdata) > 0:
                if np.all(xdata == xdata.flat[0]):
                    errors.append("All x values are identical - cannot fit")

                # Check for very small range
                x_range = np.ptp(xdata)
                if x_range < 1e-10 and x_range > 0:
                    warnings_list.append(
                        f"x data range is very small ({x_range:.2e}) - consider rescaling"
                    )

                # Check for very large range
                if x_range > 1e10:
                    warnings_list.append(
                        f"x data range is very large ({x_range:.2e}) - consider rescaling"
                    )

        # Check y data
        # Handle both JAX and NumPy arrays
        try:
            ydata_first = (
                ydata.flatten()[0] if hasattr(ydata, "flatten") else ydata.flat[0]
            )
            if np.all(ydata == ydata_first):
                warnings_list.append("All y values are identical - trivial fit")
        except Exception:
            # Skip this check if it fails
            pass

        y_range = np.ptp(ydata)
        if y_range < 1e-10 and y_range > 0:
            warnings_list.append(f"y data range is very small ({y_range:.2e})")

        # 6. Check for numerical issues if requested
        if check_finite:
            if isinstance(xdata, tuple):
                # Check each array in the tuple
                for i, x_arr in enumerate(xdata):
                    if not np.all(np.isfinite(x_arr)):
                        n_bad = np.sum(~np.isfinite(x_arr))
                        errors.append(f"xdata[{i}] contains {n_bad} NaN or Inf values")
            elif not np.all(np.isfinite(xdata)):
                n_bad = np.sum(~np.isfinite(xdata))
                errors.append(f"xdata contains {n_bad} NaN or Inf values")

            if not np.all(np.isfinite(ydata)):
                n_bad = np.sum(~np.isfinite(ydata))
                errors.append(f"ydata contains {n_bad} NaN or Inf values")

        # 7. Validate initial parameters
        if p0 is not None:
            try:
                p0 = np.asarray(p0)
                if len(p0) != n_params:
                    errors.append(
                        f"Initial guess p0 has {len(p0)} parameters, "
                        f"but function expects {n_params}"
                    )

                if not np.all(np.isfinite(p0)):
                    errors.append(
                        "Initial parameter guess p0 contains NaN or Inf values"
                    )

            except Exception as e:
                errors.append(f"Invalid initial parameter guess p0: {e}")

        # 8. Validate bounds if provided
        if bounds is not None:
            try:
                if len(bounds) != 2:
                    errors.append("bounds must be a 2-tuple of (lower, upper)")
                else:
                    lb, ub = bounds
                    if lb is not None and ub is not None:
                        lb = np.asarray(lb)
                        ub = np.asarray(ub)

                        if len(lb) != n_params or len(ub) != n_params:
                            errors.append(
                                f"bounds must have length {n_params} to match parameters"
                            )

                        if np.any(lb >= ub):
                            errors.append("Lower bounds must be less than upper bounds")

                        # Check if p0 is within bounds
                        if p0 is not None:
                            if np.any(p0 < lb) or np.any(p0 > ub):
                                warnings_list.append(
                                    "Initial guess p0 is outside bounds"
                                )

            except Exception as e:
                errors.append(f"Invalid bounds: {e}")

        # 9. Validate sigma if provided
        if sigma is not None:
            try:
                sigma = np.asarray(sigma)
                if sigma.shape != ydata.shape:
                    errors.append("sigma must have same shape as ydata")

                if np.any(sigma <= 0):
                    errors.append("sigma values must be positive")

                if not np.all(np.isfinite(sigma)):
                    errors.append("sigma contains NaN or Inf values")

            except Exception as e:
                errors.append(f"Invalid sigma: {e}")

        # 10. Check function can be called (skip in fast mode)
        if not self.fast_mode:
            try:
                # Cache function test results to avoid repeated calls
                func_id = id(f)
                if func_id not in self._function_cache:
                    if isinstance(xdata, tuple):
                        # For tuple xdata, sample from each array
                        test_x = tuple(arr[: min(10, len(arr))] for arr in xdata)
                        expected_len = min(10, len(xdata[0]))
                    else:
                        if hasattr(xdata, "ndim") and xdata.ndim > 1:
                            test_x = xdata[: min(10, len(xdata))]
                        else:
                            test_x = xdata[: min(10, len(xdata))]
                        expected_len = min(10, len(xdata))

                    if p0 is not None:
                        test_result = f(test_x, *p0)
                    else:
                        # Try with dummy parameters
                        dummy_params = np.ones(n_params)
                        test_result = f(test_x, *dummy_params)

                    # Cache the result
                    self._function_cache[func_id] = True

                    # Check output shape/length
                    if hasattr(test_result, "__len__"):
                        if len(test_result) != expected_len:
                            warnings_list.append(
                                f"Function output length {len(test_result)} doesn't match "
                                f"expected length {expected_len}"
                            )

            except Exception as e:
                errors.append(f"Cannot evaluate function: {e}")

        # 11. Data quality checks (skip in fast mode)
        if not self.fast_mode:
            if (
                not isinstance(xdata, tuple)
                and hasattr(xdata, "ndim")
                and xdata.ndim == 1
            ):
                # Check for duplicates
                unique_x = np.unique(xdata)
                if len(unique_x) < len(xdata):
                    n_dup = len(xdata) - len(unique_x)
                    warnings_list.append(f"xdata contains {n_dup} duplicate values")

            # Check for outliers in y (always check, regardless of xdata type)
            if len(ydata) > 10:
                q1, q3 = np.percentile(ydata, [25, 75])
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                n_outliers = np.sum((ydata < lower) | (ydata > upper))
                if n_outliers > 0:
                    warnings_list.append(
                        f"ydata may contain {n_outliers} outliers - "
                        "consider using robust loss function"
                    )

        # Return cleaned data
        # Keep tuples as tuples, convert arrays to numpy
        if not isinstance(xdata, tuple):
            xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)
        return errors, warnings_list, xdata, ydata

    def validate_least_squares_inputs(
        self,
        fun: Callable,
        x0: Any,
        bounds: tuple | None = None,
        method: str = "trf",
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        max_nfev: int | None = None,
    ) -> tuple[list[str], list[str], np.ndarray]:
        """Validate inputs for least_squares function.

        Parameters
        ----------
        fun : callable
            Residual function
        x0 : array_like
            Initial parameter guess
        bounds : tuple, optional
            Parameter bounds
        method : str
            Optimization method
        ftol : float
            Function tolerance
        xtol : float
            Parameter tolerance
        gtol : float
            Gradient tolerance
        max_nfev : int, optional
            Maximum function evaluations

        Returns
        -------
        errors : list
            List of error messages
        warnings : list
            List of warning messages
        x0_clean : np.ndarray
            Cleaned initial guess
        """
        errors = []
        warnings_list = []

        # Convert x0
        try:
            x0 = np.asarray(x0)
        except Exception as e:
            errors.append(f"Cannot convert x0 to array: {e}")
            return errors, warnings_list, x0

        # Check x0
        if x0.ndim != 1:
            errors.append("x0 must be 1-dimensional")

        if len(x0) == 0:
            errors.append("x0 cannot be empty")

        if not np.all(np.isfinite(x0)):
            errors.append("x0 contains NaN or Inf values")

        # Check method
        valid_methods = ["trf", "dogbox", "lm"]
        if method not in valid_methods:
            errors.append(f"method must be one of {valid_methods}, got {method}")

        # Check tolerances
        if ftol <= 0:
            errors.append(f"ftol must be positive, got {ftol}")
        if xtol <= 0:
            errors.append(f"xtol must be positive, got {xtol}")
        if gtol <= 0:
            errors.append(f"gtol must be positive, got {gtol}")

        if ftol < 1e-15:
            warnings_list.append(f"ftol={ftol} is very small, may not converge")
        if xtol < 1e-15:
            warnings_list.append(f"xtol={xtol} is very small, may not converge")

        # Check max_nfev
        if max_nfev is not None:
            if max_nfev <= 0:
                errors.append(f"max_nfev must be positive, got {max_nfev}")
            elif max_nfev < len(x0):
                warnings_list.append(
                    f"max_nfev={max_nfev} is less than number of parameters {len(x0)}"
                )

        # Check bounds
        if bounds is not None:
            if method == "lm":
                errors.append("Levenberg-Marquardt method does not support bounds")
            else:
                try:
                    lb, ub = bounds
                    lb = np.asarray(lb)
                    ub = np.asarray(ub)

                    if len(lb) != len(x0) or len(ub) != len(x0):
                        errors.append("bounds must have same length as x0")

                    if np.any(lb >= ub):
                        errors.append("Lower bounds must be less than upper bounds")

                    # Check x0 within bounds
                    if np.any(x0 < lb) or np.any(x0 > ub):
                        errors.append("Initial guess x0 is outside bounds")

                except Exception as e:
                    errors.append(f"Invalid bounds: {e}")

        # Test function
        try:
            result = fun(x0)
            result = np.asarray(result)

            if result.ndim != 1:
                errors.append("Function must return 1-dimensional residuals")

            if not np.all(np.isfinite(result)):
                warnings_list.append("Function returns NaN or Inf at initial guess")

        except Exception as e:
            errors.append(f"Cannot evaluate function at x0: {e}")

        return errors, warnings_list, x0


def validate_inputs(validation_type: str = "curve_fit"):
    """Decorator for automatic input validation.

    Parameters
    ----------
    validation_type : str
        Type of validation to perform ('curve_fit' or 'least_squares')

    Returns
    -------
    decorator : function
        Decorator function that validates inputs
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            validator = InputValidator()

            if validation_type == "curve_fit":
                # Extract arguments
                if len(args) < 3:
                    raise ValueError(
                        "curve_fit requires at least 3 arguments (f, xdata, ydata)"
                    )

                f, xdata, ydata = args[:3]
                remaining_args = args[3:]

                # Get optional arguments
                p0 = kwargs.get("p0")
                bounds = kwargs.get("bounds")
                sigma = kwargs.get("sigma")
                absolute_sigma = kwargs.get("absolute_sigma", True)
                check_finite = kwargs.get("check_finite", True)

                # Validate
                errors, warnings_list, xdata_clean, ydata_clean = (
                    validator.validate_curve_fit_inputs(
                        f, xdata, ydata, p0, bounds, sigma, absolute_sigma, check_finite
                    )
                )

                # Handle errors and warnings
                if errors:
                    raise ValueError(f"Input validation failed: {'; '.join(errors)}")

                for warning in warnings_list:
                    warnings.warn(warning, UserWarning, stacklevel=2)

                # Replace with cleaned data
                args = (f, xdata_clean, ydata_clean, *remaining_args)

            elif validation_type == "least_squares":
                # Extract arguments
                if len(args) < 2:
                    raise ValueError(
                        "least_squares requires at least 2 arguments (fun, x0)"
                    )

                fun, x0 = args[:2]
                remaining_args = args[2:]

                # Get optional arguments
                bounds = kwargs.get("bounds")
                method = kwargs.get("method", "trf")
                ftol = kwargs.get("ftol", 1e-8)
                xtol = kwargs.get("xtol", 1e-8)
                gtol = kwargs.get("gtol", 1e-8)
                max_nfev = kwargs.get("max_nfev")

                # Validate
                errors, warnings_list, x0_clean = (
                    validator.validate_least_squares_inputs(
                        fun, x0, bounds, method, ftol, xtol, gtol, max_nfev
                    )
                )

                # Handle errors and warnings
                if errors:
                    raise ValueError(f"Input validation failed: {'; '.join(errors)}")

                for warning in warnings_list:
                    warnings.warn(warning, UserWarning, stacklevel=2)

                # Replace with cleaned data
                args = (fun, x0_clean, *remaining_args)

            else:
                raise ValueError(f"Unknown validation type: {validation_type}")

            # Call original function
            return func(*args, **kwargs)

        return wrapper

    return decorator
