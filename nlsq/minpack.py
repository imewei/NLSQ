import time
import warnings
from collections.abc import Callable
from inspect import signature

import numpy as np

# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import cholesky as jax_cholesky
from jax.scipy.linalg import svd as jax_svd

from nlsq._optimize import OptimizeWarning
from nlsq.algorithm_selector import auto_select_algorithm
from nlsq.common_scipy import EPS
from nlsq.diagnostics import OptimizationDiagnostics
from nlsq.least_squares import LeastSquares, prepare_bounds
from nlsq.logging import get_logger
from nlsq.memory_manager import get_memory_manager
from nlsq.recovery import OptimizationRecovery
from nlsq.stability import NumericalStabilityGuard
from nlsq.validators import InputValidator

__all__ = ["CurveFit", "curve_fit"]


def curve_fit(f, xdata, ydata, *args, **kwargs):
    """
    Use nonlinear least squares to fit a function to data with GPU/TPU acceleration.

    This is the main user-facing function that provides a drop-in replacement for
    `scipy.optimize.curve_fit` with GPU/TPU acceleration via JAX. The function
    automatically handles JAX JIT compilation, double precision configuration,
    and optimization algorithm selection.

    Parameters
    ----------
    f : callable
        The model function f(x, \\*popt) -> y. Must be JAX-compatible, meaning it should
        use `jax.numpy` instead of `numpy` for mathematical operations to enable
        GPU acceleration and automatic differentiation.
    xdata : array_like
        The independent variable where the data is measured.
    ydata : array_like
        The dependent data, nominally ``f(xdata, *popt)``.
    *args, **kwargs
        Additional arguments passed to CurveFit.curve_fit method.

    Returns
    -------
    popt : ndarray
        Optimal values for the parameters.
    pcov : ndarray
        The estimated covariance of popt.

    Notes
    -----
    This function creates a CurveFit instance internally and calls its curve_fit method.
    For multiple fits with the same function signature, consider creating a CurveFit
    instance directly to benefit from JAX compilation caching.

    See Also
    --------
    CurveFit.curve_fit : The underlying method with full parameter documentation
    curve_fit_large : For datasets with millions of points requiring special handling

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>>
    >>> def exponential(x, a, b):
    ...     return a * jnp.exp(-b * x)
    >>>
    >>> x = np.linspace(0, 4, 50)
    >>> y = 2.5 * np.exp(-1.3 * x) + 0.1 * np.random.normal(size=len(x))
    >>> popt, _pcov = curve_fit(exponential, x, y, p0=[2, 1])
    """
    # Extract CurveFit constructor parameters from kwargs
    flength = kwargs.pop("flength", None)
    use_dynamic_sizing = kwargs.pop("use_dynamic_sizing", False)

    # Create CurveFit instance with appropriate parameters
    jcf = CurveFit(flength=flength, use_dynamic_sizing=use_dynamic_sizing)
    result = jcf.curve_fit(f, xdata, ydata, *args, **kwargs)
    # Always return exactly 2 values for SciPy compatibility
    # Extract only popt and pcov regardless of what internal method returns
    if isinstance(result, tuple):
        if len(result) >= 2:
            popt, _pcov = result[0], result[1]
        else:
            raise RuntimeError("Unexpected result format from curve_fit")
    else:
        raise RuntimeError("Unexpected result format from curve_fit")
    return popt, _pcov


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


class CurveFit:
    """Main class for nonlinear least squares curve fitting with JAX acceleration.

    This class provides the core curve fitting functionality with JAX JIT compilation,
    automatic differentiation for Jacobian computation, and multiple optimization
    algorithms. It handles data preprocessing, optimization algorithm selection,
    and covariance matrix computation.

    The class maintains compiled versions of fitting functions to avoid recompilation
    overhead when fitting multiple datasets with the same function signature.

    Attributes
    ----------
    flength : float or None
        Fixed data length for input padding to avoid JAX retracing.
    use_dynamic_sizing : bool
        Whether to use dynamic sizing instead of fixed padding.
    logger : Logger
        Internal logger for debugging and performance monitoring.

    Methods
    -------
    curve_fit : Main fitting method
    create_sigma_transform_funcs : Internal method for sigma transformation setup
    """

    def __init__(
        self,
        flength: float | None = None,
        use_dynamic_sizing: bool = False,
        enable_stability: bool = False,
        enable_recovery: bool = False,
        enable_overflow_check: bool = False,
    ):
        """Initialize CurveFit instance.

        Parameters
        ----------
        flength : float, optional
            Fixed data length for JAX compilation. Input data is padded to this length
            to avoid recompilation when fitting datasets of different sizes. If None,
            no padding is applied and each dataset size triggers recompilation.
            Ignored when use_dynamic_sizing=True for large datasets.

        use_dynamic_sizing : bool, default False
            Enable dynamic sizing to reduce memory usage. When True, padding is only
            applied when data size is smaller than flength. For large datasets,
            uses actual size to prevent excessive memory allocation. Default False
            maintains backward compatibility with fixed padding behavior.

        enable_stability : bool, default False
            Enable numerical stability checks and fixes (validation, algorithm selection).
            Note: This does NOT include overflow checking which adds overhead.

        enable_recovery : bool, default False
            Enable automatic recovery from optimization failures.

        enable_overflow_check : bool, default False
            Enable overflow/underflow checking in function evaluations. This adds
            ~30% overhead so it's separate from other stability features.

        Notes
        -----
        Fixed length compilation trades memory usage for compilation speed:
        - flength=None: Minimal memory, recompiles for each dataset size
        - flength=large_value: Higher memory, avoids recompilation
        - use_dynamic_sizing=True: Balanced approach for mixed dataset sizes
        """
        self.flength = flength
        self.use_dynamic_sizing = use_dynamic_sizing
        self.logger = get_logger("curve_fit")
        self.create_sigma_transform_funcs()
        self.create_covariance_svd()
        self.ls = LeastSquares()

        # Initialize stability and recovery systems
        self.enable_stability = enable_stability
        self.enable_recovery = enable_recovery
        self.enable_overflow_check = enable_overflow_check

        if enable_stability:
            self.stability_guard = NumericalStabilityGuard()
            # Use fast validation mode by default for performance
            self.validator = InputValidator(fast_mode=True)
            self.memory_manager = get_memory_manager()

        if enable_recovery:
            self.recovery = OptimizationRecovery()
            self.diagnostics = OptimizationDiagnostics()

    def update_flength(self, flength: float):
        """Set the fixed input data length.

        Parameters
        ----------
        flength : float
            The fixed input data length.
        """
        self.flength = flength

    def create_sigma_transform_funcs(self):
        """Create JIT-compiled sigma transform functions.

        This function creates two JIT-compiled functions: `sigma_transform1d` and
        `sigma_transform2d`, which are used to compute the sigma transform for 1D
        and 2D data, respectively. The functions are stored as attributes of the
        object on which the method is called.
        """

        @jit
        def sigma_transform1d(
            sigma: jnp.ndarray, data_mask: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute the sigma transform for 1D data.

            Parameters
            ----------
            sigma : jnp.ndarray
                The standard deviation of the data.
            data_mask : jnp.ndarray
                A binary mask indicating which data points to use in the fit.

            Returns
            -------
            jnp.ndarray
                The sigma transform for the data.
            """
            transform = 1.0 / sigma
            return transform

        @jit
        def sigma_transform2d(
            sigma: jnp.ndarray, data_mask: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute the sigma transform for 2D data.

            Parameters
            ----------
            sigma : jnp.ndarray
                The standard deviation of the data.
            data_mask : jnp.ndarray
                A binary mask indicating which data points to use in the fit.

            Returns
            -------
            jnp.ndarray
                The sigma transform for the data.
            """
            sigma = jnp.asarray(sigma)
            transform = jax_cholesky(sigma, lower=True)
            return transform

        self.sigma_transform1d = sigma_transform1d
        self.sigma_transform2d = sigma_transform2d
        """For fixed input arrays we need to pad the actual data to match the
        fixed input array size"""

    def create_covariance_svd(self):
        """Create JIT-compiled SVD function for covariance computation."""

        @jit
        def covariance_svd(jac):
            _, s, VT = jax_svd(jac, full_matrices=False)
            return s, VT

        self.covariance_svd = covariance_svd

    def _select_tr_solver(
        self, solver: str, m: int, n: int, batch_size: int | None = None
    ) -> str | None:
        """Select appropriate trust region solver based on solver type and problem size.

        Parameters
        ----------
        solver : str
            Requested solver type
        m : int
            Number of data points
        n : int
            Number of parameters
        batch_size : int, optional
            Batch size for minibatch processing

        Returns
        -------
        str or None
            Trust region solver to use, or None to use default
        """
        if solver == "auto":
            # Auto-select based on problem size
            if m * n < 10000:  # Small problems
                return "exact"  # Use SVD-based exact solver
            else:  # Large problems
                return "lsmr"  # Use iterative LSMR solver
        elif solver == "svd":
            return "exact"  # SVD-based exact solver
        elif solver == "cg":
            return "lsmr"  # LSMR is the closest to CG in current implementation
        elif solver == "lsqr":
            return "lsmr"  # Direct mapping
        elif solver == "minibatch":
            # For minibatch, we'll use lsmr but need to handle batching separately
            # This is a placeholder - full minibatch implementation would require
            # more substantial changes to the optimization loop
            self.logger.warning(
                "Minibatch solver not fully implemented yet. Using LSMR solver.",
                requested_batch_size=batch_size,
            )
            return "lsmr"
        else:
            return None  # Use default

    def pad_fit_data(
        self, xdata: np.ndarray, ydata: np.ndarray, xdims: int, len_diff: int
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
            xpad = EPS * np.ones([xdims, len_diff])
            xdata = np.concatenate([xdata, xpad], axis=1)
        else:
            xpad = EPS * np.ones([len_diff])
            xdata = np.concatenate([xdata, xpad])
        ypad = EPS * np.ones([len_diff])
        ydata = np.concatenate([ydata, ypad])
        return xdata, ydata

    def curve_fit(
        self,
        f: Callable,
        xdata: np.ndarray | tuple[np.ndarray],
        ydata: np.ndarray,
        p0: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        absolute_sigma: bool = False,
        check_finite: bool = True,
        bounds: tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
        method: str | None = None,
        solver: str = "auto",
        batch_size: int | None = None,
        jac: Callable | None = None,
        data_mask: np.ndarray | None = None,
        timeit: bool = False,
        return_eval: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Use non-linear least squares to fit a function, f, to data.
        Assumes ``ydata = f(xdata, \\*params) + eps``.

        Parameters
        ----------
        f : callable
            The model function, f(x, ...). It must take the independent
            variable as the first argument and the parameters to fit as
            separate remaining arguments.
        xdata : array_like or object
            The independent variable where the data is measured.
            Should usually be an M-length sequence or an (k,M)-shaped array for
            functions with k predictors, but can actually be any object.
        ydata : array_like
            The dependent data, a length M array - nominally ``f(xdata, ...)``.
        p0 : array_like, optional
            Initial guess for the parameters (length N). If None, then the
            initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a
            ValueError is raised).
        sigma : None or M-length sequence or MxM array, optional
            Determines the uncertainty in `ydata`. If we define residuals as
            ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
            depends on its number of dimensions:
            - A 1-D `sigma` should contain values of standard deviations of
            errors in `ydata`. In this case, the optimized function is
            ``chisq = sum((r / sigma) ** 2)``.
            - A 2-D `sigma` should contain the covariance matrix of
            errors in `ydata`. In this case, the optimized function is
            ``chisq = r.T @ inv(sigma) @ r``.
            .. versionadded:: 0.19

            None (default) is equivalent of 1-D `sigma` filled with ones.
        absolute_sigma : bool, optional
            If True, `sigma` is used in an absolute sense and the estimated parameter
            covariance `pcov` reflects these absolute values.
            If False (default), only the relative magnitudes of the `sigma` values matter.
            The returned parameter covariance matrix `pcov` is based on scaling
            `sigma` by a constant factor. This constant is set by demanding that the
            reduced `chisq` for the optimal parameters `popt` when using the
            *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
            match the sample variance of the residuals after the fit. Default is False.
            Mathematically,
            ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
        check_finite : bool, optional
            If True, check that the input arrays do not contain nans of infs,
            and raise a ValueError if they do. Setting this parameter to
            False may silently produce nonsensical results if the input arrays
            do contain nans. Default is True.
        bounds : 2-tuple of array_like, optional
            Lower and upper bounds on parameters. Defaults to no bounds.
            Each element of the tuple must be either an array with the length equal
            to the number of parameters, or a scalar (in which case the bound is
            taken to be the same for all parameters). Use ``np.inf`` with an
            appropriate sign to disable bounds on all or some parameters.
            .. versionadded:: 0.17
        method : {'trf'}, optional
            Method to use for optimization. See `least_squares` for more details.
            Currently only 'trf' is implemented.
            .. versionadded:: 0.17
        solver : {'auto', 'svd', 'cg', 'lsqr', 'minibatch'}, optional
            Solver method for handling large datasets and different problem types:
            - 'auto' (default): Automatically selects the best solver based on problem size
            - 'svd': Uses SVD decomposition (good for small to medium datasets)
            - 'cg': Uses conjugate gradient method (memory efficient for large problems)
            - 'lsqr': Uses LSQR iterative solver (good for sparse problems)
            - 'minibatch': Processes data in batches (for very large datasets)
        batch_size : int, optional
            Batch size for minibatch solver. Only used when solver='minibatch'.
            If None and minibatch solver is selected, a reasonable default based
            on data size will be chosen.
        jac : callable, string or None, optional
            Function with signature ``jac(x, ...)`` which computes the Jacobian
            matrix of the model function with respect to parameters as a dense
            array_like structure. It will be scaled according to provided `sigma`.
            If None (default), the Jacobian will be determined using JAX's automatic
            differentiation (AD) capabilities. We recommend not using an analytical
            Jacobian, as it is usually faster to use AD.
        kwargs
            Keyword arguments passed to `leastsq` for ``method='lm'`` or
            `least_squares` otherwise.

        Returns
        -------
        popt : array
            Optimal values for the parameters so that the sum of the squared
            residuals of ``f(xdata, *popt) - ydata`` is minimized.
        pcov : 2-D array
            The estimated covariance of popt. The diagonals provide the variance
            of the parameter estimate. To compute one standard deviation errors
            on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
            How the `sigma` parameter affects the estimated covariance
            depends on `absolute_sigma` argument, as described above.
            If the Jacobian matrix at the solution doesn't have a full rank, then
            'lm' method returns a matrix filled with ``np.inf``, on the other hand
            'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
            the covariance matrix.

        Raises
        ------
        ValueError
            if either `ydata` or `xdata` contain NaNs, or if incompatible options
            are used.
        RuntimeError
            if the least-squares minimization fails.
        OptimizeWarning
            if covariance of the parameters can not be estimated.
        See Also
        --------
        least_squares : Minimize the sum of squares of nonlinear functions.

        Notes
        -----
        Refer to the docstring of `least_squares` for more information.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import jax.numpy as jnp
        >>> from jaxfit import CurveFit
        >>> def func(x, a, b, c):
        ...     return a * jnp.exp(-b * x) + c
        Define the data to be fit with some noise:
        >>> xdata = np.linspace(0, 4, 50)
        >>> y = func(xdata, 2.5, 1.3, 0.5)
        >>> rng = np.random.default_rng()
        >>> y_noise = 0.2 * rng.normal(size=xdata.size)
        >>> ydata = y + y_noise
        >>> plt.plot(xdata, ydata, 'b-', label='data')
        Fit for the parameters a, b, c of the function `func`:
        >>> cf = CurveFit()
        >>> popt, _pcov = cf.curve_fit(func, xdata, ydata)
        >>> popt
        array([2.56274217, 1.37268521, 0.47427475])
        >>> plt.plot(xdata, func(xdata, *popt), 'r-',
        ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        Constrain the optimization to the region of ``0 <= a <= 3``,
        ``0 <= b <= 1`` and ``0 <= c <= 0.5``:
        >>> cf = CurveFit()
        >>> popt, _pcov = cf.curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
        >>> popt
        array([2.43736712, 1.        , 0.34463856])
        >>> plt.plot(xdata, func(xdata, *popt), 'g--',
        ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.legend()
        >>> plt.show()
        """

        if p0 is None:
            # determine number of parameters by inspecting the function
            sig = signature(f)
            args = sig.parameters
            if len(args) < 2:
                raise ValueError("Unable to determine number of fit parameters.")
            n = len(args) - 1
        else:
            p0 = np.atleast_1d(p0)
            n = p0.size

        # Validate solver parameter
        valid_solvers = {"auto", "svd", "cg", "lsqr", "minibatch"}
        if solver not in valid_solvers:
            raise ValueError(
                f"Invalid solver '{solver}'. Must be one of {valid_solvers}."
            )

        # Validate batch_size if minibatch solver is used
        if solver == "minibatch" and batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when using minibatch solver.")

        # Log curve fit start
        self.logger.info(
            "Starting curve fit",
            n_params=n,
            n_data_points=len(ydata),
            method=method if method else "trf",
            solver=solver,
            batch_size=batch_size if solver == "minibatch" else None,
            has_bounds=bounds != (-np.inf, np.inf),
            dynamic_sizing=self.use_dynamic_sizing,
        )

        lb, ub = prepare_bounds(bounds, n)
        if p0 is None:
            p0 = _initialize_feasible(lb, ub)

        # Auto-select algorithm if stability is enabled and method not specified
        if method is None:
            if self.enable_stability:
                # Use intelligent algorithm selection
                recommendations = auto_select_algorithm(f, xdata, ydata, p0, bounds)
                method = recommendations["algorithm"]
                self.logger.info(
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

        # Validate and sanitize inputs if stability checks are enabled
        if self.enable_stability:
            try:
                errors, warnings_list, xdata_clean, ydata_clean = (
                    self.validator.validate_curve_fit_inputs(f, xdata, ydata, p0)
                )

                # Handle errors
                if errors:
                    error_msg = f"Input validation failed: {'; '.join(errors)}"
                    self.logger.error("Input validation failed", error=error_msg)
                    raise ValueError(error_msg)

                # Handle warnings
                for warning in warnings_list:
                    self.logger.warning("Input validation warning", warning=warning)

                # Use cleaned data
                xdata = xdata_clean
                ydata = ydata_clean

            except ValueError as e:
                if "too many values to unpack" not in str(e):
                    self.logger.error("Input validation failed", error=str(e))
                raise

        # NaNs cannot be handled
        if check_finite:
            ydata = np.asarray_chkfinite(ydata, float)
        else:
            ydata = np.asarray(ydata, float)

        # Handle JAX arrays, NumPy arrays, lists, and tuples
        if hasattr(xdata, "__array__") or isinstance(
            xdata, (list, tuple, np.ndarray, jnp.ndarray)
        ):
            # should we be able to pass jax arrays
            # `xdata` is passed straight to the user-defined `f`, so allow
            # non-array_like `xdata`.
            if check_finite:
                xdata = np.asarray_chkfinite(xdata, float)
            else:
                xdata = np.asarray(xdata, float)
        else:
            raise ValueError("X needs arrays")

        if ydata.size == 0:
            raise ValueError("`ydata` must not be empty!")

        m = len(ydata)
        xdims = xdata.ndim
        xlen = len(xdata) if xdims == 1 else len(xdata[0])
        if xlen != m:
            raise ValueError("X and Y data lengths dont match")

        none_mask = data_mask is None

        # Handle dynamic sizing vs fixed length padding
        should_pad = False
        len_diff = 0

        if self.flength is not None:
            len_diff = self.flength - m
            if self.use_dynamic_sizing:
                # With dynamic sizing, only pad if data is smaller than flength
                # and avoid padding for large datasets to save memory
                should_pad = len_diff > 0
            else:
                # Original behavior: always try to pad to flength
                should_pad = len_diff >= 0

            if data_mask is not None:
                if len(data_mask) != m:
                    raise ValueError("Data mask doesn't match data lengths.")
            else:
                data_mask = np.ones(m, dtype=bool)
                if should_pad and len_diff > 0:
                    data_mask = np.concatenate(
                        [data_mask, np.zeros(len_diff, dtype=bool)]
                    )
        else:
            data_mask = np.ones(m, dtype=bool)

        # Apply padding if needed
        if self.flength is not None and should_pad:
            if len_diff > 0:
                xdata, ydata = self.pad_fit_data(xdata, ydata, xdims, len_diff)
            elif len_diff < 0 and not self.use_dynamic_sizing:
                # Data length greater than fixed length - retracing will occur
                # Only warn if not using dynamic sizing
                self.logger.debug(
                    "Data size exceeds fixed length, JIT retracing may occur",
                    data_size=m,
                    flength=self.flength,
                )
        elif self.use_dynamic_sizing and self.flength is not None and len_diff < 0:
            # With dynamic sizing, reset len_diff to 0 for large datasets
            len_diff = 0

            # Determine type of sigma
        if sigma is not None:
            if not isinstance(sigma, np.ndarray):
                raise ValueError("Sigma must be numpy array.")
            # if 1-D, sigma are errors, define transform = 1/sigma
            ysize = ydata.size - len_diff
            if sigma.shape == (ysize,):
                if len_diff > 0:
                    sigma = np.concatenate([sigma, np.ones([len_diff])])
                transform = self.sigma_transform1d(sigma, data_mask)
            # if 2-D, sigma is the covariance matrix,
            # define transform = L such that L L^T = C
            elif sigma.shape == (ysize, ysize):
                try:
                    if len_diff >= 0:
                        sigma_padded = np.identity(m + len_diff)
                        sigma_padded[:m, :m] = sigma
                        sigma = sigma_padded
                    # scipy.linalg.cholesky requires lower=True to return L L^T = A
                    transform = self.sigma_transform2d(sigma, data_mask)
                except (np.linalg.LinAlgError, ValueError) as e:
                    # Check eigenvalues to provide more informative error
                    try:
                        eigenvalues = np.linalg.eigvalsh(sigma[:ysize, :ysize])
                        min_eig = np.min(eigenvalues)
                        if min_eig <= 0:
                            raise ValueError(
                                f"Covariance matrix `sigma` is not positive definite. "
                                f"Minimum eigenvalue: {min_eig:.6e}. "
                                "All eigenvalues must be positive."
                            ) from e
                    except Exception:
                        # If eigenvalue check fails, provide generic error
                        pass
                    raise ValueError(
                        "Failed to compute Cholesky decomposition of `sigma`. "
                        "The covariance matrix must be symmetric and positive definite."
                    ) from e
            else:
                raise ValueError("`sigma` has incorrect shape.")
        else:
            transform = None

        if "args" in kwargs:
            # The specification for the model function `f` does not support
            # additional arguments. Refer to the `curve_fit` docstring for
            # acceptable call signatures of `f`.
            raise ValueError("'args' is not a supported keyword argument.")

        if "max_nfev" not in kwargs:
            kwargs["max_nfev"] = kwargs.pop("maxfev", None)

        # Determine the appropriate solver and configure tr_solver
        tr_solver = self._select_tr_solver(solver, m, n, batch_size)
        if tr_solver is not None:
            kwargs["tr_solver"] = tr_solver

        # Handle minibatch processing if requested
        if solver == "minibatch":
            # Set reasonable default batch size if not provided
            if batch_size is None:
                batch_size = min(1000, max(100, m // 10))  # 10% of data, clamped
                self.logger.debug(f"Using default batch size: {batch_size}")

            # For now, just log that minibatch would be used
            # Full implementation would require batching the optimization
            self.logger.info(
                "Minibatch processing requested",
                batch_size=batch_size,
                n_batches=m // batch_size + (1 if m % batch_size > 0 else 0),
            )

        st = time.time()
        if timeit:
            # Use jnp.asarray for efficient conversion without unnecessary copying
            jnp_xdata = jnp.asarray(xdata).block_until_ready()
            jnp_ydata = jnp.asarray(ydata).block_until_ready()
        else:
            jnp_xdata = jnp.asarray(xdata)
            jnp_ydata = jnp.asarray(ydata)
        ctime = time.time() - st

        jnp_data_mask = jnp.array(data_mask, dtype=bool)

        # Check memory requirements if stability is enabled
        if self.enable_stability:
            memory_required = self.memory_manager.predict_memory_requirement(
                m, n, method
            )
            is_available, msg = self.memory_manager.check_memory_availability(
                memory_required
            )
            if not is_available:
                self.logger.warning("Memory constraint detected", message=msg)
                # Switch to memory-efficient solver
                kwargs["tr_solver"] = "lsmr"

        # Start curve fit timer and call least squares
        with self.logger.timer("curve_fit"):
            self.logger.debug(
                "Calling least squares solver",
                has_sigma=sigma is not None,
                has_jacobian=jac is not None,
            )

            # Create wrapper for overflow checking if enabled
            # Note: This is separate from stability to avoid performance overhead
            if self.enable_overflow_check:
                original_f = f

                # Use a more efficient overflow check
                def stable_f(x, *params):
                    result = original_f(x, *params)
                    # Only apply clipping when needed to reduce overhead
                    # Check max/min values first (faster than checking all elements)
                    max_val = jnp.max(jnp.abs(result))
                    # Only clip if we have extreme values
                    result = jnp.where(
                        max_val > 1e8,  # Only check/clip for very large values
                        jnp.clip(result, -1e10, 1e10),
                        result,
                    )
                    return result

                f_to_use = stable_f
            else:
                f_to_use = f

            try:
                res = self.ls.least_squares(
                    f_to_use,
                    p0,
                    jac=jac,
                    xdata=jnp_xdata,
                    ydata=jnp_ydata,
                    data_mask=jnp_data_mask,
                    transform=transform,
                    bounds=bounds,
                    method=method,
                    timeit=timeit,
                    **kwargs,
                )
            except Exception as e:
                if self.enable_recovery:
                    self.logger.warning(
                        "Optimization failed, attempting recovery", error=str(e)
                    )
                    # Prepare recovery state
                    recovery_state = {
                        "params": p0,
                        "xdata": xdata,
                        "ydata": ydata,
                        "method": method if method is not None else "trf",
                        "bounds": bounds,
                    }

                    # Attempt recovery
                    success, result = self.recovery.recover_from_failure(
                        "optimization_error",
                        recovery_state,
                        lambda **state: self.ls.least_squares(
                            f_to_use,
                            state["params"],
                            jac=jac,
                            xdata=jnp.asarray(state["xdata"]),
                            ydata=jnp.asarray(state["ydata"]),
                            data_mask=jnp_data_mask,
                            transform=transform,
                            bounds=state["bounds"],
                            method=state["method"],
                            timeit=timeit,
                            **kwargs,
                        ),
                    )

                    if success:
                        res = result
                    else:
                        raise RuntimeError(
                            f"Optimization failed and recovery unsuccessful: {e}"
                        ) from e
                else:
                    raise

        if not res.success:
            self.logger.error(
                "Optimization failed", reason=res.message, status=res.status
            )
            raise RuntimeError("Optimal parameters not found: " + res.message)

        popt = res.x
        self.logger.debug(
            "Optimization succeeded",
            final_cost=res.cost,
            nfev=res.nfev,
            optimality=getattr(res, "optimality", None),
        )

        st = time.time()
        # ysize = len(res.fun)
        ysize = m
        cost = 2 * res.cost  # res.cost is half sum of squares!

        # Do Moore-Penrose inverse discarding zero singular values.
        # _, s, VT = svd(res.jac, full_matrices=False)
        outputs = self.covariance_svd(res.jac)
        # Convert JAX arrays to NumPy more efficiently using np.asarray
        s, VT = (np.asarray(output) for output in outputs)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[: s.size]
        pcov = np.dot(VT.T / s**2, VT)
        return_full = False

        warn_cov = False
        if pcov is None:
            # indeterminate covariance
            pcov = np.zeros((len(popt), len(popt)), dtype=float)
            pcov.fill(np.inf)
            warn_cov = True
        elif not absolute_sigma:
            if ysize > p0.size:
                s_sq = cost / (ysize - p0.size)
                pcov = pcov * s_sq
            else:
                pcov.fill(np.inf)
                warn_cov = True

        if warn_cov:
            self.logger.warning(
                "Covariance could not be estimated",
                reason="insufficient_data" if ysize <= p0.size else "singular_jacobian",
            )
            warnings.warn(
                "Covariance of the parameters could not be estimated",
                stacklevel=2,
                category=OptimizeWarning,
            )

        # Assign final covariance matrix
        _pcov = pcov

        # self.res = res
        post_time = time.time() - st

        # Log curve fit completion
        total_time = self.logger.timers.get("curve_fit", 0)
        self.logger.info(
            "Curve fit completed",
            total_time=total_time,
            final_cost=cost,
            covariance_warning=warn_cov,
        )

        if return_eval:
            feval = f(jnp_xdata, *popt)
            feval = np.array(feval)
            if none_mask:
                # data_mask = np.ndarray.astype(data_mask, bool)
                return popt, _pcov, feval[data_mask]
            else:
                return popt, _pcov, feval
        else:
            # lower GPU memory usage
            res.pop("jac")
            res.pop("fun")

        if return_full:
            raise RuntimeError("Return full only works for LM")
            # return popt, _pcov, infodict, errmsg, ier
        elif timeit:
            return popt, _pcov, res, post_time, ctime
        else:
            return popt, _pcov
