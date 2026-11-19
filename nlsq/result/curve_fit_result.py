"""Enhanced result objects for curve fitting with convenience methods."""

import logging
import warnings
from typing import Any, cast

import numpy as np

from nlsq.result.optimize_result import OptimizeResult
from nlsq.result.optimize_warning import OptimizeWarning

_logger = logging.getLogger(__name__)


def _check_alpha(alpha) -> float:
    """Confidence level must be finite and strictly inside (0, 1)."""
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError(
            f"alpha is the confidence level and must be in (0, 1), got {alpha!r}"
        )
    return a


def _rel_diff(d1, d2) -> float:
    with np.errstate(all="ignore"):
        return np.linalg.norm(d1 - d2) / max(np.linalg.norm(d1), np.finfo(float).tiny)


class CurveFitResult(OptimizeResult):
    """Enhanced curve fitting result with statistical properties and visualization.

    This class extends OptimizeResult with convenience methods for statistical
    analysis and visualization. It maintains backward compatibility by supporting
    tuple unpacking: ``popt, pcov = curve_fit(...)``.

    The result can be used in two ways:

    1. Tuple unpacking (backward compatible)::

        popt, pcov = curve_fit(model, x, y)

    2. Enhanced result object::

        result = curve_fit(model, x, y)
        print(f"R² = {result.r_squared:.4f}")
        print(f"RMSE = {result.rmse:.4f}")
        result.plot()
        result.summary()

    Unlike a plain ``dict``, iterating a result (``iter(result)``,
    ``tuple(result)``, tuple unpacking) yields ``(popt, pcov)``, and integer
    indexing (``result[0]``) selects from that pair. ``keys()``, ``items()``,
    ``len()``, ``in`` and string indexing still describe the dict keys.

    A result is a snapshot: do not mutate ``xdata``, ``ydata``, ``sigma`` or
    ``fun`` arrays in place. Reassigning an attribute (``result["x"] = ...``)
    invalidates the cached ``predictions``/``residuals``; in-place edits are not
    detected.

    Derived properties (``predictions``, ``residuals``, ``r_squared`` ...) are
    read-only and assigning to them raises ``AttributeError``; ``popt``,
    ``pcov`` and ``diagnostics`` can be assigned and write through to the
    underlying keys (``x``, ``pcov``, ``_diagnostics_report``).

    Additional Attributes
    ---------------------
    model : callable
        The model function f(x, \\*params) used for fitting.
    xdata : array_like
        The independent variable data.
    ydata : array_like
        The dependent variable data (observations).
    popt : array_like
        Fitted parameters (alias for self.x).
    pcov : array_like
        Parameter covariance matrix.
    diagnostics : ModelHealthReport | None
        Model health diagnostics report (if compute_diagnostics=True).

    Statistical Properties
    ----------------------
    r_squared : float
        Coefficient of determination (R²). Measures goodness of fit.
        Range: (-∞, 1], where 1 is perfect fit.

    adj_r_squared : float
        Adjusted R² accounting for number of parameters.
        Preferred over R² when comparing models with different parameters.

    rmse : float
        Root mean squared error. Lower is better.

    mae : float
        Mean absolute error. Robust to outliers.

    aic : float
        Akaike Information Criterion. Lower is better.
        Used for model selection.

    bic : float
        Bayesian Information Criterion. Lower is better.
        Penalizes model complexity more than AIC.

    residuals : array_like
        Residuals (observed - predicted). Should be random for good fit.

    predictions : array_like
        Model predictions at xdata points.

    Methods
    -------
    confidence_intervals(alpha=0.95)
        Compute parameter confidence intervals.

    prediction_interval(x, alpha=0.95)
        Compute prediction interval at new x values.

    plot(ax=None, show_residuals=True)
        Plot data, fit, and residuals.

    summary()
        Print statistical summary table.

    Examples
    --------
    Basic usage with enhanced features::

        import numpy as np
        import jax.numpy as jnp
        from nlsq import curve_fit

        # Define model
        def exponential(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        # Generate data
        x = np.linspace(0, 10, 100)
        y_true = 10 * np.exp(-0.5 * x) + 2
        y = y_true + np.random.normal(0, 0.5, 100)

        # Fit and analyze
        result = curve_fit(exponential, x, y, p0=[10, 0.5, 2])

        print(f"R² = {result.r_squared:.4f}")
        print(f"RMSE = {result.rmse:.4f}")
        print(f"AIC = {result.aic:.2f}")

        # Get confidence intervals
        ci = result.confidence_intervals(alpha=0.95)
        print(f"95% CI for parameters: {ci}")

        # Visualize fit
        result.plot()

        # Statistical summary
        result.summary()

    With diagnostics::

        result = curve_fit(exponential, x, y, compute_diagnostics=True)
        print(result.diagnostics.summary())
        print(result.diagnostics.identifiability.health_status)

    Backward compatibility::

        # Tuple unpacking still works
        popt, pcov = curve_fit(exponential, x, y)

        # But enhanced features available if not unpacked
        result = curve_fit(exponential, x, y)
        result.plot()
    """

    def __init__(self, *args, **kwargs):
        """Initialize enhanced curve fit result."""
        super().__init__(*args, **kwargs)

        # Cache for computed properties - must use object.__setattr__ because
        # OptimizeResult sets __setattr__ = dict.__setitem__, which would store
        # these as public dict keys and pollute the result dict.
        object.__setattr__(self, "_predictions_cache", None)
        object.__setattr__(self, "_residuals_cache", None)

    def __dir__(self):
        # OptimizeResult lists only the dict keys; keep the statistics and methods
        # of this class visible to tab completion too.
        public = (n for n in dir(type(self)) if not n.startswith("_"))
        return sorted(set(super().__dir__()) | set(public))

    def __setattr__(self, name, value):
        # OptimizeResult maps attribute assignment to dict keys, which a
        # read-only property would then silently shadow (``result.popt = v``
        # left ``result.popt`` unchanged). Route to the property setter or fail.
        prop = getattr(type(self), name, None)
        if isinstance(prop, property):
            if prop.fset is None:
                raise AttributeError(
                    f"'{name}' is a read-only derived property of CurveFitResult; "
                    "set the underlying key instead (e.g. result['x'] = ...)"
                )
            prop.fset(self, value)
        else:
            self[name] = value

    def __iter__(self):
        """Support tuple unpacking: popt, pcov = curve_fit(...)"""
        return iter((self.popt, self.pcov))

    def __getitem__(self, key):
        """SciPy idiom ``curve_fit(...)[0]`` indexes (popt, pcov); str keys stay dict."""
        if isinstance(key, (int, np.integer, slice)):
            return (self.popt, self.pcov)[key]
        return super().__getitem__(key)

    @property
    def popt(self):
        """Fitted parameters (alias for self.x).

        Returns
        -------
        popt : ndarray
            Fitted parameters as NumPy array for SciPy compatibility.
        """
        return np.asarray(self.x)

    @popt.setter
    def popt(self, value):
        self["x"] = value

    @property
    def pcov(self):
        """Parameter covariance matrix.

        Returns
        -------
        pcov : ndarray
            Covariance matrix as NumPy array for SciPy compatibility.
        """
        # Access from dict, convert JAX arrays to NumPy
        _pcov = self.get("pcov")
        if _pcov is not None:
            return np.asarray(_pcov)
        return _pcov

    @pcov.setter
    def pcov(self, value):
        self["pcov"] = value

    @property
    def diagnostics(self):
        """Model health diagnostics report.

        Returns the health report if compute_diagnostics=True was
        specified when calling curve_fit(), otherwise returns None.

        Returns
        -------
        diagnostics : ModelHealthReport | None
            Aggregated model health report containing identifiability
            analysis, gradient health monitoring, and other health metrics.
            None if diagnostics were not computed.

        Examples
        --------
        >>> result = curve_fit(model, x, y, compute_diagnostics=True)
        >>> if result.diagnostics is not None:
        ...     print(result.diagnostics.summary())
        ...     print(result.diagnostics.status)
        ...     print(result.diagnostics.health_score)
        ...     if result.diagnostics.identifiability is not None:
        ...         print(result.diagnostics.identifiability.health_status)

        See Also
        --------
        nlsq.diagnostics.ModelHealthReport : Aggregated health report
        nlsq.diagnostics.IdentifiabilityReport : Identifiability analysis
        nlsq.diagnostics.GradientHealthReport : Gradient health monitoring
        """
        # Stored under "_diagnostics_report" (set by minpack.py); assigning
        # ``result.diagnostics = v`` writes the same key via the setter below.
        return self.get("_diagnostics_report")

    @diagnostics.setter
    def diagnostics(self, value):
        self["_diagnostics_report"] = value

    def _cached(self, attr: str, compute):
        """Memoise ``compute()`` until the inputs it depends on are replaced.

        The key holds the model/data/fun/sigma objects (compared by identity) and the
        bytes of ``popt``, so reassigning ``x``, ``xdata``, ``ydata``, ``model``
        ``fun`` or ``sigma`` recomputes. Mutating ``xdata``/``ydata`` in place is not
        detected.
        """
        refs = tuple(self.get(k) for k in ("model", "xdata", "ydata", "fun", "sigma"))
        xb = None if self.get("x") is None else np.asarray(self.get("x")).tobytes()
        cached = getattr(self, attr)
        if (
            cached is not None
            and all(c is r for c, r in zip(cached[0], refs, strict=True))
            and cached[1] == xb
        ):
            return cached[2]
        value = compute()
        object.__setattr__(self, attr, (refs, xb, value))
        return value

    def _unweighted_fun(self):
        """Optimizer residual ``f - y`` with the sigma weighting undone.

        ``fun`` is ``(f - y) / sigma`` for a 1-D sigma and ``L^-1 (f - y)`` for
        a covariance sigma (``C = L L^T``).
        """
        fun = np.asarray(self.fun, dtype=float)
        sigma = self.get("sigma")
        if sigma is None:
            return fun
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim == 0 or (sig.ndim == 1 and sig.shape == fun.shape):
            return fun * sig
        if sig.ndim == 2 and sig.shape == (fun.size, fun.size):
            return np.linalg.cholesky(sig) @ fun
        raise ValueError(
            "Cannot undo the sigma weighting of `fun`: sigma shape "
            f"{sig.shape} does not match the residuals {fun.shape}. "
            "Provide model and xdata on the result instead."
        )

    @property
    def predictions(self):
        """Model predictions at xdata points.

        Returns
        -------
        predictions : ndarray
            Model predictions: f(xdata, \\*popt)

        Raises
        ------
        AttributeError
            If neither ``model`` and ``xdata`` nor ``ydata`` and ``fun`` are
            available on the result.
        """

        def compute():
            if self.get("model") is not None and self.get("xdata") is not None:
                return self._eval(self.xdata, *self.popt)
            # Fallback: back-calculate from fun = (f - y) / sigma
            if self.get("ydata") is not None and self.get("fun") is not None:
                return np.array(self.ydata) + self._unweighted_fun()
            raise AttributeError(
                "Cannot compute predictions: model and xdata not available. "
                "This may occur if result was created without these attributes.",
            )

        return self._cached("_predictions_cache", compute)

    @property
    def residuals(self):
        """Residuals (observed - predicted).

        Returns
        -------
        residuals : ndarray
            Residuals: ydata - predictions

        Raises
        ------
        AttributeError
            If neither ``model``, ``xdata`` and ``ydata`` nor ``ydata`` and ``fun``
            are available on the result.

        Notes
        -----
        For a good fit, residuals should be randomly distributed around zero
        with no systematic patterns.
        """

        def compute():
            # Goodness-of-fit residuals are the RAW residuals (observed -
            # predicted), NOT the optimizer's `fun` vector. When sigma weighting
            # is used, `fun` is the *weighted* residual (model - ydata)/sigma,
            # which would scale every downstream statistic (R^2, RMSE, MAE, AIC,
            # BIC, prediction/confidence intervals) by 1/sigma and also carries
            # the opposite sign. Recompute from the model so the documented
            # "ydata - predictions" definition holds regardless of weighting.
            if (
                self.get("model") is not None
                and self.get("xdata") is not None
                and self.get("ydata") is not None
            ):
                return np.array(self.ydata) - np.array(
                    self.model(self.xdata, *self.popt)
                )
            if self.get("ydata") is not None and self.get("fun") is not None:
                # No model available: undo the sigma weighting of `fun`.
                return -self._unweighted_fun()
            raise AttributeError(
                "Cannot compute residuals: neither model+data nor fun available",
            )

        return self._cached("_residuals_cache", compute)

    @property
    def r_squared(self):
        """Coefficient of determination (R²).

        Returns
        -------
        r2 : float
            R² value in range (-∞, 1]. Values closer to 1 indicate better fit.

        Raises
        ------
        AttributeError
            If ``ydata`` is not available on the result.

        Notes
        -----
        R² = 1 - SS_res / SS_tot

        where SS_res = sum((y - y_pred)²) and SS_tot = sum((y - y_mean)²)

        Interpretation:
        - R² = 1: Perfect fit
        - R² = 0: Model no better than mean
        - R² < 0: Model worse than mean (overfitting or poor model)
        """
        if not hasattr(self, "ydata"):
            raise AttributeError("Cannot compute R²: ydata not available")

        y = np.array(self.ydata)
        used = self._used_rows()
        if used is not None:
            y = y[used]
        ss_res = np.sum(self._used_residuals() ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if not np.isfinite(ss_tot) or ss_tot < np.finfo(float).tiny:
            warnings.warn(
                "Total sum of squares is zero or NaN (constant/NaN data). R² undefined.",
                OptimizeWarning,
                stacklevel=2,
            )
            return np.nan

        return 1 - (ss_res / ss_tot)

    @property
    def adj_r_squared(self):
        """Adjusted R² accounting for number of parameters.

        Returns
        -------
        adj_r2 : float
            Adjusted R² value.

        Raises
        ------
        AttributeError
            If ``ydata`` is not available on the result.

        Notes
        -----
        Adj R² = 1 - (1 - R²) * (n - 1) / (n - p)

        where n is number of data points and p is the number of fitted
        parameters (including any intercept/offset parameter).

        Adjusted R² penalizes adding parameters and is better for comparing
        models with different numbers of parameters.
        """
        if not hasattr(self, "ydata"):
            raise AttributeError("Cannot compute adjusted R²: ydata not available")

        n = self._n_used()
        p = len(self.popt)
        r2 = self.r_squared

        if n - p <= 0:
            warnings.warn(
                "Not enough degrees of freedom for adjusted R².",
                OptimizeWarning,
                stacklevel=2,
            )
            return np.nan

        return 1 - (1 - r2) * (n - 1) / (n - p)

    @property
    def rmse(self):
        """Root mean squared error.

        Returns
        -------
        rmse : float
            Root mean squared error: sqrt(mean(residuals²))

        Notes
        -----
        RMSE has the same units as ydata and provides intuitive error measure.
        Lower values indicate better fit.
        """
        return np.sqrt(np.mean(self._used_residuals() ** 2))

    @property
    def mae(self):
        """Mean absolute error.

        Returns
        -------
        mae : float
            Mean absolute error: mean(\\|residuals\\|)

        Notes
        -----
        MAE is more robust to outliers than RMSE.
        """
        return np.mean(np.abs(self._used_residuals()))

    @property
    def aic(self):
        """Akaike Information Criterion.

        Returns
        -------
        aic : float
            AIC value. Lower is better.

        Notes
        -----
        AIC = 2k + L, where k is the number of parameters, n the number of
        data points and L the -2 log-likelihood term (constants dropped):

        - no ``sigma``: L = n*ln(RSS/n), RSS = sum((y - f)²)
        - ``sigma`` with ``absolute_sigma=False``: L = n*ln(chi²/n)
        - ``sigma`` with ``absolute_sigma=True``: L = chi² (known variance)

        chi² = sum(((y - f)/sigma)²), or rᵀC⁻¹r for a 2-D covariance
        ``sigma``. Only compare AIC values computed with the same ``sigma``.

        Used for model selection. Penalizes model complexity.
        """
        return self._information_criterion("AIC", 2.0)

    def _used_rows(self):
        """Boolean mask of the rows that carry weight, or None when all do.

        A 1-D ``sigma`` entry of ``inf`` is a zero-weight row: the fit ignores it, so
        it is not an observation for R^2, RMSE/MAE, AIC/BIC or the degrees of freedom.
        """
        sig = self.get("sigma")
        if sig is None or np.ndim(sig) != 1:
            return None
        used = ~np.isinf(np.asarray(sig, dtype=float))
        n = len(self.ydata) if hasattr(self, "ydata") else len(self.residuals)
        return None if used.all() or used.shape != (n,) else used

    def _n_used(self) -> int:
        """Effective sample size: observations with non-zero weight."""
        used = self._used_rows()
        n = len(self.ydata) if hasattr(self, "ydata") else len(self.residuals)
        return n if used is None else int(used.sum())

    def _used_residuals(self):
        r = np.asarray(self.residuals)
        used = self._used_rows()
        return r if used is None else r[used]

    def _noise_known(self) -> bool:
        """True when the noise scale is known: ``absolute_sigma=True`` (with the
        given ``sigma``, or the solver's implicit unit sigma when omitted)."""
        return bool(self.get("absolute_sigma", False))

    def _effective_sigma(self):
        """The sigma the fit used: the stored one, a unit sigma when omitted
        with ``absolute_sigma=True``, else None (unweighted, estimated noise)."""
        sigma = self.get("sigma")
        if sigma is None and self._noise_known():
            return np.float64(1.0)
        return sigma

    def _chi2(self):
        """Sigma-weighted chi-square at popt, or None for unweighted fits."""
        sigma = self._effective_sigma()
        if sigma is None:
            return None
        r = np.asarray(self.residuals, dtype=float)
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim == 2:
            from scipy.linalg import solve_triangular

            z = solve_triangular(np.linalg.cholesky(sig), r, lower=True)
            return float(z @ z)
        return float(np.sum((r / sig) ** 2))

    def _information_criterion(self, name: str, penalty_per_param: float):
        if not hasattr(self, "ydata"):
            raise AttributeError(f"Cannot compute {name}: ydata not available")

        n = self._n_used()
        k = len(self.popt)
        chi2 = self._chi2()
        if chi2 is not None and self._noise_known():
            if not np.isfinite(chi2):
                warnings.warn(
                    f"chi² is NaN/inf, {name} undefined.", OptimizeWarning, stacklevel=3
                )
                return np.nan
            return penalty_per_param * k + chi2
        ss = np.sum(self._used_residuals() ** 2) if chi2 is None else chi2
        if not np.isfinite(ss) or ss <= 0:
            warnings.warn(
                f"RSS ≤ 0 or NaN, {name} undefined.", OptimizeWarning, stacklevel=3
            )
            return np.nan

        return penalty_per_param * k + n * np.log(ss / n)

    @property
    def bic(self):
        """Bayesian Information Criterion.

        Returns
        -------
        bic : float
            BIC value. Lower is better.

        Raises
        ------
        AttributeError
            If ``ydata`` is not available on the result.

        Notes
        -----
        BIC = k*ln(n) + L

        with the same -2 log-likelihood term L as ``aic``: n*ln(RSS/n)
        without ``sigma``, n*ln(chi²/n) with ``sigma`` and
        ``absolute_sigma=False``, chi² with ``absolute_sigma=True``.

        BIC penalizes model complexity more heavily than AIC.
        Preferred for larger datasets.
        """
        if not hasattr(self, "ydata"):
            raise AttributeError("Cannot compute BIC: ydata not available")
        return self._information_criterion("BIC", float(np.log(self._n_used())))

    def confidence_intervals(self, alpha: float = 0.95):
        """Compute parameter confidence intervals.

        Parameters
        ----------
        alpha : float, optional
            Confidence level (default: 0.95 for 95% CI).

        Returns
        -------
        intervals : ndarray
            Array of shape (n_params, 2) with [lower, upper] bounds for each parameter.

        Raises
        ------
        AttributeError
            If ``pcov`` is ``None`` (the fit was run with
            ``compute_covariance=False``).

        Examples
        --------
        >>> result = curve_fit(model, x, y)
        >>> ci = result.confidence_intervals(alpha=0.95)
        >>> for i, (lower, upper) in enumerate(ci):
        ...     print(f"Parameter {i}: [{lower:.3f}, {upper:.3f}]")

        Notes
        -----
        Confidence intervals are computed using the parameter covariance matrix
        and Student's t-distribution. Assumes residuals are normally distributed.
        """
        if self.pcov is None:
            raise AttributeError(
                "Cannot compute confidence intervals: pcov not available "
                "(the fit was run with compute_covariance=False).",
            )

        alpha = _check_alpha(alpha)
        p = len(self.popt)
        t_val, _ = self._interval_quantile(alpha, "Confidence intervals")
        if t_val is None:
            return np.full((p, 2), np.nan)

        # Standard errors from covariance diagonal
        perr = np.sqrt(np.diag(self.pcov))

        # Confidence intervals
        intervals = np.zeros((p, 2))
        intervals[:, 0] = self.popt - t_val * perr  # Lower bound
        intervals[:, 1] = self.popt + t_val * perr  # Upper bound

        return intervals

    def _interval_quantile(self, alpha: float, what: str):
        """Return ``(quantile, dof)`` for a two-sided ``alpha`` interval.

        With a known noise scale (``sigma`` with ``absolute_sigma=True``) the
        covariance is exact and the quantile is Gaussian for every ``n``;
        otherwise it is Student-t with ``dof = n - p``. With ``n <= p`` and an
        estimated noise scale there is no residual degree of freedom: warn
        (from the public caller's frame) and return ``None`` (bounds
        undefined). The upper-tail form ``isf((1 - alpha) / 2)`` keeps the
        representable tail probability when ``alpha`` is within an ULP of 1.
        """
        from scipy import stats

        n = self._n_used()
        p = len(self.popt)
        dof = max(n - p, 1)
        tail = (1.0 - alpha) / 2.0
        known_noise = self._noise_known()
        if n > p or known_noise:
            if n <= p:
                warnings.warn(
                    f"Underdetermined fit: {n} data points and {p} parameters. "
                    f"{what} use the normal quantile (known absolute sigma).",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return (
                stats.norm.isf(tail) if known_noise else stats.t.isf(tail, dof)
            ), dof
        warnings.warn(
            f"Underdetermined fit: {n} data points and {p} parameters. "
            f"{what} are undefined (NaN): no residual degrees of freedom.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None, dof

    @staticmethod
    def _propagated_variance(jacobian, pcov):
        """Diagonal of ``J pcov J^T``; infinite when ``pcov`` is not finite."""
        if not np.all(np.isfinite(pcov)):
            return np.full(jacobian.shape[0], np.inf)
        return np.sum((jacobian @ pcov) * jacobian, axis=1)

    def _finite_diff_jacobian(
        self, x, popt, eps: float = float(np.sqrt(np.finfo(float).eps))
    ):
        """Compute y_pred and the Jacobian of ``self.model`` w.r.t. ``popt`` at
        ``x``. Shared by ``prediction_interval`` and ``confidence_band``, which
        both need the same parameter-uncertainty propagation through the model.

        The Jacobian is exact (``jax.jacfwd``). Central differences are only a
        fallback for models ``jax.jacfwd`` cannot trace (e.g. NumPy-only), and
        assume the model is smooth on the scale of the standard step
        ``cbrt(eps) * max(|p|, 1)``. Models that oscillate or vary on a scale
        comparable to that step can alias (two neighbouring step sizes may agree
        on a wrong value); write such models with ``jax.numpy`` so autodiff is
        exact.

        Fallback details: a ladder of steps ``h_a * 10**k`` around ``h_a`` is
        tried lazily (rung 0 plus one neighbour when the standard step is well
        resolved; otherwise descending). It reaches down to about
        ``1e-12 * |p|`` (``|p|`` is floored at 1e-8 only for ``p == 0`` or
        non-finite ``p``) so probes can stay inside a nearby domain edge, with
        a hard cap of ``k >= -40`` (smallest step 6e-46) and ``k <= 3``: at most
        44 rungs, i.e. 88 model evaluations per parameter. Non-finite estimates,
        probes that raise ``ValueError``/``ArithmeticError``, and steps whose
        function difference is within rounding noise of ``f`` are discarded;
        steps more than ``1e3 * |p|`` are "nonlocal" and are used only when two
        adjacent ones agree. A parameter with no effect gets an exact 0.
        ``FloatingPointError`` is raised when no finite, local derivative can
        be obtained (including when some probes raised and the rest were non-finite); if every probe raised, the model's own error is re-raised.
        """
        popt = np.asarray(popt, dtype=float)
        n_params = len(popt)

        y_pred = np.atleast_1d(self._eval(x, *popt))
        # Size from the predictions: 2-D xdata is (k, N), so len(x) would be k.
        n_points = y_pred.shape[0]
        try:
            import jax
            import jax.numpy as jnp

            jac = jax.jacfwd(lambda p: jnp.asarray(self.model(x, *p)))(
                jnp.asarray(popt)
            )
            # a scalar model output (broadcast over the points) has a single row
            jac = np.broadcast_to(
                np.asarray(jac, dtype=float).reshape(-1, n_params), (n_points, n_params)
            )
            if np.all(np.isfinite(jac)):
                return y_pred, jac
            # non-finite autodiff (e.g. sqrt at 0, or a NaN-gradient trap):
            # let central differences decide; they raise if it is undefined too
            _logger.debug("non-finite jax.jacfwd Jacobian; using finite differences")
        except (
            TypeError,  # incl. jax.errors.JAXTypeError (tracer/concretization)
            ValueError,  # e.g. pure_callback has no JVP rule
            NotImplementedError,  # primitive without a JVP rule
            jax.errors.JAXIndexError,
        ) as exc:
            # untraceable (e.g. numpy-only) model: fall back to finite differences
            _logger.debug("jax.jacfwd failed (%s); using finite differences", exc)

        jacobian = np.zeros((n_points, n_params))
        step = eps ** (2.0 / 3.0)  # = cbrt(machine eps), optimal for central diff

        for i in range(n_params):
            jacobian[:, i] = self._fd_derivative(x, popt, i, step)

        if not np.all(np.isfinite(jacobian)):
            raise FloatingPointError(
                "Finite-difference Jacobian is not finite; the model is not "
                "differentiable at popt in float64 (use a JAX-traceable model)."
            )
        return y_pred, jacobian

    def _fd_central(self, x, popt, i: int, h: float):
        """Central difference at step ``h``: ``(derivative, strong, weak, exc)``.

        ``weak``: finite and ``|f+ - f-|`` above ``8 eps |f|`` (a single-ULP
        change is rounding noise and fails). ``strong``: above ``1e4 eps |f|``
        (roundoff error below ~1e-4 relative). A probe that raises a
        numerical/domain error (``ValueError``, ``ArithmeticError``) is invalid
        (``exc`` is that error); any other exception propagates.
        """
        eps_f = np.finfo(float).eps
        plus, minus = popt.copy(), popt.copy()
        plus[i] += h
        minus[i] -= h
        try:
            with np.errstate(all="ignore"):
                f_plus = np.asarray(np.atleast_1d(self.model(x, *plus)), dtype=float)
                f_minus = np.asarray(np.atleast_1d(self.model(x, *minus)), dtype=float)
                dy = f_plus - f_minus
        except (ValueError, ArithmeticError) as exc:
            return np.array(np.nan), False, False, exc
        with np.errstate(all="ignore"):
            # divide by the representable step actually taken
            d = dy / (plus[i] - minus[i])
            scale = eps_f * np.linalg.norm(np.maximum(abs(f_plus), abs(f_minus)))
            finite = bool(np.all(np.isfinite(d)))
            dy_norm = np.linalg.norm(dy)
            return (
                d,
                finite and dy_norm > 1e4 * scale,
                finite and dy_norm > 8 * scale,
                None,
            )

    def _fd_derivative(self, x, popt, i: int, step: float):  # noqa: C901 - one linear decision list
        """Finite-difference column ``df/dp_i`` (see ``_finite_diff_jacobian``)."""
        first_exc: list[Exception] = []
        n_completed = [0]  # probes that returned (finite or not) instead of raising

        def central(i, h):
            d, strong, weak, exc = self._fd_central(x, popt, i, h)
            if exc is not None:
                first_exc.append(exc)
            else:
                n_completed[0] += 1
            return d, strong, weak

        # Ladder of central differences at h_k = h_a * 10**k around the
        # standard step h_a = cbrt(eps)*max(|p|, 1) (the first choice), for
        # k_lo <= k <= k_hi. Down to about 1e-12*|p| (|p| floored at 1e-8 only if
        # p == 0 or non-finite), so probes stay inside a nearby domain edge,
        # and up to 1e-2*|p|. Rungs above 1e3*|p| are "nonlocal" (they perturb p
        # by many multiples of itself).
        # Budget: k_lo >= -40 and k_hi <= 3 -> at most 44 rungs = 88 model
        # evaluations per parameter (smallest step 6e-46). Rungs are evaluated
        # lazily, so the common case costs rung 0 plus one neighbour
        # (4 evaluations).
        p_i = abs(popt[i])
        finite_p = bool(np.isfinite(p_i) and p_i > 0)
        scale = p_i if finite_p else 1e-8
        h_a = step * max(p_i if np.isfinite(p_i) else 0.0, 1.0)
        # logs (not ratios): scale can be subnormal, so 1e-12*scale underflows
        log_a = np.log10(h_a)
        k_lo = max(min(-1, -int(np.ceil(log_a - (np.log10(scale) - 12.0) - 1e-9))), -40)
        k_hi = int(np.floor(np.log10(max(h_a, 1e-2 * scale)) - log_a + 1e-9))
        rungs = {}

        def rung(k):
            if k not in rungs:
                rungs[k] = central(i, h_a * 10.0**k)
            return rungs[k]

        def local(k):
            return not finite_p or h_a * 10.0**k <= 1e3 * p_i * (1 + 1e-9)

        d0, strong0, _ = rung(0)
        # 1. The standard step, when local, well resolved and in agreement
        #    with a neighbouring rung (a few percent), is used as is.
        if (
            strong0
            and local(0)
            and any(
                k_lo <= k <= k_hi and rung(k)[2] and _rel_diff(d0, rung(k)[0]) <= 2e-2
                for k in (-1, 1)
            )
        ):
            return d0
        # 2. Otherwise descend lazily (largest steps first, least roundoff)
        #    and stop at the first adjacent pair of local rungs, both clearly
        #    above rounding noise, that agree to 1e-4; keep its larger step.
        for k in [*range(0, k_lo - 1, -1), *range(1, k_hi + 1)]:
            if rung(k)[2] and local(k):
                for kn in (k + 1, k - 1):
                    if kn in rungs and rungs[kn][2] and local(kn):
                        if _rel_diff(rung(k)[0], rungs[kn][0]) <= 1e-4:
                            return rung(max(k, kn))[0]
        # 3. No such pair: Ridders-style, the best-agreeing adjacent pair
        #    among rungs clearly above noise (non-finite and one-ULP rungs are
        #    discarded). Local rungs first; nonlocal ones only as an agreeing pair.
        ok = [k for k in range(k_lo, k_hi + 1) if rung(k)[2]]
        ok_local = [k for k in ok if local(k)]
        ok_nonlocal = [k for k in ok if not local(k)]

        def pairs_of(ks, tol=np.inf):
            out = [
                (_rel_diff(rung(k)[0], rung(k + 1)[0]), k + 1)
                for k in ks
                if k + 1 in ks
            ]
            return [t for t in out if t[0] <= tol]

        if pairs_of(ok_local):
            return rung(min(pairs_of(ok_local))[1])[0]
        if ok_local:  # no adjacent pair: the best-resolved local survivor
            return rung(ok_local[-1])[0]
        if pairs_of(ok_nonlocal, 2e-2):  # consistent over decades: ~linear
            return rung(min(pairs_of(ok_nonlocal, 2e-2))[1])[0]
        if ok_nonlocal:  # lone nonlocal probe(s): not evidence of a derivative
            raise FloatingPointError(
                f"Finite-difference derivative with respect to parameter {i} "
                f"(|p| = {p_i:.3g}) is only resolved at steps far larger than "
                "the parameter; use a JAX-traceable model."
            )
        fin = [d for d, _, _ in rungs.values() if np.all(np.isfinite(d))]
        if fin:
            # nothing clears the noise floor: f is (nearly) unchanged at every
            # rung, i.e. a zero (or unresolvable) derivative -> standard step
            return d0 if np.all(np.isfinite(d0)) else fin[0]
        if first_exc and not n_completed[0]:  # every probe raised: keep its error
            raise first_exc[0]
        return d0

    def prediction_interval(self, x=None, alpha: float = 0.95):
        """Compute prediction interval at x values.

        Parameters
        ----------
        x : array_like, optional
            x values for prediction. If None, uses self.xdata (default: None).
        alpha : float, optional
            Confidence level (default: 0.95).

        Returns
        -------
        intervals : ndarray
            Array of shape (n_points, 2) with [lower, upper] bounds for each point.

        Raises
        ------
        AttributeError
            If ``model`` is not available, if ``pcov`` is ``None`` (the fit was
            run with ``compute_covariance=False``), or if ``x`` is None and
            ``xdata`` is not stored on the result.

        Examples
        --------
        >>> result = curve_fit(model, x, y)
        >>> pi = result.prediction_interval()  # Use fitted x values
        >>> pi_new = result.prediction_interval(x_new)  # Use new x values

        Notes
        -----
        Prediction intervals account for both parameter uncertainty (from pcov)
        and inherent data variability. For an unweighted fit the latter is the
        pooled residual variance; for a fit with ``sigma`` it is weighted by
        ``sigma(x)**2`` (see ``_noise_variance_at``).

        ``x`` may be a scalar (treated as one point). With no residual degrees
        of freedom (``n <= n_params``) a ``RuntimeWarning`` is issued and, unless
        the noise is known (``sigma`` with ``absolute_sigma=True``, which then
        uses the normal quantile), every bound is NaN. ``confidence_intervals``
        and ``confidence_band`` behave the same way.
        """
        if not hasattr(self, "model"):
            raise AttributeError(
                "Cannot compute prediction interval: model not available",
            )
        if self.get("pcov") is None:
            raise AttributeError(
                "Cannot compute prediction interval: pcov not available "
                "(the fit was run with compute_covariance=False).",
            )

        # Use self.xdata if x not provided
        if x is None:
            if not hasattr(self, "xdata"):
                raise AttributeError(
                    "Cannot compute prediction interval: xdata not available. "
                    "Either pass x explicitly or ensure xdata is stored in result.",
                )
            x = self.xdata

        alpha = _check_alpha(alpha)
        x = np.atleast_1d(np.asarray(x))
        popt = self.popt
        pcov = self.pcov

        t_val, dof = self._interval_quantile(alpha, "Prediction intervals")
        if t_val is None:  # undefined: no need (or right) to differentiate
            n_out = np.atleast_1d(self._eval(x, *popt)).shape[0]
            return np.full((n_out, 2), np.nan)
        y_pred, jacobian = self._finite_diff_jacobian(x, popt)

        # Variance of the mean response (parameter-uncertainty contribution),
        # same as confidence_band: J @ pcov @ J.T (diagonal)
        variance = self._propagated_variance(jacobian, pcov)

        # Residual variance (inherent data variability)
        s2 = self._noise_variance_at(x, self._n_used(), dof)

        # Prediction interval = confidence band variance + residual variance
        se_pred = np.sqrt(np.maximum(variance, 0) + s2)

        lower = y_pred - t_val * se_pred
        upper = y_pred + t_val * se_pred

        # Return as (n, 2) array
        intervals = np.column_stack([lower, upper])
        return intervals

    def _n_obs(self, x) -> int:
        """Number of observations in ``x``: its last axis, or its first axis for the
        points-first ``(n_points, n_features)`` layout of the hybrid path."""
        shape = np.shape(x)
        if len(shape) <= 1:
            return shape[0] if shape else 1
        if self.get("x_points_first"):  # recorded by the hybrid solver: (n_points, k)
            return shape[0]
        xd, y = np.shape(cast(Any, self.get("xdata"))), self.get("ydata")
        if y is not None and len(xd) > 1 and xd[0] == len(y) != xd[-1]:
            return shape[0]
        return shape[-1]

    def _eval(self, x, *params) -> np.ndarray:
        """Model values at ``x``; a scalar output is broadcast over the observations,
        as the optimizer does."""
        y = np.array(self.model(x, *params))
        return np.full(self._n_obs(x), y) if y.ndim == 0 else y

    def _noise_variance_at(self, x, n: int, dof: int):
        """Observation-noise variance for a prediction interval at ``x``.

        Unweighted fits: pooled residual variance (a scalar). With a 1-D
        ``sigma`` (a scalar sigma is broadcast; a 2-D covariance ``sigma`` uses
        its diagonal) the noise follows the fit's weighting: ``sigma(x)**2`` times
        the reduced chi-square (``absolute_sigma=False``) or ``sigma(x)**2``
        alone (``absolute_sigma=True``); ``sigma(x)`` is the data sigma at the
        data points and linearly interpolated in x elsewhere (1-D x only;
        ``np.interp`` clamps, so outside the data range ``sigma(x)`` is the edge
        value, not an extrapolation). If the stored ``sigma`` does not match the
        residuals (or is non-finite / non-positive) the pooled variance is used
        and a ``RuntimeWarning`` is issued.
        """
        resid = np.asarray(self.residuals, dtype=float)
        sigma = self._effective_sigma()
        if sigma is not None:
            sig = np.asarray(sigma, dtype=float)
            if sig.ndim == 0:
                sig = np.full(resid.shape, float(sig))
            elif sig.ndim == 2 and sig.shape[0] == sig.shape[1]:
                # covariance-matrix sigma: marginal variance per observation
                sig = np.sqrt(np.abs(np.diag(sig)))
            sig = np.abs(sig)  # the noise scale is |sigma| (solver accepts signed)
            # sigma=inf is a zero-weight row (the fit ignores it): it carries no noise
            # information, so it is left out of the interpolation/average below.
            fin = np.isfinite(sig)
            if (
                sig.ndim == 1
                and sig.shape == resid.shape
                and not np.any(np.isnan(sig))
                and fin.any()
                and np.all(sig > 0)
            ):
                scale = 1.0 if self._noise_known() else self._chi2() / dof
                xd = np.asarray(getattr(self, "xdata", []), dtype=float)
                xq = np.asarray(x, dtype=float)
                if xq.shape == xd.shape and np.array_equal(xq, xd):
                    sig_x = sig
                elif xd.ndim == 1 and xq.ndim == 1:
                    order = np.argsort(xd[fin])
                    sig_x = np.interp(xq, xd[fin][order], sig[fin][order])
                else:
                    sig_x = np.full(
                        self._n_obs(xq), float(np.sqrt(np.mean(sig[fin] ** 2)))
                    )
                return scale * sig_x**2
            warnings.warn(
                "Stored sigma does not match the residuals (shape "
                f"{sig.shape} vs {resid.shape}) or is non-finite/non-positive; "
                "the prediction interval uses the pooled unweighted variance "
                "while pcov is sigma-weighted.",
                RuntimeWarning,
                stacklevel=3,
            )
        return np.sum(resid**2) / dof

    def confidence_band(self, x=None, alpha: float = 0.95):
        """Compute mean prediction confidence band.

        Parameters
        ----------
        x : array_like, optional
            x values for prediction. If None, uses self.xdata (default: None).
        alpha : float, optional
            Confidence level (default: 0.95 for 95% CI).

        Returns
        -------
        lower : ndarray
            Lower confidence bound.
        upper : ndarray
            Upper confidence bound.

        Raises
        ------
        AttributeError
            If ``model`` is not available, if ``pcov`` is missing or ``None`` (the
            fit was run with ``compute_covariance=False``), or if ``x`` is None
            and ``xdata`` is not stored on the result.

        Notes
        -----
        This calculates the confidence interval for the mean response (f(x)),
        representing the uncertainty in the model curve itself due to parameter
        uncertainties.
        """
        if not hasattr(self, "model"):
            raise AttributeError("Cannot compute confidence band: model not available")
        if not hasattr(self, "pcov") or self.pcov is None:
            raise AttributeError(
                "Cannot compute confidence band: pcov not available "
                "(the fit was run with compute_covariance=False).",
            )

        if x is None:
            if not hasattr(self, "xdata"):
                raise AttributeError(
                    "Cannot compute confidence band: xdata not available",
                )
            x = self.xdata

        alpha = _check_alpha(alpha)
        # Ensure inputs are numpy arrays (a scalar x is one point)
        x = np.atleast_1d(np.asarray(x))
        popt = self.popt
        pcov = self.pcov

        t_val, _ = self._interval_quantile(alpha, "Confidence bands")
        if t_val is None:  # undefined: no need (or right) to differentiate
            shape = np.atleast_1d(self._eval(x, *popt)).shape
            return np.full(shape, np.nan), np.full(shape, np.nan)
        y0, jacobian = self._finite_diff_jacobian(x, popt)

        # Variance of the mean response: J @ pcov @ J.T (diagonal)
        variance = self._propagated_variance(jacobian, pcov)
        std_error = np.sqrt(np.maximum(variance, 0))

        lower = y0 - t_val * std_error
        upper = y0 + t_val * std_error

        return lower, upper

    def plot(
        self,
        ax=None,
        show_residuals: bool = True,
        show_confidence: bool = True,
        **kwargs,
    ):
        """Plot data, fitted curve, and residuals.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_residuals : bool, optional
            Whether to show residual plot (default: True).
        show_confidence : bool, optional
            Whether to show 95% confidence band (default: True).
        **kwargs
            Additional keyword arguments passed to plotting functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        axes : matplotlib.axes.Axes or array of Axes
            Axes object(s).

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        AttributeError
            If ``xdata`` or ``ydata`` is not available on the result.

        Examples
        --------
        >>> result = curve_fit(model, x, y)
        >>> result.plot()
        >>> plt.show()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib",
            ) from None

        if not hasattr(self, "xdata") or not hasattr(self, "ydata"):
            raise AttributeError("Cannot plot: xdata and ydata not available")

        # Create figure if needed
        if ax is None:
            if show_residuals:
                fig, (ax1, ax2) = plt.subplots(
                    2,
                    1,
                    figsize=(10, 8),
                    gridspec_kw={"height_ratios": [3, 1]},
                )
            else:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax2 = None
        else:
            fig = ax.figure
            ax1 = ax
            ax2 = None

        # Extract x and y data
        x = np.array(self.xdata)
        y = np.array(self.ydata)

        multi_dim = x.ndim > 1
        if multi_dim:
            # Multi-dimensional xdata (k, n): no single abscissa exists, so plot
            # against the observation index and evaluate at the data points.
            x_eval = x
            x = np.arange(y.size)
            x_dense = x
            y_dense = self._eval(x_eval, *self.popt)
            xlabel = "observation"
        else:
            # Dense grid for a smooth line and band
            x_eval = x_dense = np.linspace(x.min(), x.max(), 500)
            y_dense = self._eval(x_dense, *self.popt)
            xlabel = "x"

        # Extract plotting parameters from kwargs
        scatter_alpha = kwargs.pop("alpha", 0.6)
        scatter_color = kwargs.pop("color", None)

        # Plot data points
        scatter_kwargs = {"alpha": scatter_alpha, "label": "Data"}
        if scatter_color is not None:
            scatter_kwargs["color"] = scatter_color
        scatter_kwargs.update(kwargs)
        ax1.scatter(x, y, **scatter_kwargs)

        # Plot confidence band
        fit_color = "red" if scatter_color is None else scatter_color
        if show_confidence and hasattr(self, "pcov") and self.pcov is not None:
            try:
                lower, upper = self.confidence_band(x_eval, alpha=0.95)
                ax1.fill_between(
                    x_dense,
                    lower,
                    upper,
                    color=fit_color,
                    alpha=0.2,
                    label="95% CI",
                )
            except Exception:
                _logger.warning(
                    "Confidence band calculation failed",
                    exc_info=True,
                )

        # Plot fitted curve
        ax1.plot(
            x_dense,
            y_dense,
            color=fit_color,
            linewidth=2,
            label="Fit",
            zorder=10,
        )

        # Labels and title
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("y")
        ax1.set_title(f"Curve Fit (R² = {self.r_squared:.4f}, RMSE = {self.rmse:.4f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residual plot
        if show_residuals and ax2 is not None:
            ax2.scatter(x, self.residuals, alpha=0.6, color="gray")
            ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel("Residuals")
            ax2.set_title("Residual Plot")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if ax2 is not None:
            return fig, (ax1, ax2)
        return fig, ax1

    def summary(self):
        """Print statistical summary of fit.

        Displays:
        - Fitted parameters with standard errors
        - Goodness of fit metrics (R², RMSE, MAE)
        - Model selection criteria (AIC, BIC)
        - Convergence information
        - Diagnostics summary (if available)

        Examples
        --------
        >>> result = curve_fit(model, x, y)
        >>> result.summary()
        """
        print("=" * 70)
        print("Curve Fit Summary")
        print("=" * 70)

        # Parameters
        print("\nFitted Parameters:")
        print("-" * 70)

        if self.pcov is not None:
            perr = np.sqrt(np.diag(self.pcov))
            print(f"{'Parameter':<15} {'Value':>12} {'Std Error':>12} {'95% CI':>25}")
            print("-" * 70)

            ci = self.confidence_intervals(alpha=0.95)
            for i, (val, err, (ci_low, ci_high)) in enumerate(
                zip(self.popt, perr, ci, strict=False),
            ):
                print(
                    f"{'p' + str(i):<15} {val:>12.6f} {err:>12.6f} "
                    f"[{ci_low:>10.6f}, {ci_high:>10.6f}]",
                )
        else:
            print(f"{'Parameter':<15} {'Value':>12}")
            print("-" * 70)
            for i, val in enumerate(self.popt):
                print(f"{'p' + str(i):<15} {val:>12.6f}")

        # Goodness of fit
        print("\nGoodness of Fit:")
        print("-" * 70)

        def stat(name: str, spec: str) -> str:
            # a result built without ydata/model has no residual-based statistics
            try:
                return f"{getattr(self, name):>12{spec}}"
            except AttributeError:
                return f"{'N/A':>12}"

        print(f"R²                : {stat('r_squared', '.6f')}")
        print(f"Adjusted R²       : {stat('adj_r_squared', '.6f')}")
        print(f"RMSE              : {stat('rmse', '.6f')}")
        print(f"MAE               : {stat('mae', '.6f')}")

        # Model selection
        print("\nModel Selection Criteria:")
        print("-" * 70)
        print(f"AIC               : {stat('aic', '.2f')}")
        print(f"BIC               : {stat('bic', '.2f')}")

        # Convergence
        print("\nConvergence Information:")
        print("-" * 70)
        print(f"Success           : {self.success}")
        print(f"Message           : {self.message}")
        print(f"Function evals    : {self.nfev if hasattr(self, 'nfev') else 'N/A'}")
        if self.get("nit") is not None:
            print(f"Iterations        : {self['nit']}")
        print(f"Final cost        : {self.cost if hasattr(self, 'cost') else 'N/A'}")
        # self.optimality is None for CMA-ES results and absent entirely for
        # hybrid-streaming/multistart results -- applying the ':.6e' numeric
        # format spec to either (None, or the 'N/A' string fallback) raised
        # TypeError, so .summary() crashed for those result kinds. Format
        # only when there's an actual finite value to format.
        optimality = getattr(self, "optimality", None)
        optimality_str = f"{optimality:.6e}" if optimality is not None else "N/A"
        print(f"Optimality        : {optimality_str}")

        # Diagnostics summary (if available)
        if self.diagnostics is not None:
            print("\n" + "=" * 70)
            print("Model Health Diagnostics:")
            print("-" * 70)
            print(f"Overall Status    : {self.diagnostics.status.name}")
            print(f"Health Score      : {self.diagnostics.health_score:.2f}")
            if self.diagnostics.all_issues:
                print(f"Issues            : {len(self.diagnostics.all_issues)}")
                for issue in self.diagnostics.all_issues[:3]:  # Show first 3
                    print(f"  - [{issue.severity.name}] {issue.code}: {issue.message}")
            if self.diagnostics.identifiability is not None:
                ident = self.diagnostics.identifiability
                print(f"Identifiability   : {ident.health_status.name}")
                print(f"  Condition #     : {ident.condition_number:.2e}")
                print(f"  Numerical Rank  : {ident.numerical_rank}/{ident.n_params}")

        print("=" * 70)
