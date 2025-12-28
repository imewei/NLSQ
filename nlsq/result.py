"""Enhanced result objects for curve fitting with convenience methods."""

import warnings

import numpy as np

from nlsq.core._optimize import OptimizeResult


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

        # Cache for computed properties
        self._predictions_cache = None
        self._residuals_cache = None

    def __iter__(self):
        """Support tuple unpacking: popt, pcov = curve_fit(...)"""
        return iter((self.popt, self.pcov))

    @property
    def popt(self):
        """Fitted parameters (alias for self.x).

        Returns
        -------
        popt : ndarray
            Fitted parameters as NumPy array for SciPy compatibility.
        """
        return np.asarray(self.x)

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

    @property
    def predictions(self):
        """Model predictions at xdata points.

        Returns
        -------
        predictions : ndarray
            Model predictions: f(xdata, \\*popt)
        """
        if self._predictions_cache is None:
            if hasattr(self, "model") and hasattr(self, "xdata"):
                self._predictions_cache = np.array(self.model(self.xdata, *self.popt))
            # Fallback: use fun (residuals) to back-calculate
            elif hasattr(self, "ydata") and hasattr(self, "fun"):
                self._predictions_cache = np.array(self.ydata) - np.array(self.fun)
            else:
                raise AttributeError(
                    "Cannot compute predictions: model and xdata not available. "
                    "This may occur if result was created without these attributes."
                )
        return self._predictions_cache

    @property
    def residuals(self):
        """Residuals (observed - predicted).

        Returns
        -------
        residuals : ndarray
            Residuals: ydata - predictions

        Notes
        -----
        For a good fit, residuals should be randomly distributed around zero
        with no systematic patterns.
        """
        if self._residuals_cache is None:
            if hasattr(self, "fun"):
                # fun is the residual vector from optimization
                self._residuals_cache = np.array(self.fun)
            elif hasattr(self, "ydata"):
                # Calculate from predictions
                self._residuals_cache = np.array(self.ydata) - self.predictions
            else:
                raise AttributeError(
                    "Cannot compute residuals: neither fun nor ydata available"
                )
        return self._residuals_cache

    @property
    def r_squared(self):
        """Coefficient of determination (R²).

        Returns
        -------
        r2 : float
            R² value in range (-∞, 1]. Values closer to 1 indicate better fit.

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
        ss_res = np.sum(self.residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            warnings.warn("Total sum of squares is zero (constant data). R² undefined.")
            return np.nan

        return 1 - (ss_res / ss_tot)

    @property
    def adj_r_squared(self):
        """Adjusted R² accounting for number of parameters.

        Returns
        -------
        adj_r2 : float
            Adjusted R² value.

        Notes
        -----
        Adj R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)

        where n is number of data points and p is number of parameters.

        Adjusted R² penalizes adding parameters and is better for comparing
        models with different numbers of parameters.
        """
        if not hasattr(self, "ydata"):
            raise AttributeError("Cannot compute adjusted R²: ydata not available")

        n = len(self.ydata)
        p = len(self.popt)
        r2 = self.r_squared

        if n - p - 1 <= 0:
            warnings.warn("Not enough degrees of freedom for adjusted R².")
            return np.nan

        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

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
        return np.sqrt(np.mean(self.residuals**2))

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
        return np.mean(np.abs(self.residuals))

    @property
    def aic(self):
        """Akaike Information Criterion.

        Returns
        -------
        aic : float
            AIC value. Lower is better.

        Notes
        -----
        AIC = 2k + n*ln(RSS/n)

        where k is number of parameters, n is number of data points,
        and RSS is residual sum of squares.

        Used for model selection. Penalizes model complexity.
        """
        if not hasattr(self, "ydata"):
            raise AttributeError("Cannot compute AIC: ydata not available")

        n = len(self.ydata)
        k = len(self.popt)
        rss = np.sum(self.residuals**2)

        if rss <= 0:
            warnings.warn("RSS ≤ 0, AIC undefined.")
            return np.nan

        return 2 * k + n * np.log(rss / n)

    @property
    def bic(self):
        """Bayesian Information Criterion.

        Returns
        -------
        bic : float
            BIC value. Lower is better.

        Notes
        -----
        BIC = k*ln(n) + n*ln(RSS/n)

        where k is number of parameters, n is number of data points,
        and RSS is residual sum of squares.

        BIC penalizes model complexity more heavily than AIC.
        Preferred for larger datasets.
        """
        if not hasattr(self, "ydata"):
            raise AttributeError("Cannot compute BIC: ydata not available")

        n = len(self.ydata)
        k = len(self.popt)
        rss = np.sum(self.residuals**2)

        if rss <= 0:
            warnings.warn("RSS ≤ 0, BIC undefined.")
            return np.nan

        return k * np.log(n) + n * np.log(rss / n)

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
                "Cannot compute confidence intervals: pcov not available. "
                "Try setting full_output=True in curve_fit."
            )

        from scipy import stats

        n = len(self.ydata) if hasattr(self, "ydata") else len(self.residuals)
        p = len(self.popt)

        # Degrees of freedom
        dof = max(n - p, 1)

        # t-value for confidence level
        t_val = stats.t.ppf((1 + alpha) / 2, dof)

        # Standard errors from covariance diagonal
        perr = np.sqrt(np.diag(self.pcov))

        # Confidence intervals
        intervals = np.zeros((p, 2))
        intervals[:, 0] = self.popt - t_val * perr  # Lower bound
        intervals[:, 1] = self.popt + t_val * perr  # Upper bound

        return intervals

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

        Examples
        --------
        >>> result = curve_fit(model, x, y)
        >>> pi = result.prediction_interval()  # Use fitted x values
        >>> pi_new = result.prediction_interval(x_new)  # Use new x values

        Notes
        -----
        Prediction intervals account for both parameter uncertainty (from pcov)
        and inherent data variability (residual variance).
        """
        if not hasattr(self, "model"):
            raise AttributeError(
                "Cannot compute prediction interval: model not available"
            )
        if not hasattr(self, "pcov"):
            raise AttributeError(
                "Cannot compute prediction interval: pcov not available"
            )

        from scipy import stats

        # Use self.xdata if x not provided
        if x is None:
            if not hasattr(self, "xdata"):
                raise AttributeError(
                    "Cannot compute prediction interval: xdata not available. "
                    "Either pass x explicitly or ensure xdata is stored in result."
                )
            x = self.xdata

        # Predictions at x
        y_pred = np.array(self.model(x, *self.popt))

        # Residual variance
        n = len(self.ydata) if hasattr(self, "ydata") else len(self.residuals)
        p = len(self.popt)
        dof = max(n - p, 1)
        s2 = np.sum(self.residuals**2) / dof

        # t-value
        t_val = stats.t.ppf((1 + alpha) / 2, dof)

        # Prediction interval (simplified - assumes diagonal pcov dominance)
        # Full calculation would require Jacobian at new x
        se_pred = np.sqrt(s2 * (1 + np.trace(self.pcov) / p))

        lower = y_pred - t_val * se_pred
        upper = y_pred + t_val * se_pred

        # Return as (n, 2) array
        intervals = np.column_stack([lower, upper])
        return intervals

    def plot(self, ax=None, show_residuals: bool = True, **kwargs):
        """Plot data, fitted curve, and residuals.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show_residuals : bool, optional
            Whether to show residual plot (default: True).
        **kwargs
            Additional keyword arguments passed to plotting functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        axes : matplotlib.axes.Axes or array of Axes
            Axes object(s).

        Examples
        --------
        >>> result = curve_fit(model, x, y)
        >>> result.plot()
        >>> plt.show()

        >>> # Custom styling
        >>> fig, ax = plt.subplots()
        >>> result.plot(ax=ax, show_residuals=False, color='red', alpha=0.7)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from None

        if not hasattr(self, "xdata") or not hasattr(self, "ydata"):
            raise AttributeError("Cannot plot: xdata and ydata not available")

        # Create figure if needed
        if ax is None:
            if show_residuals:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
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

        # Sort for smooth curve plotting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_pred_sorted = self.predictions[sort_idx]

        # Extract plotting parameters from kwargs
        scatter_alpha = kwargs.pop("alpha", 0.6)
        scatter_color = kwargs.pop("color", None)

        # Plot data points
        scatter_kwargs = {"alpha": scatter_alpha, "label": "Data"}
        if scatter_color is not None:
            scatter_kwargs["color"] = scatter_color
        scatter_kwargs.update(kwargs)
        ax1.scatter(x, y, **scatter_kwargs)

        # Plot fitted curve
        fit_color = "red" if scatter_color is None else scatter_color
        ax1.plot(
            x_sorted,
            y_pred_sorted,
            color=fit_color,
            linewidth=2,
            label="Fit",
            zorder=10,
        )

        # Labels and title
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title(f"Curve Fit (R² = {self.r_squared:.4f}, RMSE = {self.rmse:.4f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residual plot
        if show_residuals and ax2 is not None:
            ax2.scatter(x, self.residuals, alpha=0.6, color="gray")
            ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
            ax2.set_xlabel("x")
            ax2.set_ylabel("Residuals")
            ax2.set_title("Residual Plot")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if ax2 is not None:
            return fig, (ax1, ax2)
        else:
            return fig, ax1

    def summary(self):
        """Print statistical summary of fit.

        Displays:
        - Fitted parameters with standard errors
        - Goodness of fit metrics (R², RMSE, MAE)
        - Model selection criteria (AIC, BIC)
        - Convergence information

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

        if hasattr(self, "pcov"):
            perr = np.sqrt(np.diag(self.pcov))
            print(f"{'Parameter':<15} {'Value':>12} {'Std Error':>12} {'95% CI':>25}")
            print("-" * 70)

            ci = self.confidence_intervals(alpha=0.95)
            for i, (val, err, (ci_low, ci_high)) in enumerate(
                zip(self.popt, perr, ci, strict=False)
            ):
                print(
                    f"{'p' + str(i):<15} {val:>12.6f} {err:>12.6f} "
                    f"[{ci_low:>10.6f}, {ci_high:>10.6f}]"
                )
        else:
            print(f"{'Parameter':<15} {'Value':>12}")
            print("-" * 70)
            for i, val in enumerate(self.popt):
                print(f"{'p' + str(i):<15} {val:>12.6f}")

        # Goodness of fit
        print("\nGoodness of Fit:")
        print("-" * 70)
        print(f"R²                : {self.r_squared:>12.6f}")
        print(f"Adjusted R²       : {self.adj_r_squared:>12.6f}")
        print(f"RMSE              : {self.rmse:>12.6f}")
        print(f"MAE               : {self.mae:>12.6f}")

        # Model selection
        print("\nModel Selection Criteria:")
        print("-" * 70)
        print(f"AIC               : {self.aic:>12.2f}")
        print(f"BIC               : {self.bic:>12.2f}")

        # Convergence
        print("\nConvergence Information:")
        print("-" * 70)
        print(f"Success           : {self.success}")
        print(f"Message           : {self.message}")
        print(f"Iterations        : {self.nfev if hasattr(self, 'nfev') else 'N/A'}")
        print(f"Final cost        : {self.cost if hasattr(self, 'cost') else 'N/A'}")
        print(
            f"Optimality        : {self.optimality if hasattr(self, 'optimality') else 'N/A':.6e}"
        )

        print("=" * 70)
