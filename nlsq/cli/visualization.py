"""Publication-quality visualization module for NLSQ CLI.

This module provides the FitVisualizer class for generating publication-quality
plots of curve fitting results, including:
- Combined main plot + residuals layout
- Separate histogram of residuals
- Confidence bands from covariance matrix error propagation
- Multiple style presets (publication, presentation, nature, science, minimal)
- Multi-format output (PDF vector, PNG raster)
- Fit statistics annotation (R-squared, RMSE)
- Colorblind-safe palette support

Example Usage
-------------
>>> from nlsq.cli.visualization import FitVisualizer
>>>
>>> visualizer = FitVisualizer()
>>> result = {"popt": [1.0, 0.5, 0.1], "pcov": [[0.01, 0, 0], [0, 0.02, 0], [0, 0, 0.005]], ...}
>>> data = {"xdata": x, "ydata": y, "sigma": sigma}
>>> config = {"visualization": {"enabled": True, "output_dir": "figures", ...}}
>>> output_paths = visualizer.generate(result, data, model, config)
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CLI use

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# =============================================================================
# Style Presets Dictionary
# =============================================================================

STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "publication": {
        # Clean serif fonts, 300 DPI, standard figure size
        "font_family": "serif",
        "font_size": 10,
        "math_fontset": "cm",
        "dpi": 300,
        "figure_size": [6.0, 4.5],
        "grid": True,
        "grid_alpha": 0.3,
        "spine_visibility": {"top": True, "right": True, "bottom": True, "left": True},
        "linewidth": 1.5,
    },
    "presentation": {
        # Larger sans-serif fonts, lower DPI for slides
        "font_family": "sans-serif",
        "font_size": 14,
        "math_fontset": "dejavusans",
        "dpi": 150,
        "figure_size": [10.0, 7.5],
        "grid": True,
        "grid_alpha": 0.4,
        "spine_visibility": {"top": True, "right": True, "bottom": True, "left": True},
        "linewidth": 2.0,
    },
    "nature": {
        # Nature journal specs: 3.5" width (single column), Arial font
        "font_family": "sans-serif",
        "font_size": 8,
        "math_fontset": "dejavusans",
        "dpi": 300,
        "figure_size": [3.5, 2.625],  # Single column width, 4:3 aspect
        "grid": False,
        "grid_alpha": 0.0,
        "spine_visibility": {
            "top": False,
            "right": False,
            "bottom": True,
            "left": True,
        },
        "linewidth": 1.0,
    },
    "science": {
        # Science journal specifications
        "font_family": "sans-serif",
        "font_size": 9,
        "math_fontset": "dejavusans",
        "dpi": 300,
        "figure_size": [3.5, 2.625],
        "grid": False,
        "grid_alpha": 0.0,
        "spine_visibility": {
            "top": False,
            "right": False,
            "bottom": True,
            "left": True,
        },
        "linewidth": 1.0,
    },
    "minimal": {
        # No top/right spines, no grid, clean look
        "font_family": "sans-serif",
        "font_size": 10,
        "math_fontset": "dejavusans",
        "dpi": 300,
        "figure_size": [6.0, 4.5],
        "grid": False,
        "grid_alpha": 0.0,
        "spine_visibility": {
            "top": False,
            "right": False,
            "bottom": True,
            "left": True,
        },
        "linewidth": 1.5,
    },
}


# =============================================================================
# FitVisualizer Class
# =============================================================================


class FitVisualizer:
    """Visualizer for curve fitting results.

    Generates publication-quality plots including combined fit + residuals
    layouts, histograms, and confidence bands.

    Attributes
    ----------
    None

    Methods
    -------
    generate(result, data, model, config)
        Generate all configured visualizations and save to files.

    Examples
    --------
    >>> visualizer = FitVisualizer()
    >>> result = {"popt": [1.0, 0.5], "pcov": [[0.01, 0], [0, 0.02]], ...}
    >>> data = {"xdata": x, "ydata": y}
    >>> config = {"visualization": {"enabled": True, "output_dir": "figures"}}
    >>> output_paths = visualizer.generate(result, data, model, config)
    """

    def generate(
        self,
        result: dict[str, Any],
        data: dict[str, Any],
        model: Callable,
        config: dict[str, Any],
    ) -> list[str]:
        """Generate visualizations based on configuration.

        Parameters
        ----------
        result : dict
            Fit result dictionary containing:
            - popt: Fitted parameters
            - pcov: Covariance matrix
            - fun: Residuals (optional)
            - statistics: Dict with r_squared, rmse, etc.
        data : dict
            Data dictionary containing:
            - xdata: Independent variable array
            - ydata: Dependent variable array
            - sigma: Uncertainties (optional)
        model : callable
            Model function ``f(x, *params)``.
        config : dict
            Configuration dictionary with visualization section.

        Returns
        -------
        list[str]
            List of output file paths that were generated.
        """
        vis_config = config.get("visualization", {})

        if not vis_config.get("enabled", True):
            return []

        output_paths: list[str] = []
        output_dir = Path(vis_config.get("output_dir", "figures"))
        output_dir.mkdir(parents=True, exist_ok=True)

        filename_prefix = vis_config.get("filename_prefix", "fit")
        formats = vis_config.get("formats", ["pdf", "png"])

        # Apply style preset
        self._apply_style_preset(vis_config)

        # Generate combined plot (main + residuals)
        fig_combined = self._create_combined_figure(result, data, model, vis_config)
        combined_paths = self._save_figure(
            fig_combined, output_dir, f"{filename_prefix}_combined", formats, vis_config
        )
        output_paths.extend(combined_paths)
        plt.close(fig_combined)

        # Generate histogram if enabled
        histogram_config = vis_config.get("histogram", {})
        if histogram_config.get("enabled", False):
            residuals = self._get_residuals(result, data, model)
            if residuals is not None:
                fig_hist = self._create_histogram_figure(residuals, vis_config)
                hist_paths = self._save_figure(
                    fig_hist,
                    output_dir,
                    f"{filename_prefix}_histogram",
                    formats,
                    vis_config,
                )
                output_paths.extend(hist_paths)
                plt.close(fig_hist)

        return output_paths

    def _apply_style_preset(self, config: dict[str, Any]) -> None:
        """Apply style preset to matplotlib rcParams.

        Parameters
        ----------
        config : dict
            Visualization configuration containing style preset name.
        """
        style_name = config.get("style", "publication")
        preset = STYLE_PRESETS.get(style_name, STYLE_PRESETS["publication"])

        # Apply font settings
        plt.rcParams["font.family"] = preset.get("font_family", "serif")
        plt.rcParams["font.size"] = preset.get("font_size", 10)
        plt.rcParams["mathtext.fontset"] = preset.get("math_fontset", "cm")

        # Apply line settings
        plt.rcParams["lines.linewidth"] = preset.get("linewidth", 1.5)

        # Override with config-specific font settings if provided
        font_config = config.get("font", {})
        if "family" in font_config:
            plt.rcParams["font.family"] = font_config["family"]
        if "size" in font_config:
            plt.rcParams["font.size"] = font_config["size"]
        if "math_fontset" in font_config:
            plt.rcParams["mathtext.fontset"] = font_config["math_fontset"]

    def _get_color_scheme(self, config: dict[str, Any]) -> dict[str, str]:
        """Get the active color scheme from configuration.

        Parameters
        ----------
        config : dict
            Visualization configuration.

        Returns
        -------
        dict
            Dictionary mapping element names to hex colors.
        """
        active_scheme = config.get("active_scheme", "default")
        color_schemes = config.get("color_schemes", {})

        # Default fallback colors
        default_colors = {
            "data": "#1f77b4",
            "fit": "#d62728",
            "residuals": "#2ca02c",
            "confidence": "#ff7f0e",
        }

        return color_schemes.get(active_scheme, default_colors)

    def _create_combined_figure(
        self,
        result: dict[str, Any],
        data: dict[str, Any],
        model: Callable,
        config: dict[str, Any],
    ) -> plt.Figure:
        """Create combined figure with main plot and residuals.

        Parameters
        ----------
        result : dict
            Fit result dictionary.
        data : dict
            Data dictionary.
        model : callable
            Model function.
        config : dict
            Visualization configuration.

        Returns
        -------
        matplotlib.figure.Figure
            The combined figure with two subplots.
        """
        style_name = config.get("style", "publication")
        preset = STYLE_PRESETS.get(style_name, STYLE_PRESETS["publication"])

        # Get figure size from config or preset
        figure_size = config.get("figure_size", preset.get("figure_size", [6.0, 4.5]))

        # Check if residuals plot is enabled
        residuals_config = config.get("residuals_plot", {})
        show_residuals = residuals_config.get("enabled", True)

        if show_residuals:
            # Create figure with 2 subplots (3:1 height ratio)
            # Use constrained layout for better handling of shared axes
            fig, (ax_main, ax_residuals) = plt.subplots(
                2,
                1,
                figsize=figure_size,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
                sharex=True,
                layout="constrained",
            )
        else:
            fig, ax_main = plt.subplots(figsize=figure_size, layout="constrained")
            ax_residuals = None

        # Get color scheme
        colors = self._get_color_scheme(config)

        # Extract data
        xdata = np.asarray(data.get("xdata", []))
        ydata = np.asarray(data.get("ydata", []))
        sigma = data.get("sigma")
        if sigma is not None:
            sigma = np.asarray(sigma)

        popt = np.asarray(result.get("popt", []))
        pcov = result.get("pcov")
        if pcov is not None:
            pcov = np.asarray(pcov)

        # Plot main figure
        self._plot_main(
            ax_main, xdata, ydata, sigma, popt, pcov, model, config, colors, result
        )

        # Plot residuals if enabled
        if ax_residuals is not None:
            residuals = self._get_residuals(result, data, model)
            if residuals is not None:
                self._plot_residuals(ax_residuals, xdata, residuals, config, colors)

        # Apply spine visibility
        spine_vis = preset.get("spine_visibility", {})
        for spine, visible in spine_vis.items():
            if spine in ax_main.spines:
                ax_main.spines[spine].set_visible(visible)
            if ax_residuals is not None and spine in ax_residuals.spines:
                ax_residuals.spines[spine].set_visible(visible)

        return fig

    def _plot_main(
        self,
        ax: plt.Axes,
        xdata: np.ndarray,
        ydata: np.ndarray,
        sigma: np.ndarray | None,
        popt: np.ndarray,
        pcov: np.ndarray | None,
        model: Callable,
        config: dict[str, Any],
        colors: dict[str, str],
        result: dict[str, Any],
    ) -> None:
        """Plot main fit data and curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        xdata : ndarray
            X data points.
        ydata : ndarray
            Y data points.
        sigma : ndarray or None
            Y uncertainties.
        popt : ndarray
            Fitted parameters.
        pcov : ndarray or None
            Covariance matrix.
        model : callable
            Model function.
        config : dict
            Visualization configuration.
        colors : dict
            Color scheme dictionary.
        result : dict
            Fit result dictionary (for statistics annotation).
        """
        main_config = config.get("main_plot", {})
        data_config = main_config.get("data", {})
        fit_config = main_config.get("fit", {})
        confidence_config = main_config.get("confidence_band", {})

        # Get data color from config or color scheme
        data_color = data_config.get("color", colors.get("data", "#1f77b4"))
        fit_color = fit_config.get("color", colors.get("fit", "#d62728"))

        # Plot data points
        marker = data_config.get("marker", "o")
        size = data_config.get("size", 20)
        alpha = data_config.get("alpha", 0.7)
        data_label = data_config.get("label", "Data")

        if sigma is not None and data_config.get("show_errorbars", True):
            ax.errorbar(
                xdata,
                ydata,
                yerr=sigma,
                fmt=marker,
                color=data_color,
                markersize=np.sqrt(size),
                alpha=alpha,
                label=data_label,
                capsize=data_config.get("capsize", 2),
            )
        else:
            ax.scatter(
                xdata,
                ydata,
                marker=marker,
                s=size,
                c=data_color,
                alpha=alpha,
                label=data_label,
            )

        # Generate fit curve
        n_fit_points = fit_config.get("n_points", 500)
        x_fit = np.linspace(xdata.min(), xdata.max(), n_fit_points)

        if len(popt) > 0:
            y_fit = model(x_fit, *popt)

            # Plot confidence band if enabled
            if confidence_config.get("enabled", False) and pcov is not None:
                confidence_level = confidence_config.get("level", 0.95)
                confidence_color = confidence_config.get(
                    "color", colors.get("confidence", fit_color)
                )
                confidence_alpha = confidence_config.get("alpha", 0.2)

                lower, upper = self._calculate_confidence_band(
                    model, x_fit, popt, pcov, confidence_level
                )
                ax.fill_between(
                    x_fit,
                    lower,
                    upper,
                    color=confidence_color,
                    alpha=confidence_alpha,
                    label=f"{confidence_level * 100:.0f}% CI",
                )

            # Plot fit curve
            ax.plot(
                x_fit,
                y_fit,
                color=fit_color,
                linewidth=fit_config.get("linewidth", 1.5),
                linestyle=fit_config.get("linestyle", "-"),
                label=fit_config.get("label", "Fit"),
            )

        # Set labels
        ax.set_xlabel(main_config.get("x_label", "x"))
        ax.set_ylabel(main_config.get("y_label", "y"))

        if main_config.get("title"):
            ax.set_title(main_config["title"])

        # Grid
        if main_config.get("show_grid", True):
            ax.grid(True, alpha=main_config.get("grid_alpha", 0.3))

        # Legend
        legend_config = main_config.get("legend", {})
        if legend_config.get("enabled", True):
            ax.legend(
                loc=legend_config.get("location", "best"),
                frameon=legend_config.get("frameon", True),
                fontsize=legend_config.get("fontsize"),
            )

        # Annotation (fit statistics)
        annotation_config = main_config.get("annotation", {})
        if annotation_config.get("enabled", False):
            self._add_statistics_annotation(ax, result, annotation_config)

    def _plot_residuals(
        self,
        ax: plt.Axes,
        xdata: np.ndarray,
        residuals: np.ndarray,
        config: dict[str, Any],
        colors: dict[str, str],
    ) -> None:
        """Plot residuals subplot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        xdata : ndarray
            X data points.
        residuals : ndarray
            Residual values.
        config : dict
            Visualization configuration.
        colors : dict
            Color scheme dictionary.
        """
        residuals_config = config.get("residuals_plot", {})

        # Get color from config or scheme
        residual_color = residuals_config.get(
            "color", colors.get("residuals", "#2ca02c")
        )

        # Plot residuals
        plot_type = residuals_config.get("type", "scatter")
        marker = residuals_config.get("marker", "o")
        size = residuals_config.get("size", 15)
        alpha = residuals_config.get("alpha", 0.7)

        if plot_type == "scatter":
            ax.scatter(
                xdata, residuals, marker=marker, s=size, c=residual_color, alpha=alpha
            )
        elif plot_type == "stem":
            markerline, stemlines, baseline = ax.stem(xdata, residuals)
            plt.setp(markerline, color=residual_color, markersize=np.sqrt(size))
            plt.setp(stemlines, color=residual_color, alpha=alpha)
            plt.setp(baseline, visible=False)
        else:  # line
            ax.plot(xdata, residuals, marker=marker, color=residual_color, alpha=alpha)

        # Zero reference line
        if residuals_config.get("show_zero_line", True):
            ax.axhline(
                0,
                linestyle=residuals_config.get("zero_line_style", "--"),
                color=residuals_config.get("zero_line_color", "gray"),
                linewidth=residuals_config.get("zero_line_width", 1.0),
            )

        # Standard deviation bands
        std_config = residuals_config.get("std_bands", {})
        if std_config.get("enabled", False):
            std_residual = np.std(residuals)
            levels = std_config.get("levels", [1, 2])
            band_colors = std_config.get("colors", ["#fff3cd", "#ffe69c"])
            band_alpha = std_config.get("alpha", 0.4)

            for i, level in enumerate(reversed(levels)):
                color = band_colors[min(i, len(band_colors) - 1)]
                ax.axhspan(
                    -level * std_residual,
                    level * std_residual,
                    color=color,
                    alpha=band_alpha,
                    zorder=0,
                )

        # Labels
        ax.set_xlabel(residuals_config.get("x_label", "x"))
        ax.set_ylabel(residuals_config.get("y_label", "Residual"))

        if residuals_config.get("title"):
            ax.set_title(residuals_config["title"])

    def _create_histogram_figure(
        self,
        residuals: np.ndarray,
        config: dict[str, Any],
    ) -> plt.Figure:
        """Create histogram of residuals.

        Parameters
        ----------
        residuals : ndarray
            Residual values.
        config : dict
            Visualization configuration.

        Returns
        -------
        matplotlib.figure.Figure
            The histogram figure.
        """
        style_name = config.get("style", "publication")
        preset = STYLE_PRESETS.get(style_name, STYLE_PRESETS["publication"])
        figure_size = config.get("figure_size", preset.get("figure_size", [6.0, 4.5]))

        fig, ax = plt.subplots(figsize=figure_size, layout="constrained")

        histogram_config = config.get("histogram", {})

        # Get bins
        bins = histogram_config.get("bins", "auto")
        if bins == "sqrt":
            bins = int(np.sqrt(len(residuals)))
        elif bins == "sturges":
            bins = int(np.ceil(np.log2(len(residuals))) + 1)

        # Get colors
        colors = self._get_color_scheme(config)
        bar_color = histogram_config.get("color", colors.get("residuals", "#9467bd"))
        bar_alpha = histogram_config.get("alpha", 0.7)
        edgecolor = histogram_config.get("edgecolor", "white")

        # Plot histogram
        _n, _bin_edges, _patches = ax.hist(
            residuals,
            bins=bins,
            color=bar_color,
            alpha=bar_alpha,
            edgecolor=edgecolor,
            density=True,
        )

        # Overlay normal distribution fit
        if histogram_config.get("show_normal_fit", True):
            mu, std = stats.norm.fit(residuals)
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            y_norm = stats.norm.pdf(x_norm, mu, std)

            normal_color = histogram_config.get(
                "normal_color", colors.get("fit", "#d62728")
            )
            ax.plot(x_norm, y_norm, color=normal_color, linewidth=2, label="Normal fit")
            ax.legend()

        # Labels
        ax.set_xlabel(histogram_config.get("x_label", "Residual"))
        ax.set_ylabel(histogram_config.get("y_label", "Frequency"))
        if histogram_config.get("title"):
            ax.set_title(histogram_config["title"])

        return fig

    def _calculate_confidence_band(
        self,
        model: Callable,
        x: np.ndarray,
        popt: np.ndarray,
        pcov: np.ndarray,
        confidence_level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate confidence bands using error propagation.

        Uses the Jacobian of the model with respect to parameters and
        the covariance matrix to compute prediction uncertainties.

        Parameters
        ----------
        model : callable
            Model function ``f(x, *params)``.
        x : ndarray
            X values for computing the band.
        popt : ndarray
            Fitted parameters.
        pcov : ndarray
            Parameter covariance matrix.
        confidence_level : float
            Confidence level (default 0.95 for 95% CI).

        Returns
        -------
        tuple[ndarray, ndarray]
            Lower and upper bounds of the confidence band.
        """
        n_points = len(x)
        n_params = len(popt)

        # Compute Jacobian numerically using finite differences
        eps = 1e-8
        jacobian = np.zeros((n_points, n_params))

        y0 = model(x, *popt)

        for i in range(n_params):
            params_plus = popt.copy()
            params_plus[i] += eps
            y_plus = model(x, *params_plus)
            jacobian[:, i] = (y_plus - y0) / eps

        # Compute variance of predictions: var(y) = J @ pcov @ J.T (diagonal elements)
        # For efficiency, compute element-wise
        variance = np.zeros(n_points)
        for i in range(n_points):
            j_i = jacobian[i, :]
            variance[i] = j_i @ pcov @ j_i

        std_prediction = np.sqrt(np.maximum(variance, 0))

        # Compute confidence interval using t-distribution
        # For large samples, use normal distribution
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha / 2)

        lower = y0 - z * std_prediction
        upper = y0 + z * std_prediction

        return lower, upper

    def _get_residuals(
        self,
        result: dict[str, Any],
        data: dict[str, Any],
        model: Callable,
    ) -> np.ndarray | None:
        """Extract or compute residuals from result.

        Parameters
        ----------
        result : dict
            Fit result dictionary.
        data : dict
            Data dictionary.
        model : callable
            Model function.

        Returns
        -------
        ndarray or None
            Residual values, or None if cannot be computed.
        """
        # Try to get from result
        if "fun" in result and result["fun"] is not None:
            return np.asarray(result["fun"])

        # Compute from data and fit
        popt = result.get("popt")
        if popt is None:
            return None

        xdata = data.get("xdata")
        ydata = data.get("ydata")
        if xdata is None or ydata is None:
            return None

        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)
        popt = np.asarray(popt)

        try:
            y_fit = model(xdata, *popt)
            return ydata - y_fit
        except Exception:
            return None

    def _add_statistics_annotation(
        self,
        ax: plt.Axes,
        result: dict[str, Any],
        annotation_config: dict[str, Any],
    ) -> None:
        """Add fit statistics annotation to plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to annotate.
        result : dict
            Fit result dictionary containing statistics.
        annotation_config : dict
            Annotation configuration.
        """
        lines = []

        statistics = result.get("statistics", {})

        if annotation_config.get("show_r_squared", True):
            r_squared = statistics.get("r_squared")
            if r_squared is not None:
                lines.append(f"$R^2 = {r_squared:.4f}$")

        if annotation_config.get("show_rmse", False):
            rmse = statistics.get("rmse")
            if rmse is not None:
                lines.append(f"RMSE = {rmse:.4g}")

        if annotation_config.get("show_chi_squared", False):
            chi_sq = statistics.get("chi_squared")
            if chi_sq is not None:
                lines.append(f"$\\chi^2 = {chi_sq:.4g}$")

        if not lines:
            return

        text = "\n".join(lines)
        fontsize = annotation_config.get("fontsize", 9)
        location = annotation_config.get("location", "upper right")

        # Map location string to axes coordinates
        location_map = {
            "upper right": (0.95, 0.95),
            "upper left": (0.05, 0.95),
            "lower right": (0.95, 0.05),
            "lower left": (0.05, 0.05),
            "center": (0.5, 0.5),
        }
        coords = location_map.get(location, (0.95, 0.95))

        # Determine alignment based on position
        ha = "right" if coords[0] > 0.5 else "left"
        va = "top" if coords[1] > 0.5 else "bottom"

        ax.text(
            coords[0],
            coords[1],
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment=va,
            horizontalalignment=ha,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    def _save_figure(
        self,
        fig: plt.Figure,
        output_dir: Path,
        filename_base: str,
        formats: list[str],
        config: dict[str, Any],
    ) -> list[str]:
        """Save figure to multiple formats.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        output_dir : Path
            Output directory path.
        filename_base : str
            Base filename without extension.
        formats : list[str]
            List of format extensions (e.g., ["pdf", "png"]).
        config : dict
            Visualization configuration.

        Returns
        -------
        list[str]
            List of saved file paths.
        """
        style_name = config.get("style", "publication")
        preset = STYLE_PRESETS.get(style_name, STYLE_PRESETS["publication"])
        dpi = config.get("dpi", preset.get("dpi", 300))

        output_paths = []

        for fmt in formats:
            output_path = output_dir / f"{filename_base}.{fmt}"
            fig.savefig(
                output_path,
                format=fmt,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            output_paths.append(str(output_path))

        return output_paths
