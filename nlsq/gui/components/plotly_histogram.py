"""Plotly residuals histogram component for NLSQ GUI.

This module provides Plotly-based visualization of residual distribution
with a normal distribution overlay for assessing fit quality.

Functions
---------
create_residuals_histogram
    Create a Plotly histogram with normal overlay.
render_residuals_histogram
    Render the histogram in Streamlit.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from nlsq.gui.utils.theme import (
    get_annotation_style,
    get_current_theme,
    get_histogram_colors,
    get_plotly_colors,
    get_plotly_template,
)

# =============================================================================
# Histogram Creation
# =============================================================================


def create_residuals_histogram(
    residuals: np.ndarray,
    n_bins: int | str = "auto",
    show_normal: bool = True,
    show_kde: bool = False,
    title: str = "Residual Distribution",
    x_label: str = "Residual",
    y_label: str = "Density",
    theme: str | None = None,
) -> go.Figure:
    """Create a Plotly histogram of residuals with normal overlay.

    Parameters
    ----------
    residuals : np.ndarray
        Residual values (observed - predicted).
    n_bins : int | str
        Number of bins or 'auto' for automatic selection.
    show_normal : bool
        Whether to show the fitted normal distribution overlay.
    show_kde : bool
        Whether to show kernel density estimate (in addition to normal).
    title : str
        Plot title.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    theme : str | None
        Theme to use ('light' or 'dark'). If None, uses current theme.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> residuals = np.random.normal(0, 0.5, 100)
    >>> fig = create_residuals_histogram(residuals)
    """
    # Get theme colors
    if theme is None:
        theme = get_current_theme()

    colors = get_plotly_colors(theme)
    hist_colors = get_histogram_colors(theme)
    annotation_style = get_annotation_style(theme)

    residuals = np.asarray(residuals).flatten()

    # Remove NaN/Inf values
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) == 0:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No valid residuals to display",
            showarrow=False,
            font={"size": 14, "color": colors["text"]},
        )
        fig.update_layout(
            template=get_plotly_template(theme),
            paper_bgcolor=colors["background"],
            plot_bgcolor=colors["background"],
        )
        return fig

    # Calculate statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)

    # Determine number of bins
    if n_bins == "auto":
        # Use Freedman-Diaconis rule
        iqr = np.percentile(residuals, 75) - np.percentile(residuals, 25)
        if iqr > 0:
            bin_width = 2 * iqr / (len(residuals) ** (1 / 3))
            n_bins = max(10, int((residuals.max() - residuals.min()) / bin_width))
        else:
            n_bins = 30
        n_bins = min(n_bins, 100)  # Cap at 100 bins

    # Create figure
    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=n_bins,
            name="Residuals",
            histnorm="probability density",
            marker={
                "color": hist_colors["bar"],
                "line": {"color": hist_colors["line"], "width": 1},
            },
            hovertemplate="Residual: %{x:.4g}<br>Density: %{y:.4g}<extra></extra>",
        )
    )

    # Generate x values for overlay curves
    x_min = mean_res - 4 * std_res
    x_max = mean_res + 4 * std_res
    x_curve = np.linspace(x_min, x_max, 200)

    # Add normal distribution overlay if requested
    if show_normal and std_res > 0:
        normal_y = stats.norm.pdf(x_curve, loc=mean_res, scale=std_res)

        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=normal_y,
                mode="lines",
                name="Normal",
                line={"color": hist_colors["normal"], "width": 2, "dash": "dash"},
                hovertemplate="x: %{x:.4g}<br>N(mu, sigma): %{y:.4g}<extra></extra>",
            )
        )

    # Add KDE if requested
    if show_kde and len(residuals) >= 10:
        try:
            kde = stats.gaussian_kde(residuals)
            kde_y = kde(x_curve)

            fig.add_trace(
                go.Scatter(
                    x=x_curve,
                    y=kde_y,
                    mode="lines",
                    name="KDE",
                    line={"color": hist_colors["kde"], "width": 2},
                    hovertemplate="x: %{x:.4g}<br>KDE: %{y:.4g}<extra></extra>",
                )
            )
        except (np.linalg.LinAlgError, ValueError):
            # KDE can fail for degenerate cases
            pass

    # Configure layout with theme
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16},
        },
        xaxis={
            "title": x_label,
            "showgrid": True,
            "gridcolor": colors["grid"],
        },
        yaxis={
            "title": y_label,
            "showgrid": True,
            "gridcolor": colors["grid"],
        },
        template=get_plotly_template(theme),
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["background"],
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        height=300,
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "right",
            "x": 0.99,
            "bgcolor": colors["legend_bg"],
        },
        bargap=0.05,
        hovermode="x unified",
        font={"color": colors["text"]},
    )

    # Add statistics annotation
    # Skip skew/kurtosis for constant residuals (std=0) to avoid precision warnings
    skew = stats.skew(residuals) if len(residuals) >= 3 and std_res > 0 else np.nan
    kurt = stats.kurtosis(residuals) if len(residuals) >= 4 and std_res > 0 else np.nan

    stats_text = f"Mean: {mean_res:.4g}<br>Std: {std_res:.4g}"
    if np.isfinite(skew):
        stats_text += f"<br>Skew: {skew:.3f}"
    if np.isfinite(kurt):
        stats_text += f"<br>Kurt: {kurt:.3f}"

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font={"size": 10, "color": colors["text"]},
        align="left",
        xanchor="left",
        yanchor="top",
        bgcolor=annotation_style["bgcolor"],
        bordercolor=annotation_style["bordercolor"],
        borderwidth=1,
        borderpad=4,
    )

    return fig


def create_histogram_from_result(
    result: Any,
    n_bins: int | str = "auto",
    show_normal: bool = True,
    theme: str | None = None,
) -> go.Figure:
    """Create a residuals histogram directly from a CurveFitResult.

    Parameters
    ----------
    result : CurveFitResult
        The curve fitting result object.
    n_bins : int | str
        Number of bins or 'auto'.
    show_normal : bool
        Whether to show normal distribution overlay.
    theme : str | None
        Theme to use ('light' or 'dark'). If None, uses current theme.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    residuals = np.asarray(result.residuals) if hasattr(result, "residuals") else None

    if residuals is None:
        raise ValueError("Result must contain residuals")

    return create_residuals_histogram(
        residuals=residuals,
        n_bins=n_bins,
        show_normal=show_normal,
        theme=theme,
    )


def compute_normality_tests(residuals: np.ndarray) -> dict[str, Any]:
    """Compute normality tests on residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Residual values.

    Returns
    -------
    dict[str, Any]
        Dictionary with test results:
        - shapiro_stat: Shapiro-Wilk statistic
        - shapiro_pvalue: p-value
        - dagostino_stat: D'Agostino-Pearson statistic
        - dagostino_pvalue: p-value
        - is_normal: bool (True if likely normal at alpha=0.05)
    """
    residuals = np.asarray(residuals).flatten()
    residuals = residuals[np.isfinite(residuals)]

    results: dict[str, Any] = {
        "shapiro_stat": np.nan,
        "shapiro_pvalue": np.nan,
        "dagostino_stat": np.nan,
        "dagostino_pvalue": np.nan,
        "is_normal": None,
    }

    if len(residuals) < 8:
        return results

    # Shapiro-Wilk test (good for n < 5000)
    if len(residuals) <= 5000:
        try:
            stat, pvalue = stats.shapiro(residuals)
            results["shapiro_stat"] = float(stat)
            results["shapiro_pvalue"] = float(pvalue)
        except (ValueError, np.linalg.LinAlgError):
            pass

    # D'Agostino-Pearson test (requires n >= 20)
    if len(residuals) >= 20:
        try:
            stat, pvalue = stats.normaltest(residuals)
            results["dagostino_stat"] = float(stat)
            results["dagostino_pvalue"] = float(pvalue)
        except (ValueError, np.linalg.LinAlgError):
            pass

    # Determine if likely normal (use Shapiro if available, else D'Agostino)
    alpha = 0.05
    if np.isfinite(results["shapiro_pvalue"]):
        results["is_normal"] = results["shapiro_pvalue"] > alpha
    elif np.isfinite(results["dagostino_pvalue"]):
        results["is_normal"] = results["dagostino_pvalue"] > alpha

    return results


# =============================================================================
# Streamlit Rendering
# =============================================================================


def render_residuals_histogram(
    residuals: np.ndarray | None = None,
    result: Any = None,
    n_bins: int | str = "auto",
    show_normal: bool = True,
    show_kde: bool = False,
    show_normality_test: bool = True,
    title: str = "Residual Distribution",
    container: Any = None,
    key: str = "residuals_histogram",
) -> None:
    """Render the residuals histogram in Streamlit.

    Can be called either with explicit array or with a result object.
    Uses the current theme from session state.

    Parameters
    ----------
    residuals : np.ndarray | None
        Residual values.
    result : CurveFitResult | None
        Alternatively, provide a result object.
    n_bins : int | str
        Number of bins or 'auto'.
    show_normal : bool
        Whether to show normal distribution overlay.
    show_kde : bool
        Whether to show kernel density estimate.
    show_normality_test : bool
        Whether to show normality test results.
    title : str
        Plot title.
    container : Any, optional
        Streamlit container to render in.
    key : str
        Unique key for the Streamlit component.

    Examples
    --------
    >>> import streamlit as st
    >>> result = curve_fit(model, x, y)
    >>> render_residuals_histogram(result=result)
    """
    import streamlit as st

    target = container if container is not None else st

    # Get current theme
    theme = get_current_theme()

    # Get residuals from result or explicit array
    if result is not None:
        try:
            residuals = np.asarray(result.residuals)
        except (AttributeError, TypeError):
            target.warning("Cannot extract residuals from result")
            return
    elif residuals is None:
        target.warning("No residuals available for histogram")
        return

    # Create and render histogram
    try:
        fig = create_residuals_histogram(
            residuals=residuals,
            n_bins=n_bins,
            show_normal=show_normal,
            show_kde=show_kde,
            title=title,
            theme=theme,
        )
    except Exception as e:
        target.warning(f"Cannot create histogram: {e}")
        return

    target.plotly_chart(
        fig,
        width="stretch",
        key=key,
    )

    # Show normality test results if requested
    if show_normality_test:
        tests = compute_normality_tests(residuals)

        if tests["is_normal"] is not None:
            if tests["is_normal"]:
                target.success(
                    "Residuals appear normally distributed (Shapiro-Wilk p > 0.05)"
                )
            else:
                target.warning(
                    "Residuals may not be normally distributed (Shapiro-Wilk p < 0.05)"
                )

            # Show detailed test results in expander
            with target.expander("Normality Test Details"):
                if np.isfinite(tests["shapiro_pvalue"]):
                    st.markdown(
                        f"**Shapiro-Wilk Test:** W = {tests['shapiro_stat']:.4f}, "
                        f"p = {tests['shapiro_pvalue']:.4f}"
                    )
                if np.isfinite(tests["dagostino_pvalue"]):
                    st.markdown(
                        f"**D'Agostino-Pearson Test:** K^2 = {tests['dagostino_stat']:.4f}, "
                        f"p = {tests['dagostino_pvalue']:.4f}"
                    )
                st.caption(
                    "A p-value > 0.05 suggests the residuals are normally distributed, "
                    "which is an assumption of least squares fitting."
                )
