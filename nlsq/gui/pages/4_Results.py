"""Results Page for NLSQ GUI.

This page displays the fitting results including:
- Fitted parameters with uncertainties and confidence intervals
- Fit quality statistics (R^2, RMSE, MAE, AIC, BIC)
- Convergence information
- Interactive visualizations (fit plot, residuals, histogram)
- Export options

The page is only available after a successful curve fit.
"""

from typing import Any

import numpy as np
import streamlit as st

from nlsq.gui.components.fit_statistics import (
    format_convergence_info,
    format_statistics,
    render_fit_statistics,
)
from nlsq.gui.components.param_config import get_param_names_from_model
from nlsq.gui.components.param_results import (
    format_parameter_table,
    render_parameter_results,
)
from nlsq.gui.components.plotly_fit_plot import (
    render_fit_plot,
)
from nlsq.gui.components.plotly_histogram import (
    render_residuals_histogram,
)
from nlsq.gui.components.plotly_residuals import (
    render_residuals_plot,
)
from nlsq.gui.state import SessionState, initialize_state
from nlsq.gui.utils.theme import apply_dark_theme_css

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Results - NLSQ",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


# =============================================================================
# Session State
# =============================================================================


def get_session_state() -> SessionState:
    """Get or initialize the session state."""
    if "nlsq_state" not in st.session_state:
        st.session_state.nlsq_state = initialize_state()
    return st.session_state.nlsq_state


def get_fit_result() -> Any:
    """Get the fit result from session state."""
    state = get_session_state()
    return state.fit_result


def get_current_model() -> Any:
    """Get the current model from session state."""
    return st.session_state.get("current_model", None)


# =============================================================================
# Results Check
# =============================================================================


def has_valid_result() -> bool:
    """Check if there is a valid fit result to display."""
    result = get_fit_result()
    if result is None:
        return False

    # Check for required attributes
    return hasattr(result, "popt")


# =============================================================================
# Parameter Results Section
# =============================================================================


def render_parameters_section() -> None:
    """Render the fitted parameters section."""
    result = get_fit_result()
    model = get_current_model()

    # Get parameter names
    param_names = None
    if model is not None:
        param_names = get_param_names_from_model(model)

    st.markdown("## Fitted Parameters")

    render_parameter_results(
        result=result,
        param_names=param_names,
        show_ci=True,
        alpha=0.05,
    )


# =============================================================================
# Statistics Section
# =============================================================================


def render_statistics_section() -> None:
    """Render the fit statistics section."""
    result = get_fit_result()

    st.markdown("## Fit Statistics")

    render_fit_statistics(
        result=result,
        show_convergence=True,
    )


# =============================================================================
# Visualization Section
# =============================================================================


def render_visualizations_section() -> None:
    """Render the visualization section with tabs for different plots."""
    result = get_fit_result()
    state = get_session_state()

    st.markdown("## Visualizations")

    # Create tabs for different visualizations
    tab_fit, tab_residuals, tab_histogram = st.tabs(
        ["Fit Plot", "Residuals", "Histogram"]
    )

    with tab_fit:
        st.markdown("### Data and Fitted Curve")

        # Options for fit plot
        col1, col2 = st.columns([3, 1])

        with col2:
            show_confidence = st.checkbox(
                "Show confidence band",
                value=True,
                key="fit_show_confidence",
                help="Display 95% prediction interval",
            )

        with col1:
            try:
                render_fit_plot(
                    result=result,
                    x_label="x",
                    y_label="y",
                    key="results_fit_plot",
                )
            except (ValueError, AttributeError) as e:
                st.warning(f"Cannot display fit plot: {e}")

    with tab_residuals:
        st.markdown("### Residual Analysis")

        # Options for residuals plot
        col1, col2, col3 = st.columns([2, 1, 1])

        with col2:
            show_std_bands = st.checkbox(
                "Show std bands",
                value=True,
                key="residuals_show_std",
                help="Display 1, 2, 3 standard deviation bands",
            )

        with col3:
            show_zero = st.checkbox(
                "Show zero line",
                value=True,
                key="residuals_show_zero",
            )

        with col1:
            try:
                render_residuals_plot(
                    result=result,
                    show_std_bands=show_std_bands,
                    show_zero_line=show_zero,
                    key="results_residuals_plot",
                )
            except (ValueError, AttributeError) as e:
                st.warning(f"Cannot display residuals plot: {e}")

        # Residual statistics
        if hasattr(result, "residuals"):
            residuals = np.asarray(result.residuals)
            st.markdown("**Residual Statistics**")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean", f"{np.mean(residuals):.4g}")
            with col2:
                st.metric("Std Dev", f"{np.std(residuals):.4g}")
            with col3:
                st.metric("Min", f"{np.min(residuals):.4g}")
            with col4:
                st.metric("Max", f"{np.max(residuals):.4g}")

    with tab_histogram:
        st.markdown("### Residual Distribution")

        # Options for histogram
        col1, col2, col3 = st.columns([2, 1, 1])

        with col2:
            show_normal = st.checkbox(
                "Show normal overlay",
                value=True,
                key="hist_show_normal",
            )

        with col3:
            show_normality_test = st.checkbox(
                "Show normality test",
                value=True,
                key="hist_show_test",
            )

        with col1:
            try:
                render_residuals_histogram(
                    result=result,
                    show_normal=show_normal,
                    show_normality_test=show_normality_test,
                    key="results_histogram",
                )
            except (ValueError, AttributeError) as e:
                st.warning(f"Cannot display histogram: {e}")


# =============================================================================
# Export Section
# =============================================================================


def render_export_section() -> None:
    """Render the export options section."""
    result = get_fit_result()
    model = get_current_model()
    state = get_session_state()

    st.markdown("## Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Parameters**")

        # Get parameter names
        param_names = None
        if model is not None:
            param_names = get_param_names_from_model(model)

        # Create parameter table for export
        try:
            df = format_parameter_table(
                popt=result.popt,
                pcov=result.pcov,
                param_names=param_names,
            )

            csv_params = df.to_csv(index=False)
            st.download_button(
                label="Download Parameters (CSV)",
                data=csv_params,
                file_name="nlsq_parameters.csv",
                mime="text/csv",
                key="download_params",
            )
        except Exception as e:
            st.warning(f"Cannot export parameters: {e}")

    with col2:
        st.markdown("**Statistics**")

        try:
            stats = format_statistics(result)
            info = format_convergence_info(result)

            import pandas as pd

            df_stats = pd.DataFrame(
                {
                    "Metric": [
                        "R-squared",
                        "Adj. R-squared",
                        "RMSE",
                        "MAE",
                        "AIC",
                        "BIC",
                        "Success",
                        "Message",
                        "Function Evaluations",
                        "Final Cost",
                    ],
                    "Value": [
                        stats["r_squared"],
                        stats["adj_r_squared"],
                        stats["rmse"],
                        stats["mae"],
                        stats["aic"],
                        stats["bic"],
                        info["success_str"],
                        info["message"],
                        info["nfev_str"],
                        info["cost_str"],
                    ],
                }
            )

            csv_stats = df_stats.to_csv(index=False)
            st.download_button(
                label="Download Statistics (CSV)",
                data=csv_stats,
                file_name="nlsq_statistics.csv",
                mime="text/csv",
                key="download_stats",
            )
        except Exception as e:
            st.warning(f"Cannot export statistics: {e}")

    with col3:
        st.markdown("**Full Report**")

        try:
            # Generate text report
            report_lines = [
                "NLSQ Curve Fitting Report",
                "=" * 50,
                "",
                "Fitted Parameters:",
                "-" * 30,
            ]

            popt = result.popt
            pcov = result.pcov
            names = param_names or [f"p{i}" for i in range(len(popt))]
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * len(popt)

            for name, val, err in zip(names, popt, perr, strict=False):
                if np.isfinite(err):
                    report_lines.append(f"  {name}: {val:.6g} +/- {err:.6g}")
                else:
                    report_lines.append(f"  {name}: {val:.6g}")

            report_lines.extend(
                [
                    "",
                    "Fit Statistics:",
                    "-" * 30,
                    f"  R-squared: {stats['r_squared']}",
                    f"  RMSE: {stats['rmse']}",
                    f"  MAE: {stats['mae']}",
                    f"  AIC: {stats['aic']}",
                    f"  BIC: {stats['bic']}",
                    "",
                    "Convergence:",
                    "-" * 30,
                    f"  Success: {info['success_str']}",
                    f"  Message: {info['message']}",
                    f"  Function Evaluations: {info['nfev_str']}",
                    f"  Final Cost: {info['cost_str']}",
                ]
            )

            report_text = "\n".join(report_lines)

            st.download_button(
                label="Download Report (TXT)",
                data=report_text,
                file_name="nlsq_report.txt",
                mime="text/plain",
                key="download_report",
            )
        except Exception as e:
            st.warning(f"Cannot generate report: {e}")


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar() -> None:
    """Render the sidebar with quick stats and navigation."""
    result = get_fit_result()

    st.sidebar.subheader("Quick Summary")

    if result is not None:
        # Success indicator
        if hasattr(result, "success") and result.success:
            st.sidebar.success("Fit Converged")
        else:
            st.sidebar.warning("Fit may not have converged")

        # Key metrics
        try:
            r2 = float(result.r_squared)
            st.sidebar.metric("R-squared", f"{r2:.4f}")
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            rmse = float(result.rmse)
            st.sidebar.metric("RMSE", f"{rmse:.4g}")
        except (AttributeError, TypeError, ValueError):
            pass

        # Number of parameters
        if hasattr(result, "popt"):
            st.sidebar.metric("Parameters", len(result.popt))

    st.sidebar.divider()

    # Navigation hints
    st.sidebar.caption("Use the tabs above to explore different visualizations.")
    st.sidebar.caption("Export your results using the Export section below.")


# =============================================================================
# No Results Page
# =============================================================================


def render_no_results() -> None:
    """Render the page when no results are available."""
    st.title("Results")
    st.warning("No fitting results available yet.")

    st.markdown("""
    To view results:

    1. **Load Data** - Go to the Data Loading page and import your dataset
    2. **Select Model** - Choose or define a model on the Model Selection page
    3. **Configure & Fit** - Set parameters and run the fit on the Fitting Options page

    Once the fit completes successfully, return here to view detailed results.
    """)

    # Quick status
    state = get_session_state()

    st.markdown("### Current Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        if state.xdata is not None:
            st.success("Data loaded")
        else:
            st.error("No data loaded")

    with col2:
        if get_current_model() is not None:
            st.success("Model selected")
        else:
            st.error("No model selected")

    with col3:
        if state.fit_result is not None:
            st.success("Fit complete")
        else:
            st.error("No fit results")


# =============================================================================
# Main Page
# =============================================================================


def main() -> None:
    """Main entry point for the Results page."""
    apply_dark_theme_css()

    st.title("Fitting Results")

    # Check if we have results to display
    if not has_valid_result():
        render_no_results()
        return

    st.caption("Detailed analysis of curve fitting results")

    # Render sidebar
    render_sidebar()

    # Main content sections
    render_parameters_section()

    st.divider()

    render_statistics_section()

    st.divider()

    render_visualizations_section()

    st.divider()

    render_export_section()


if __name__ == "__main__":
    main()
