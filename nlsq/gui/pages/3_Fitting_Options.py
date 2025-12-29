"""Fitting Options Page for NLSQ GUI.

This page provides the interface for configuring curve fitting parameters.
It includes two modes: Guided (with presets) and Advanced (full control).

Features:
- Parameter configuration (p0, bounds, transforms)
- Guided mode with Fast/Robust/Quality presets
- Advanced mode with tabbed options
- YAML configuration import
- Run Fit button with progress feedback
- Real-time cost plot and iteration table
"""

import logging
from io import StringIO

import numpy as np
import streamlit as st

# Configure logging for GUI fitting operations
logger = logging.getLogger("nlsq.gui.fitting")

from nlsq.gui.adapters.config_adapter import (
    load_yaml_config,
    validate_yaml_config,
)
from nlsq.gui.adapters.fit_adapter import (
    create_fit_config_from_state,
    execute_fit,
    extract_confidence_intervals,
    extract_convergence_info,
    extract_fit_statistics,
    is_large_dataset,
)
from nlsq.gui.components.advanced_options import render_advanced_options
from nlsq.gui.components.iteration_table import (
    create_iteration_history,
    render_iteration_table,
    update_iteration_history,
)
from nlsq.gui.components.live_cost_plot import (
    create_cost_history,
    render_live_cost_plot,
    update_cost_history,
)
from nlsq.gui.components.param_config import (
    get_param_names_from_model,
    render_param_config,
)
from nlsq.gui.presets import (
    get_preset_description,
)
from nlsq.gui.state import SessionState, apply_preset_to_state, initialize_state
from nlsq.gui.utils.theme import apply_dark_theme_css

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Fitting Options - NLSQ",
    page_icon="sliders",
    layout="wide",
)


# =============================================================================
# Session State Initialization
# =============================================================================


def get_session_state() -> SessionState:
    """Get or initialize the session state."""
    if "nlsq_state" not in st.session_state:
        st.session_state.nlsq_state = initialize_state()
    return st.session_state.nlsq_state


def init_fitting_page_state() -> None:
    """Initialize fitting page specific state."""
    if "yaml_import_success" not in st.session_state:
        st.session_state.yaml_import_success = None
    if "yaml_import_message" not in st.session_state:
        st.session_state.yaml_import_message = ""
    if "cost_history" not in st.session_state:
        st.session_state.cost_history = create_cost_history()
    if "iteration_history" not in st.session_state:
        st.session_state.iteration_history = create_iteration_history()


# =============================================================================
# Mode Toggle
# =============================================================================


def render_mode_toggle() -> str:
    """Render the mode toggle between Guided and Advanced.

    Returns
    -------
    str
        Selected mode: "guided" or "advanced".
    """
    state = get_session_state()

    mode = st.radio(
        "Configuration Mode",
        options=["Guided", "Advanced"],
        index=0 if state.mode == "guided" else 1,
        horizontal=True,
        help=(
            "Guided: Use presets for common scenarios\n"
            "Advanced: Full control over all parameters"
        ),
    )

    state.mode = "guided" if mode == "Guided" else "advanced"
    return state.mode


# =============================================================================
# Guided Mode UI
# =============================================================================


def render_guided_mode(state: SessionState) -> None:
    """Render the guided mode UI with presets.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.markdown("### Preset Selection")
    st.caption("Choose a preset that matches your fitting needs")

    # Preset radio buttons
    preset_names = ["fast", "robust", "quality"]
    preset_labels = {
        "fast": "Fast - Quick results with lower precision",
        "robust": "Robust - Balanced precision with multi-start",
        "quality": "Quality - Highest precision with extensive multi-start",
    }

    # Find current preset index
    current_preset = state.preset.lower() if state.preset else "standard"
    if current_preset not in preset_names:
        current_preset = "robust"  # Default to robust if standard/unknown

    selected = st.radio(
        "Preset",
        options=preset_names,
        format_func=lambda x: preset_labels.get(x, x.capitalize()),
        index=preset_names.index(current_preset)
        if current_preset in preset_names
        else 1,
        key="preset_selector",
    )

    # Apply selected preset
    if selected != state.preset:
        apply_preset_to_state(state, selected)
        st.rerun()

    st.divider()

    # Display current settings
    render_preset_details(state)

    st.divider()

    # YAML import section
    render_yaml_import()


def render_preset_details(state: SessionState) -> None:
    """Render the current preset details.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.markdown("### Current Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Tolerances**")
        st.code(f"gtol: {state.gtol:.0e}")
        st.code(f"ftol: {state.ftol:.0e}")
        st.code(f"xtol: {state.xtol:.0e}")

    with col2:
        st.markdown("**Optimization**")
        st.code(f"Method: {state.method}")
        st.code(f"Max Iterations: {state.max_iterations}")
        st.code(f"Loss: {state.loss}")

    with col3:
        st.markdown("**Multi-start**")
        if state.enable_multistart:
            st.success(f"Enabled ({state.n_starts} starts)")
            st.caption(f"Sampler: {state.sampler}")
        else:
            st.info("Disabled")

    # Preset description
    if state.preset:
        desc = get_preset_description(state.preset)
        if desc:
            st.caption(f"_{desc}_")


# =============================================================================
# YAML Import
# =============================================================================


def render_yaml_import() -> None:
    """Render the YAML configuration import section."""
    st.markdown("### Import Configuration")

    uploaded_file = st.file_uploader(
        "Load YAML Config",
        type=["yaml", "yml"],
        key="yaml_config_upload",
        help="Import a CLI workflow configuration file",
    )

    if uploaded_file is not None:
        try:
            # Read content
            content = uploaded_file.read().decode("utf-8")

            # Validate
            is_valid, error = validate_yaml_config(content)

            if not is_valid:
                st.error(f"Invalid YAML: {error}")
                return

            # Parse and apply
            imported_state = load_yaml_config(StringIO(content))

            # Copy relevant fields to current state
            state = get_session_state()
            apply_imported_config(state, imported_state)

            st.success(f"Configuration imported from {uploaded_file.name}")
            st.rerun()

        except Exception as e:
            st.error(f"Error importing configuration: {e}")


def apply_imported_config(state: SessionState, imported: SessionState) -> None:
    """Apply imported configuration to current state.

    Parameters
    ----------
    state : SessionState
        Current session state to update.
    imported : SessionState
        Imported configuration.
    """
    # Model settings
    state.model_type = imported.model_type
    state.model_name = imported.model_name
    state.auto_p0 = imported.auto_p0
    state.auto_bounds = imported.auto_bounds

    # Fitting settings
    state.method = imported.method
    state.gtol = imported.gtol
    state.ftol = imported.ftol
    state.xtol = imported.xtol
    state.max_iterations = imported.max_iterations
    state.max_function_evals = imported.max_function_evals
    state.loss = imported.loss

    # Multi-start settings
    state.enable_multistart = imported.enable_multistart
    state.n_starts = imported.n_starts
    state.sampler = imported.sampler
    state.center_on_p0 = imported.center_on_p0
    state.scale_factor = imported.scale_factor

    # Streaming settings
    state.chunk_size = imported.chunk_size
    state.normalize = imported.normalize
    state.warmup_iterations = imported.warmup_iterations
    state.max_warmup_iterations = imported.max_warmup_iterations

    # HPC settings
    state.enable_multi_device = imported.enable_multi_device
    state.enable_checkpoints = imported.enable_checkpoints
    state.checkpoint_dir = imported.checkpoint_dir

    # Defense layers
    state.layer1_enabled = imported.layer1_enabled
    state.layer1_threshold = imported.layer1_threshold
    state.layer2_enabled = imported.layer2_enabled
    state.layer3_enabled = imported.layer3_enabled
    state.layer3_tolerance = imported.layer3_tolerance
    state.layer4_enabled = imported.layer4_enabled
    state.layer4_max_step = imported.layer4_max_step

    # Batch settings
    state.batch_max_workers = imported.batch_max_workers
    state.batch_continue_on_error = imported.batch_continue_on_error
    state.batch_summary_format = imported.batch_summary_format

    # Parameters if set
    if imported.p0 is not None:
        state.p0 = imported.p0
    if imported.bounds is not None:
        state.bounds = imported.bounds
    if imported.transforms:
        state.transforms = dict(imported.transforms)

    # Switch to advanced mode to show all imported settings
    state.mode = "advanced"


# =============================================================================
# Parameter Configuration
# =============================================================================


def render_parameter_section(state: SessionState) -> None:
    """Render the parameter configuration section.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    # Check if model is loaded
    if (
        "current_model" not in st.session_state
        or st.session_state.current_model is None
    ):
        st.warning(
            "No model selected. Please select a model on the Model Selection page first."
        )
        return

    model = st.session_state.current_model

    # Check if data is loaded (for auto_p0)
    xdata = state.xdata
    ydata = state.ydata

    if xdata is None or ydata is None:
        st.info(
            "Data not loaded. Auto p0 estimation will be available after loading data."
        )

    # Render parameter configuration
    render_param_config(state, model, xdata, ydata)


# =============================================================================
# Fitting Execution
# =============================================================================


class GUIProgressCallback:
    """Progress callback for GUI updates during fitting."""

    def __init__(self, state: SessionState):
        self.state = state
        self.iteration = 0
        self.cost_history = st.session_state.get("cost_history", create_cost_history())
        self.iteration_history = st.session_state.get(
            "iteration_history", create_iteration_history()
        )

    def on_iteration(self, iteration: int, cost: float, params: np.ndarray) -> None:
        """Handle iteration update."""
        self.iteration = iteration
        self.cost_history = update_cost_history(self.cost_history, iteration, cost)
        self.iteration_history = update_iteration_history(
            self.iteration_history, iteration, params, cost
        )
        # Store back to session state
        st.session_state.cost_history = self.cost_history
        st.session_state.iteration_history = self.iteration_history

    def should_abort(self) -> bool:
        """Check if fitting should be aborted."""
        return self.state.fit_aborted


def is_ready_to_fit(state: SessionState) -> tuple[bool, str]:
    """Check if all prerequisites for fitting are met.

    Returns
    -------
    tuple[bool, str]
        (is_ready, message)
    """
    if state.xdata is None or state.ydata is None:
        return False, "Data not loaded. Go to Data Loading page."

    if (
        "current_model" not in st.session_state
        or st.session_state.current_model is None
    ):
        return False, "Model not selected. Go to Model Selection page."

    # Check p0 - either auto_p0 must be enabled with a model that supports it,
    # or p0 must have at least one non-None value
    model = st.session_state.current_model
    has_auto_p0 = hasattr(model, "estimate_p0") and callable(
        getattr(model, "estimate_p0", None)
    )

    if state.p0 is None:
        if not (state.auto_p0 and has_auto_p0):
            return False, "Initial parameters not set. Configure p0 above."
    else:
        # Check if all p0 values are None
        all_none = all(v is None for v in state.p0)
        if all_none and not (state.auto_p0 and has_auto_p0):
            return False, "Initial parameters not set. Configure p0 above."

    if state.fit_running:
        return False, "Fit is already running."

    return True, "Ready to fit"


def run_fit(state: SessionState) -> str | None:
    """Execute the fitting procedure.

    Returns
    -------
    str | None
        Error message if fitting failed, None if successful.
    """
    state.fit_running = True
    state.fit_aborted = False
    state.fit_result = None
    error_message: str | None = None

    # Clear previous history
    st.session_state.cost_history = create_cost_history()
    st.session_state.iteration_history = create_iteration_history()

    model = st.session_state.current_model

    # Apply auto_p0 if enabled and p0 contains None values
    has_auto_p0 = hasattr(model, "estimate_p0") and callable(
        getattr(model, "estimate_p0", None)
    )

    if state.auto_p0 and has_auto_p0:
        try:
            estimated_p0 = model.estimate_p0(state.xdata, state.ydata)
            if estimated_p0 is not None:
                estimated_p0 = list(estimated_p0)
                # Fill in None values with estimated values
                if state.p0 is None:
                    state.p0 = estimated_p0
                    logger.info(
                        "Auto p0 applied (full replacement) | estimated=%s",
                        estimated_p0,
                    )
                else:
                    filled_indices = []
                    for i in range(len(state.p0)):
                        if state.p0[i] is None and i < len(estimated_p0):
                            state.p0[i] = estimated_p0[i]
                            filled_indices.append(i)
                    if filled_indices:
                        logger.info(
                            "Auto p0 applied (partial) | filled_indices=%s, final_p0=%s",
                            filled_indices,
                            state.p0,
                        )
                    else:
                        logger.info(
                            "Auto p0 skipped (all values set by user) | user_p0=%s",
                            state.p0,
                        )
        except Exception as e:
            logger.warning("Auto p0 estimation failed | error=%s", e)
            error_message = f"Auto p0 estimation failed: {e}"
            state.fit_running = False
            return error_message
    elif state.p0 is not None:
        logger.info("Manual p0 used | p0=%s", state.p0)
    else:
        logger.warning("No p0 available and auto_p0 disabled")

    # Ensure p0 has no None values
    if state.p0 is None or any(v is None for v in state.p0):
        error_message = "p0 contains undefined values. Please set initial parameters."
        state.fit_running = False
        return error_message

    config = create_fit_config_from_state(state)

    # Create callback
    callback = GUIProgressCallback(state)

    try:
        result = execute_fit(
            xdata=state.xdata,
            ydata=state.ydata,
            sigma=state.sigma,
            model=model,
            config=config,
            progress_callback=callback,
        )
        state.fit_result = result
    except Exception as e:
        error_message = f"Fitting failed: {e}"
        state.fit_result = None
    finally:
        state.fit_running = False

    return error_message


def render_fit_execution_section(state: SessionState) -> None:
    """Render the fit execution section with Run Fit button and progress.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.markdown("### Run Fitting")

    # Check readiness
    is_ready, message = is_ready_to_fit(state)

    # Initialize error state in session
    if "fit_error" not in st.session_state:
        st.session_state.fit_error = None

    # Button row
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        if st.button(
            "Run Fit",
            disabled=not is_ready or state.fit_running,
            type="primary",
            key="run_fit_button",
        ):
            # Clear previous error
            st.session_state.fit_error = None
            error = run_fit(state)
            if error:
                st.session_state.fit_error = error
            st.rerun()

    with col2:
        if state.fit_running:
            if st.button(
                "Abort",
                type="secondary",
                key="abort_button",
            ):
                state.fit_aborted = True
                st.warning("Abort requested...")

    with col3:
        if not is_ready:
            st.info(message)
        elif state.fit_running:
            st.info("Fitting in progress...")
        elif st.session_state.fit_error:
            st.error(st.session_state.fit_error)
        elif state.fit_result is not None:
            if state.fit_result.success:
                st.success("Fit completed successfully")
            else:
                st.warning("Fit completed but may not have converged")

    # Progress display during fitting
    if state.fit_running:
        render_fit_progress(state)

    # Show results summary if fit completed
    if state.fit_result is not None and not state.fit_running:
        render_fit_summary(state)


def render_fit_progress(state: SessionState) -> None:
    """Render progress indicators during fitting.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    # Progress bar
    iteration = len(st.session_state.get("cost_history", {}).get("iterations", []))
    max_iter = state.max_iterations

    progress = min(1.0, iteration / max_iter) if max_iter > 0 else 0.0
    st.progress(progress, text=f"Iteration {iteration}/{max_iter}")

    # Live plots in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Cost vs Iteration**")
        cost_history = st.session_state.get("cost_history", create_cost_history())
        render_live_cost_plot(cost_history, key="fitting_cost_plot")

    with col2:
        st.markdown("**Parameter Values**")
        iteration_history = st.session_state.get(
            "iteration_history", create_iteration_history()
        )
        model = st.session_state.get("current_model")
        param_names = get_param_names_from_model(model) if model else None
        render_iteration_table(
            iteration_history, param_names=param_names, key="fitting_params_table"
        )

    # Streaming/checkpoint indicator for large datasets
    if state.xdata is not None and is_large_dataset(state.xdata):
        st.caption("Large dataset mode: streaming optimization active")


def render_fit_summary(state: SessionState) -> None:
    """Render a summary of the fit results.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    result = state.fit_result
    if result is None:
        return

    st.divider()
    st.markdown("### Fit Summary")

    # Extract statistics
    stats = extract_fit_statistics(result)
    info = extract_convergence_info(result)

    # Display in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R-squared", f"{stats['r_squared']:.6f}")
        st.metric("RMSE", f"{stats['rmse']:.6g}")

    with col2:
        st.metric("AIC", f"{stats['aic']:.2f}")
        st.metric("BIC", f"{stats['bic']:.2f}")

    with col3:
        st.metric("Function Evaluations", info["nfev"])
        st.metric("Final Cost", f"{info['cost']:.6e}")

    # Fitted parameters
    st.markdown("**Fitted Parameters**")
    model = st.session_state.get("current_model")
    param_names = get_param_names_from_model(model) if model else None

    if param_names and hasattr(result, "popt"):
        ci = extract_confidence_intervals(result, alpha=0.95)
        cols = st.columns(len(result.popt))
        for i, (col, popt) in enumerate(zip(cols, result.popt, strict=False)):
            name = param_names[i] if i < len(param_names) else f"p{i}"
            with col:
                st.metric(
                    label=name,
                    value=f"{popt:.6g}",
                    help=f"95% CI: [{ci[i][0]:.4g}, {ci[i][1]:.4g}]",
                )

    # Link to results page
    st.info("View detailed results on the Results page")


# =============================================================================
# Sidebar Status
# =============================================================================


def render_sidebar_status(state: SessionState) -> None:
    """Render the sidebar status panel.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.sidebar.subheader("Fitting Status")

    # Mode indicator
    st.sidebar.markdown(f"**Mode:** {state.mode.capitalize()}")

    # Preset indicator (for guided mode)
    if state.mode == "guided" and state.preset:
        st.sidebar.markdown(f"**Preset:** {state.preset.capitalize()}")

    # Tolerance summary
    st.sidebar.markdown("**Tolerances:**")
    st.sidebar.code(f"gtol={state.gtol:.0e}")

    # Multi-start status
    if state.enable_multistart:
        st.sidebar.success(f"Multi-start: {state.n_starts} starts")
    else:
        st.sidebar.info("Multi-start: Off")

    # Model status
    if (
        "current_model" in st.session_state
        and st.session_state.current_model is not None
    ):
        model = st.session_state.current_model
        param_names = get_param_names_from_model(model)
        st.sidebar.markdown(f"**Parameters:** {len(param_names)}")
    else:
        st.sidebar.warning("No model selected")

    # Data status
    if state.xdata is not None and state.ydata is not None:
        try:
            n_points = len(state.xdata)
            st.sidebar.markdown(f"**Data Points:** {n_points:,}")
        except (TypeError, AttributeError):
            st.sidebar.markdown("**Data:** Loaded")
    else:
        st.sidebar.warning("No data loaded")

    # Fit status
    st.sidebar.divider()
    if state.fit_running:
        st.sidebar.warning("Fit running...")
    elif state.fit_result is not None:
        if state.fit_result.success:
            st.sidebar.success("Last fit: Success")
        else:
            st.sidebar.warning("Last fit: Did not converge")


# =============================================================================
# Main Page
# =============================================================================


def main() -> None:
    """Main entry point for the Fitting Options page."""
    apply_dark_theme_css()
    init_fitting_page_state()

    st.title("Fitting Options")
    st.caption("Configure curve fitting parameters and optimization settings")

    state = get_session_state()

    # Mode toggle
    mode = render_mode_toggle()

    st.divider()

    # Parameter configuration section (always shown)
    with st.expander("Parameter Configuration", expanded=True):
        render_parameter_section(state)

    st.divider()

    # Mode-specific content
    if mode == "guided":
        render_guided_mode(state)
    else:
        render_advanced_options(state)

    st.divider()

    # Fit execution section
    render_fit_execution_section(state)

    # Sidebar status
    render_sidebar_status(state)


if __name__ == "__main__":
    main()
