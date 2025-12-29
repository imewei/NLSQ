"""Advanced options component for NLSQ GUI.

This module provides the advanced mode options for the Fitting Options page,
organized into tabs for Fitting, Multi-start, Streaming, HPC, and Batch options.

Functions
---------
render_advanced_options
    Render the full advanced options UI with tabs.
render_fitting_tab
    Render the Fitting options tab.
render_multistart_tab
    Render the Multi-start options tab.
render_streaming_tab
    Render the Streaming options tab.
render_hpc_tab
    Render the HPC options tab.
render_batch_tab
    Render the Batch processing options tab.
"""

from typing import Any

import streamlit as st

from nlsq.gui.presets import (
    get_streaming_preset,
    get_streaming_preset_names,
)
from nlsq.gui.state import SessionState

# Available options for dropdowns
FITTING_METHODS = ["trf", "lm", "dogbox"]
LOSS_FUNCTIONS = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
SAMPLERS = ["lhs", "sobol", "halton"]
SUMMARY_FORMATS = ["json", "csv", "txt"]


def get_fitting_tab_config(state: SessionState) -> dict[str, Any]:
    """Get configuration values for the Fitting tab.

    Parameters
    ----------
    state : SessionState
        The current session state.

    Returns
    -------
    dict[str, Any]
        Dictionary with fitting configuration values.
    """
    return {
        "method": state.method,
        "gtol": state.gtol,
        "ftol": state.ftol,
        "xtol": state.xtol,
        "max_iterations": state.max_iterations,
        "max_function_evals": state.max_function_evals,
        "loss": state.loss,
    }


def get_multistart_tab_config(state: SessionState) -> dict[str, Any]:
    """Get configuration values for the Multi-start tab.

    Parameters
    ----------
    state : SessionState
        The current session state.

    Returns
    -------
    dict[str, Any]
        Dictionary with multi-start configuration values.
    """
    return {
        "enabled": state.enable_multistart,
        "n_starts": state.n_starts,
        "sampler": state.sampler,
        "center_on_p0": state.center_on_p0,
        "scale_factor": state.scale_factor,
    }


def get_streaming_tab_config(state: SessionState) -> dict[str, Any]:
    """Get configuration values for the Streaming tab.

    Parameters
    ----------
    state : SessionState
        The current session state.

    Returns
    -------
    dict[str, Any]
        Dictionary with streaming configuration values.
    """
    return {
        "chunk_size": state.chunk_size,
        "normalize": state.normalize,
        "warmup_iterations": state.warmup_iterations,
        "max_warmup_iterations": state.max_warmup_iterations,
        "defense_preset": state.defense_preset,
    }


def get_hpc_tab_config(state: SessionState) -> dict[str, Any]:
    """Get configuration values for the HPC tab.

    Parameters
    ----------
    state : SessionState
        The current session state.

    Returns
    -------
    dict[str, Any]
        Dictionary with HPC configuration values.
    """
    return {
        "enable_multi_device": state.enable_multi_device,
        "enable_checkpoints": state.enable_checkpoints,
        "checkpoint_dir": state.checkpoint_dir,
    }


def get_batch_tab_config(state: SessionState) -> dict[str, Any]:
    """Get configuration values for the Batch tab.

    Parameters
    ----------
    state : SessionState
        The current session state.

    Returns
    -------
    dict[str, Any]
        Dictionary with batch configuration values.
    """
    return {
        "max_workers": state.batch_max_workers,
        "continue_on_error": state.batch_continue_on_error,
        "summary_format": state.batch_summary_format,
    }


def validate_max_iterations(value: int) -> bool:
    """Validate max_iterations value.

    Parameters
    ----------
    value : int
        The value to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    return value > 0


def validate_chunk_size(value: int) -> bool:
    """Validate chunk_size value.

    Parameters
    ----------
    value : int
        The value to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    return value > 0


def validate_n_starts(value: int) -> bool:
    """Validate n_starts value.

    Parameters
    ----------
    value : int
        The value to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    return value > 0


def render_advanced_options(state: SessionState) -> None:
    """Render the full advanced options UI with tabs.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.subheader("Advanced Options")

    tabs = st.tabs(
        [
            "Fitting",
            "Multi-start",
            "Streaming",
            "HPC",
            "Batch",
        ]
    )

    with tabs[0]:
        render_fitting_tab(state)

    with tabs[1]:
        render_multistart_tab(state)

    with tabs[2]:
        render_streaming_tab(state)

    with tabs[3]:
        render_hpc_tab(state)

    with tabs[4]:
        render_batch_tab(state)


def render_fitting_tab(state: SessionState) -> None:
    """Render the Fitting options tab.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.markdown("#### Optimization Settings")

    col1, col2 = st.columns(2)

    with col1:
        # Method dropdown
        method = st.selectbox(
            "Optimization Method",
            options=FITTING_METHODS,
            index=FITTING_METHODS.index(state.method)
            if state.method in FITTING_METHODS
            else 0,
            help=(
                "trf: Trust Region Reflective (supports bounds)\n"
                "lm: Levenberg-Marquardt (no bounds, faster)\n"
                "dogbox: Dogleg (supports bounds, simpler)"
            ),
        )
        state.method = method

        # Loss function
        loss = st.selectbox(
            "Loss Function",
            options=LOSS_FUNCTIONS,
            index=LOSS_FUNCTIONS.index(state.loss)
            if state.loss in LOSS_FUNCTIONS
            else 0,
            help=(
                "linear: Standard least squares\n"
                "soft_l1, huber, cauchy, arctan: Robust to outliers"
            ),
        )
        state.loss = loss

    with col2:
        # Max iterations
        max_iter = st.number_input(
            "Max Iterations",
            min_value=1,
            max_value=100000,
            value=state.max_iterations,
            step=100,
            help="Maximum number of iterations for optimization",
        )
        state.max_iterations = int(max_iter)

        # Max function evaluations
        max_fev = st.number_input(
            "Max Function Evaluations",
            min_value=1,
            max_value=1000000,
            value=state.max_function_evals,
            step=100,
            help="Maximum number of function evaluations",
        )
        state.max_function_evals = int(max_fev)

    st.markdown("#### Tolerances")

    col1, col2, col3 = st.columns(3)

    with col1:
        gtol_exp = st.slider(
            "gtol (10^x)",
            min_value=-15,
            max_value=-1,
            value=round(import_math_log10(state.gtol)) if state.gtol > 0 else -8,
            help="Gradient tolerance for convergence",
        )
        state.gtol = 10**gtol_exp
        st.caption(f"gtol = {state.gtol:.0e}")

    with col2:
        ftol_exp = st.slider(
            "ftol (10^x)",
            min_value=-15,
            max_value=-1,
            value=round(import_math_log10(state.ftol)) if state.ftol > 0 else -8,
            help="Function tolerance for convergence",
        )
        state.ftol = 10**ftol_exp
        st.caption(f"ftol = {state.ftol:.0e}")

    with col3:
        xtol_exp = st.slider(
            "xtol (10^x)",
            min_value=-15,
            max_value=-1,
            value=round(import_math_log10(state.xtol)) if state.xtol > 0 else -8,
            help="Parameter tolerance for convergence",
        )
        state.xtol = 10**xtol_exp
        st.caption(f"xtol = {state.xtol:.0e}")


def import_math_log10(value: float) -> float:
    """Import math.log10 to avoid issues with streamlit reruns.

    Parameters
    ----------
    value : float
        Value to compute log10 of.

    Returns
    -------
    float
        log10 of value.
    """
    import math

    return math.log10(value) if value > 0 else -8


def render_multistart_tab(state: SessionState) -> None:
    """Render the Multi-start options tab.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.markdown("#### Multi-start Optimization")

    # Enable toggle
    enable = st.toggle(
        "Enable Multi-start",
        value=state.enable_multistart,
        help="Run optimization from multiple starting points to find global minimum",
    )
    state.enable_multistart = enable

    if enable:
        col1, col2 = st.columns(2)

        with col1:
            # Number of starts
            n_starts = st.slider(
                "Number of Starts",
                min_value=2,
                max_value=100,
                value=state.n_starts,
                help="Number of starting points to sample",
            )
            state.n_starts = n_starts

            # Sampler
            sampler = st.selectbox(
                "Sampler",
                options=SAMPLERS,
                index=SAMPLERS.index(state.sampler) if state.sampler in SAMPLERS else 0,
                help=(
                    "lhs: Latin Hypercube Sampling\n"
                    "sobol: Sobol quasi-random sequence\n"
                    "halton: Halton quasi-random sequence"
                ),
            )
            state.sampler = sampler

        with col2:
            # Center on p0
            center = st.checkbox(
                "Center on p0",
                value=state.center_on_p0,
                help="Center the sampling distribution on initial guess",
            )
            state.center_on_p0 = center

            # Scale factor
            scale = st.slider(
                "Scale Factor",
                min_value=0.1,
                max_value=10.0,
                value=state.scale_factor,
                step=0.1,
                help="Scale factor for sampling region around p0",
            )
            state.scale_factor = scale

        st.info(
            f"Multi-start will run {n_starts} optimizations using {sampler} sampling"
        )
    else:
        st.caption(
            "Enable multi-start to run optimization from multiple starting points"
        )


def render_streaming_tab(state: SessionState) -> None:
    """Render the Streaming options tab.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.markdown("#### Streaming Optimization")
    st.caption("For large datasets that don't fit in memory")

    # Streaming preset selector
    st.markdown("##### Quick Presets")
    preset_names = get_streaming_preset_names()
    preset_options = ["Custom", *preset_names]

    selected_preset = st.selectbox(
        "Streaming Preset",
        options=preset_options,
        index=0,
        help="Select a preset or customize settings",
    )

    if selected_preset != "Custom":
        preset = get_streaming_preset(selected_preset)
        state.chunk_size = preset["chunk_size"]
        state.normalize = preset["normalize"]
        state.warmup_iterations = preset["warmup_iterations"]
        if "max_warmup_iterations" in preset:
            state.max_warmup_iterations = preset["max_warmup_iterations"]

    st.markdown("##### Settings")

    col1, col2 = st.columns(2)

    with col1:
        # Chunk size
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=1000000,
            value=state.chunk_size,
            step=1000,
            help="Number of data points per chunk",
        )
        state.chunk_size = int(chunk_size)

        # Warmup iterations
        warmup = st.number_input(
            "Warmup Iterations",
            min_value=10,
            max_value=10000,
            value=state.warmup_iterations,
            step=50,
            help="Iterations on first chunk for initial convergence",
        )
        state.warmup_iterations = int(warmup)

    with col2:
        # Normalize toggle
        normalize = st.checkbox(
            "Normalize Parameters",
            value=state.normalize,
            help="Normalize parameters for numerical stability",
        )
        state.normalize = normalize

        # Max warmup iterations
        max_warmup = st.number_input(
            "Max Warmup Iterations",
            min_value=100,
            max_value=10000,
            value=state.max_warmup_iterations,
            step=100,
            help="Maximum iterations during warmup phase",
        )
        state.max_warmup_iterations = int(max_warmup)

    # Defense layers section
    st.markdown("##### Defense Layers")

    with st.expander("Configure Defense Layers", expanded=False):
        # Layer 1: Warm start
        st.markdown("**Layer 1: Warm Start**")
        l1_enabled = st.checkbox(
            "Enable warm start",
            value=state.layer1_enabled,
            key="layer1_enable",
            help="Initialize from previous chunk results",
        )
        state.layer1_enabled = l1_enabled

        if l1_enabled:
            l1_threshold = st.slider(
                "Threshold",
                min_value=0.001,
                max_value=0.1,
                value=state.layer1_threshold,
                step=0.001,
                key="layer1_threshold",
                help="Threshold for warm start activation",
            )
            state.layer1_threshold = l1_threshold

        # Layer 2: Adaptive learning rate
        st.markdown("**Layer 2: Adaptive Learning Rate**")
        l2_enabled = st.checkbox(
            "Enable adaptive learning rate",
            value=state.layer2_enabled,
            key="layer2_enable",
            help="Adjust step size based on convergence",
        )
        state.layer2_enabled = l2_enabled

        # Layer 3: Cost guard
        st.markdown("**Layer 3: Cost Guard**")
        l3_enabled = st.checkbox(
            "Enable cost guard",
            value=state.layer3_enabled,
            key="layer3_enable",
            help="Protect against cost increases",
        )
        state.layer3_enabled = l3_enabled

        if l3_enabled:
            l3_tolerance = st.slider(
                "Tolerance",
                min_value=0.01,
                max_value=0.5,
                value=state.layer3_tolerance,
                step=0.01,
                key="layer3_tolerance",
                help="Allowed cost increase tolerance",
            )
            state.layer3_tolerance = l3_tolerance

        # Layer 4: Step clipping
        st.markdown("**Layer 4: Step Clipping**")
        l4_enabled = st.checkbox(
            "Enable step clipping",
            value=state.layer4_enabled,
            key="layer4_enable",
            help="Clip large parameter steps",
        )
        state.layer4_enabled = l4_enabled

        if l4_enabled:
            l4_max_step = st.slider(
                "Max Step Size",
                min_value=0.01,
                max_value=1.0,
                value=state.layer4_max_step,
                step=0.01,
                key="layer4_max_step",
                help="Maximum allowed parameter step",
            )
            state.layer4_max_step = l4_max_step


def render_hpc_tab(state: SessionState) -> None:
    """Render the HPC options tab.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.markdown("#### High-Performance Computing")

    # Multi-device
    st.markdown("##### Multi-Device")
    multi_device = st.checkbox(
        "Enable Multi-Device (Multi-GPU)",
        value=state.enable_multi_device,
        help="Distribute computation across multiple GPUs/TPUs",
    )
    state.enable_multi_device = multi_device

    if multi_device:
        st.info("Multi-device mode will automatically detect available devices")

    st.divider()

    # Checkpointing
    st.markdown("##### Checkpointing")
    enable_checkpoints = st.checkbox(
        "Enable Checkpointing",
        value=state.enable_checkpoints,
        help="Save intermediate results for resumable fitting",
    )
    state.enable_checkpoints = enable_checkpoints

    if enable_checkpoints:
        checkpoint_dir = st.text_input(
            "Checkpoint Directory",
            value=state.checkpoint_dir or "",
            placeholder="/path/to/checkpoints",
            help="Directory to save checkpoint files",
        )
        state.checkpoint_dir = checkpoint_dir if checkpoint_dir else None

        st.caption("Checkpoints enable resuming interrupted fits")


def render_batch_tab(state: SessionState) -> None:
    """Render the Batch processing options tab.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.markdown("#### Batch Processing")
    st.caption("Settings for processing multiple datasets")

    col1, col2 = st.columns(2)

    with col1:
        # Max workers
        use_auto_workers = st.checkbox(
            "Auto Workers",
            value=state.batch_max_workers is None,
            help="Automatically determine number of workers",
        )

        if use_auto_workers:
            state.batch_max_workers = None
            st.caption("Workers: Auto (based on CPU count)")
        else:
            workers = st.number_input(
                "Max Workers",
                min_value=1,
                max_value=64,
                value=state.batch_max_workers or 4,
                step=1,
                help="Maximum parallel workers for batch processing",
            )
            state.batch_max_workers = int(workers)

    with col2:
        # Continue on error
        continue_on_error = st.checkbox(
            "Continue on Error",
            value=state.batch_continue_on_error,
            help="Continue processing remaining files if one fails",
        )
        state.batch_continue_on_error = continue_on_error

        # Summary format
        summary_format = st.selectbox(
            "Summary Format",
            options=SUMMARY_FORMATS,
            index=SUMMARY_FORMATS.index(state.batch_summary_format)
            if state.batch_summary_format in SUMMARY_FORMATS
            else 0,
            help="Output format for batch summary",
        )
        state.batch_summary_format = summary_format
