"""Session state management for the NLSQ GUI.

This module provides the SessionState dataclass and functions for managing
application state across Streamlit reruns. It handles state initialization,
reset, and export to workflow configuration dictionaries.

The SessionState dataclass contains all fields needed to configure a curve
fitting workflow, matching the fields from workflow_config_template.yaml.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SessionState:
    """Container for all GUI session state fields.

    This dataclass holds all configuration fields for a curve fitting workflow,
    organized into logical groups: data, model, fitting parameters, multi-start,
    streaming, and export settings.

    Attributes
    ----------
    xdata : array-like or None
        Independent variable data.
    ydata : array-like or None
        Dependent variable data.
    zdata : array-like or None
        Z-axis data for 2D surface fitting.
    sigma : array-like or None
        Uncertainties in ydata.
    data_mode : str
        Data dimensionality mode ("1d" or "2d").
    model_type : str
        Model source type ("builtin", "polynomial", "custom").
    model_name : str
        Name of the selected built-in model.
    polynomial_degree : int
        Degree for polynomial models.
    custom_code : str
        Python code for custom model functions.
    custom_file_path : str or None
        Path to external custom model file.
    custom_function_name : str
        Name of function to use from custom file.
    p0 : list or None
        Initial parameter guesses.
    bounds : tuple or None
        Parameter bounds as (lower, upper).
    transforms : dict
        Parameter transforms by name.
    auto_p0 : bool
        Whether to auto-estimate initial parameters.
    auto_bounds : bool
        Whether to use model default bounds.
    gtol : float
        Gradient tolerance for convergence.
    ftol : float
        Function tolerance for convergence.
    xtol : float
        Parameter tolerance for convergence.
    max_iterations : int
        Maximum number of iterations.
    method : str
        Optimization method ("trf", "lm", "dogbox").
    loss : str
        Loss function type.
    enable_multistart : bool
        Whether to enable multi-start optimization.
    n_starts : int
        Number of starting points for multi-start.
    sampler : str
        Sampling method for multi-start ("lhs", "sobol", "halton").
    center_on_p0 : bool
        Whether to center multi-start samples on p0.
    scale_factor : float
        Scale factor for multi-start sampling region.
    chunk_size : int
        Chunk size for streaming optimization.
    normalize : bool
        Whether to normalize parameters in streaming.
    enable_checkpoints : bool
        Whether to enable checkpointing.
    checkpoint_dir : str or None
        Directory for checkpoint files.
    enable_multi_device : bool
        Whether to enable multi-device (multi-GPU) mode.
    defense_preset : str or None
        Defense layer preset for hybrid streaming.
    preset : str
        Active preset name ("standard", "fast", "robust", "quality").
    mode : str
        UI mode ("guided" or "advanced").
    fit_result : Any
        Result from the last fit operation.
    fit_running : bool
        Whether a fit is currently running.
    """

    # Data fields
    xdata: Any = None
    ydata: Any = None
    zdata: Any = None
    sigma: Any = None
    data_mode: str = "1d"
    data_file_name: str | None = None
    x_column: int | str | None = None
    y_column: int | str | None = None
    z_column: int | str | None = None
    sigma_column: int | str | None = None

    # Model fields
    model_type: str = "builtin"
    model_name: str = "exponential_decay"
    polynomial_degree: int = 3
    custom_code: str = ""
    custom_file_path: str | None = None
    custom_function_name: str = ""

    # Parameter configuration
    p0: list[float] | None = None
    bounds: tuple[list[float], list[float]] | None = None
    transforms: dict[str, str] = field(default_factory=dict)
    auto_p0: bool = True
    auto_bounds: bool = False

    # Fitting tolerances and method
    gtol: float = 1e-8
    ftol: float = 1e-8
    xtol: float = 1e-8
    max_iterations: int = 200
    max_function_evals: int = 2000
    method: str = "trf"
    loss: str = "linear"

    # Multi-start settings
    enable_multistart: bool = False
    n_starts: int = 10
    sampler: str = "lhs"
    center_on_p0: bool = True
    scale_factor: float = 1.0

    # Streaming settings
    chunk_size: int = 10000
    normalize: bool = True
    warmup_iterations: int = 200
    max_warmup_iterations: int = 500

    # Checkpointing
    enable_checkpoints: bool = False
    checkpoint_dir: str | None = None

    # HPC settings
    enable_multi_device: bool = False

    # Defense layer settings
    defense_preset: str | None = None
    layer1_enabled: bool = True
    layer1_threshold: float = 0.01
    layer2_enabled: bool = True
    layer3_enabled: bool = True
    layer3_tolerance: float = 0.05
    layer4_enabled: bool = True
    layer4_max_step: float = 0.1

    # Batch processing settings
    batch_max_workers: int | None = None
    batch_continue_on_error: bool = True
    batch_summary_format: str = "json"

    # UI state
    preset: str = "standard"
    mode: str = "guided"

    # Fit result state
    fit_result: Any = None
    fit_running: bool = False
    fit_aborted: bool = False

    def copy(self) -> "SessionState":
        """Create a shallow copy of this SessionState.

        Returns
        -------
        SessionState
            A new SessionState instance with the same values.
        """
        return SessionState(
            xdata=self.xdata,
            ydata=self.ydata,
            zdata=self.zdata,
            sigma=self.sigma,
            data_mode=self.data_mode,
            data_file_name=self.data_file_name,
            x_column=self.x_column,
            y_column=self.y_column,
            z_column=self.z_column,
            sigma_column=self.sigma_column,
            model_type=self.model_type,
            model_name=self.model_name,
            polynomial_degree=self.polynomial_degree,
            custom_code=self.custom_code,
            custom_file_path=self.custom_file_path,
            custom_function_name=self.custom_function_name,
            p0=self.p0,
            bounds=self.bounds,
            transforms=dict(self.transforms),
            auto_p0=self.auto_p0,
            auto_bounds=self.auto_bounds,
            gtol=self.gtol,
            ftol=self.ftol,
            xtol=self.xtol,
            max_iterations=self.max_iterations,
            max_function_evals=self.max_function_evals,
            method=self.method,
            loss=self.loss,
            enable_multistart=self.enable_multistart,
            n_starts=self.n_starts,
            sampler=self.sampler,
            center_on_p0=self.center_on_p0,
            scale_factor=self.scale_factor,
            chunk_size=self.chunk_size,
            normalize=self.normalize,
            warmup_iterations=self.warmup_iterations,
            max_warmup_iterations=self.max_warmup_iterations,
            enable_checkpoints=self.enable_checkpoints,
            checkpoint_dir=self.checkpoint_dir,
            enable_multi_device=self.enable_multi_device,
            defense_preset=self.defense_preset,
            layer1_enabled=self.layer1_enabled,
            layer1_threshold=self.layer1_threshold,
            layer2_enabled=self.layer2_enabled,
            layer3_enabled=self.layer3_enabled,
            layer3_tolerance=self.layer3_tolerance,
            layer4_enabled=self.layer4_enabled,
            layer4_max_step=self.layer4_max_step,
            batch_max_workers=self.batch_max_workers,
            batch_continue_on_error=self.batch_continue_on_error,
            batch_summary_format=self.batch_summary_format,
            preset=self.preset,
            mode=self.mode,
            fit_result=self.fit_result,
            fit_running=self.fit_running,
            fit_aborted=self.fit_aborted,
        )


def initialize_state(**kwargs: Any) -> SessionState:
    """Initialize a new SessionState with default values.

    Parameters
    ----------
    **kwargs
        Optional keyword arguments to override default values.

    Returns
    -------
    SessionState
        A new SessionState instance with defaults and any provided overrides.

    Examples
    --------
    >>> state = initialize_state()
    >>> state.model_name
    'exponential_decay'

    >>> state = initialize_state(model_name="gaussian", gtol=1e-10)
    >>> state.model_name
    'gaussian'
    >>> state.gtol
    1e-10
    """
    return SessionState(**kwargs)


def reset_state(state: SessionState, preserve_preferences: bool = False) -> None:
    """Reset session state to default values.

    Parameters
    ----------
    state : SessionState
        The session state to reset.
    preserve_preferences : bool, optional
        If True, preserve user preferences like mode and preset.
        Default is False.

    Examples
    --------
    >>> state = initialize_state()
    >>> state.xdata = [1, 2, 3]
    >>> state.mode = "advanced"
    >>> reset_state(state)
    >>> state.xdata is None
    True
    >>> state.mode
    'guided'

    >>> state.mode = "advanced"
    >>> reset_state(state, preserve_preferences=True)
    >>> state.mode
    'advanced'
    """
    # Save preferences if needed
    saved_mode = state.mode if preserve_preferences else "guided"
    saved_preset = state.preset if preserve_preferences else "standard"

    # Create a fresh state
    defaults = SessionState()

    # Reset data fields
    state.xdata = defaults.xdata
    state.ydata = defaults.ydata
    state.zdata = defaults.zdata
    state.sigma = defaults.sigma
    state.data_mode = defaults.data_mode
    state.data_file_name = defaults.data_file_name
    state.x_column = defaults.x_column
    state.y_column = defaults.y_column
    state.z_column = defaults.z_column
    state.sigma_column = defaults.sigma_column

    # Reset model fields
    state.model_type = defaults.model_type
    state.model_name = defaults.model_name
    state.polynomial_degree = defaults.polynomial_degree
    state.custom_code = defaults.custom_code
    state.custom_file_path = defaults.custom_file_path
    state.custom_function_name = defaults.custom_function_name

    # Reset parameter configuration
    state.p0 = defaults.p0
    state.bounds = defaults.bounds
    state.transforms = dict(defaults.transforms)
    state.auto_p0 = defaults.auto_p0
    state.auto_bounds = defaults.auto_bounds

    # Reset fitting parameters
    state.gtol = defaults.gtol
    state.ftol = defaults.ftol
    state.xtol = defaults.xtol
    state.max_iterations = defaults.max_iterations
    state.max_function_evals = defaults.max_function_evals
    state.method = defaults.method
    state.loss = defaults.loss

    # Reset multi-start settings
    state.enable_multistart = defaults.enable_multistart
    state.n_starts = defaults.n_starts
    state.sampler = defaults.sampler
    state.center_on_p0 = defaults.center_on_p0
    state.scale_factor = defaults.scale_factor

    # Reset streaming settings
    state.chunk_size = defaults.chunk_size
    state.normalize = defaults.normalize
    state.warmup_iterations = defaults.warmup_iterations
    state.max_warmup_iterations = defaults.max_warmup_iterations

    # Reset checkpointing
    state.enable_checkpoints = defaults.enable_checkpoints
    state.checkpoint_dir = defaults.checkpoint_dir

    # Reset HPC settings
    state.enable_multi_device = defaults.enable_multi_device

    # Reset defense layers
    state.defense_preset = defaults.defense_preset
    state.layer1_enabled = defaults.layer1_enabled
    state.layer1_threshold = defaults.layer1_threshold
    state.layer2_enabled = defaults.layer2_enabled
    state.layer3_enabled = defaults.layer3_enabled
    state.layer3_tolerance = defaults.layer3_tolerance
    state.layer4_enabled = defaults.layer4_enabled
    state.layer4_max_step = defaults.layer4_max_step

    # Reset batch settings
    state.batch_max_workers = defaults.batch_max_workers
    state.batch_continue_on_error = defaults.batch_continue_on_error
    state.batch_summary_format = defaults.batch_summary_format

    # Restore or reset preferences
    state.mode = saved_mode
    state.preset = saved_preset

    # Reset fit result state
    state.fit_result = defaults.fit_result
    state.fit_running = defaults.fit_running
    state.fit_aborted = defaults.fit_aborted


def get_current_config(state: SessionState) -> dict[str, Any]:
    """Export the current session state as a workflow config dictionary.

    This function converts the SessionState into a dictionary format compatible
    with the workflow_config_template.yaml structure, suitable for export or
    passing to NLSQ fitting functions.

    Parameters
    ----------
    state : SessionState
        The session state to export.

    Returns
    -------
    dict
        A workflow configuration dictionary.

    Examples
    --------
    >>> state = initialize_state()
    >>> config = get_current_config(state)
    >>> config["model"]["type"]
    'builtin'
    >>> config["fitting"]["termination"]["gtol"]
    1e-08
    """
    config: dict[str, Any] = {
        "metadata": {
            "workflow_name": "gui_workflow",
            "description": "Workflow exported from NLSQ GUI",
        },
        "model": {
            "type": state.model_type,
            "name": state.model_name,
            "auto_p0": state.auto_p0,
            "auto_bounds": state.auto_bounds,
            "polynomial": {
                "degree": state.polynomial_degree,
            },
            "custom": {
                "file": state.custom_file_path or "",
                "function": state.custom_function_name,
            },
        },
        "fitting": {
            "method": state.method,
            "objective": {
                "robust_loss": state.loss,
            },
            "termination": {
                "gtol": state.gtol,
                "ftol": state.ftol,
                "xtol": state.xtol,
                "max_iterations": state.max_iterations,
                "max_function_evals": state.max_function_evals,
            },
            "multistart": {
                "enabled": state.enable_multistart,
                "num_starts": state.n_starts,
                "sampler": state.sampler,
                "center_on_p0": state.center_on_p0,
                "scale_factor": state.scale_factor,
            },
        },
        "hybrid_streaming": {
            "normalize": state.normalize,
            "warmup_iterations": state.warmup_iterations,
            "max_warmup_iterations": state.max_warmup_iterations,
            "chunk_size": state.chunk_size,
            "enable_checkpoints": state.enable_checkpoints,
            "checkpoint_dir": state.checkpoint_dir,
            "enable_multi_device": state.enable_multi_device,
            "defense_layers": {
                "preset": state.defense_preset,
                "layer1_warm_start": {
                    "enabled": state.layer1_enabled,
                    "threshold": state.layer1_threshold,
                },
                "layer2_adaptive_lr": {
                    "enabled": state.layer2_enabled,
                },
                "layer3_cost_guard": {
                    "enabled": state.layer3_enabled,
                    "tolerance": state.layer3_tolerance,
                },
                "layer4_step_clipping": {
                    "enabled": state.layer4_enabled,
                    "max_step_size": state.layer4_max_step,
                },
            },
        },
        "batch": {
            "max_workers": state.batch_max_workers,
            "continue_on_error": state.batch_continue_on_error,
            "summary_format": state.batch_summary_format,
        },
    }

    # Add parameter configuration if present
    if state.p0 is not None:
        config["model"]["parameters"] = []
        for i, val in enumerate(state.p0):
            param_config: dict[str, Any] = {
                "name": f"param_{i}",
                "initial": val,
            }
            if state.bounds is not None:
                param_config["bounds"] = [
                    state.bounds[0][i] if state.bounds[0] else None,
                    state.bounds[1][i] if state.bounds[1] else None,
                ]
            config["model"]["parameters"].append(param_config)

    return config


def apply_preset_to_state(state: SessionState, preset_name: str) -> None:
    """Apply a preset configuration to the session state.

    Parameters
    ----------
    state : SessionState
        The session state to modify.
    preset_name : str
        The name of the preset to apply ("fast", "robust", "quality", etc.).

    Raises
    ------
    ValueError
        If the preset name is not recognized.
    """
    from nlsq.gui.presets import get_preset

    preset = get_preset(preset_name)

    state.preset = preset_name
    state.gtol = preset["gtol"]
    state.ftol = preset["ftol"]
    state.xtol = preset["xtol"]
    state.enable_multistart = preset["enable_multistart"]

    if preset["enable_multistart"]:
        state.n_starts = preset.get("n_starts", 10)
