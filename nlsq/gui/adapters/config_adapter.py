"""Workflow configuration adapter for the NLSQ GUI.

This module provides functions for importing and exporting workflow configurations
in YAML format, enabling interoperability between the GUI and CLI workflows.

The adapter maps all fields from workflow_config_template.yaml to SessionState
fields and vice versa.
"""

from typing import TextIO

import yaml

from nlsq.gui.state import SessionState, initialize_state


def load_yaml_config(file: TextIO | str) -> SessionState:
    """Load a YAML workflow configuration file into a SessionState.

    This function parses a YAML configuration file (compatible with
    workflow_config_template.yaml) and creates a SessionState with the
    corresponding values.

    Parameters
    ----------
    file : TextIO or str
        A file-like object or string containing YAML configuration.

    Returns
    -------
    SessionState
        A new SessionState populated with values from the YAML config.

    Examples
    --------
    >>> from io import StringIO
    >>> yaml_content = '''
    ... model:
    ...   type: builtin
    ...   name: gaussian
    ... fitting:
    ...   termination:
    ...     gtol: 1.0e-10
    ... '''
    >>> state = load_yaml_config(StringIO(yaml_content))
    >>> state.model_name
    'gaussian'
    >>> state.gtol
    1e-10
    """
    if isinstance(file, str):
        config = yaml.safe_load(file)
    else:
        config = yaml.safe_load(file)

    if config is None:
        config = {}

    # Initialize state with defaults
    state = initialize_state()

    # Parse model section
    if "model" in config:
        model = config["model"]
        state.model_type = model.get("type", state.model_type)
        state.model_name = model.get("name", state.model_name)
        state.auto_p0 = model.get("auto_p0", state.auto_p0)
        state.auto_bounds = model.get("auto_bounds", state.auto_bounds)

        # Polynomial configuration
        if "polynomial" in model:
            state.polynomial_degree = model["polynomial"].get(
                "degree", state.polynomial_degree
            )

        # Custom model configuration
        if "custom" in model:
            state.custom_file_path = model["custom"].get("file", None)
            state.custom_function_name = model["custom"].get("function", "")

        # Parameter definitions
        if "parameters" in model:
            p0_values = []
            lower_bounds = []
            upper_bounds = []
            transforms = {}

            for param in model["parameters"]:
                if "initial" in param:
                    p0_values.append(param["initial"])
                if "bounds" in param:
                    bounds = param["bounds"]
                    lower_bounds.append(
                        bounds[0] if bounds[0] is not None else float("-inf")
                    )
                    upper_bounds.append(
                        bounds[1] if bounds[1] is not None else float("inf")
                    )
                if "transform" in param and "name" in param:
                    transforms[param["name"]] = param["transform"]

            if p0_values:
                state.p0 = p0_values
            if lower_bounds or upper_bounds:
                state.bounds = (lower_bounds, upper_bounds)
            if transforms:
                state.transforms = transforms

    # Parse fitting section
    if "fitting" in config:
        fitting = config["fitting"]
        state.method = fitting.get("method", state.method)

        # Objective settings
        if "objective" in fitting:
            objective = fitting["objective"]
            state.loss = objective.get("robust_loss", state.loss)

        # Termination criteria
        if "termination" in fitting:
            term = fitting["termination"]
            state.gtol = term.get("gtol", state.gtol)
            state.ftol = term.get("ftol", state.ftol)
            state.xtol = term.get("xtol", state.xtol)
            state.max_iterations = term.get("max_iterations", state.max_iterations)
            state.max_function_evals = term.get(
                "max_function_evals", state.max_function_evals
            )

        # Multi-start settings
        if "multistart" in fitting:
            ms = fitting["multistart"]
            state.enable_multistart = ms.get("enabled", state.enable_multistart)
            state.n_starts = ms.get("num_starts", state.n_starts)
            state.sampler = ms.get("sampler", state.sampler)
            state.center_on_p0 = ms.get("center_on_p0", state.center_on_p0)
            state.scale_factor = ms.get("scale_factor", state.scale_factor)

    # Parse hybrid streaming section
    if "hybrid_streaming" in config:
        hs = config["hybrid_streaming"]
        state.normalize = hs.get("normalize", state.normalize)
        state.warmup_iterations = hs.get("warmup_iterations", state.warmup_iterations)
        state.max_warmup_iterations = hs.get(
            "max_warmup_iterations", state.max_warmup_iterations
        )
        state.chunk_size = hs.get("chunk_size", state.chunk_size)
        state.enable_checkpoints = hs.get(
            "enable_checkpoints", state.enable_checkpoints
        )
        state.checkpoint_dir = hs.get("checkpoint_dir", state.checkpoint_dir)
        state.enable_multi_device = hs.get(
            "enable_multi_device", state.enable_multi_device
        )

        # Defense layer settings
        if "defense_layers" in hs:
            dl = hs["defense_layers"]
            state.defense_preset = dl.get("preset", state.defense_preset)

            if "layer1_warm_start" in dl:
                l1 = dl["layer1_warm_start"]
                state.layer1_enabled = l1.get("enabled", state.layer1_enabled)
                state.layer1_threshold = l1.get("threshold", state.layer1_threshold)

            if "layer2_adaptive_lr" in dl:
                l2 = dl["layer2_adaptive_lr"]
                state.layer2_enabled = l2.get("enabled", state.layer2_enabled)

            if "layer3_cost_guard" in dl:
                l3 = dl["layer3_cost_guard"]
                state.layer3_enabled = l3.get("enabled", state.layer3_enabled)
                state.layer3_tolerance = l3.get("tolerance", state.layer3_tolerance)

            if "layer4_step_clipping" in dl:
                l4 = dl["layer4_step_clipping"]
                state.layer4_enabled = l4.get("enabled", state.layer4_enabled)
                state.layer4_max_step = l4.get("max_step_size", state.layer4_max_step)

    # Parse batch section
    if "batch" in config:
        batch = config["batch"]
        state.batch_max_workers = batch.get("max_workers", state.batch_max_workers)
        state.batch_continue_on_error = batch.get(
            "continue_on_error", state.batch_continue_on_error
        )
        state.batch_summary_format = batch.get(
            "summary_format", state.batch_summary_format
        )

    # Parse global optimization section (alternative multi-start config)
    if "global_optimization" in config:
        go = config["global_optimization"]
        if "n_starts" in go and go["n_starts"] > 0:
            state.enable_multistart = True
            state.n_starts = go["n_starts"]
        state.sampler = go.get("sampler", state.sampler)
        state.center_on_p0 = go.get("center_on_p0", state.center_on_p0)
        state.scale_factor = go.get("scale_factor", state.scale_factor)

    return state


def export_yaml_config(state: SessionState) -> str:
    """Export a SessionState as a YAML configuration string.

    This function converts a SessionState into a YAML string compatible
    with workflow_config_template.yaml format.

    Parameters
    ----------
    state : SessionState
        The session state to export.

    Returns
    -------
    str
        A YAML-formatted configuration string.

    Examples
    --------
    >>> state = initialize_state()
    >>> state.model_name = "gaussian"
    >>> yaml_str = export_yaml_config(state)
    >>> "gaussian" in yaml_str
    True
    """
    from nlsq.gui.state import get_current_config

    config = get_current_config(state)

    # Use yaml.dump with flow style for compact output
    yaml_str = yaml.dump(
        config,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=80,
    )

    return yaml_str


def import_cli_workflow(file_path: str) -> SessionState:
    """Import a CLI workflow configuration file.

    This is a convenience function that opens and parses a YAML file
    from the filesystem.

    Parameters
    ----------
    file_path : str
        Path to the YAML configuration file.

    Returns
    -------
    SessionState
        A new SessionState populated with values from the file.
    """
    with open(file_path, encoding="utf-8") as f:
        return load_yaml_config(f)


def validate_yaml_config(yaml_content: str) -> tuple[bool, str | None]:
    """Validate a YAML configuration string.

    Parameters
    ----------
    yaml_content : str
        The YAML content to validate.

    Returns
    -------
    tuple[bool, str | None]
        A tuple of (is_valid, error_message).
        If valid, error_message is None.
    """
    try:
        config = yaml.safe_load(yaml_content)

        if config is None:
            return False, "Empty or invalid YAML content"

        if not isinstance(config, dict):
            return False, "YAML content must be a dictionary"

        # Basic structure validation
        valid_sections = {
            "metadata",
            "model",
            "fitting",
            "data",
            "paths",
            "runtime",
            "logging",
            "preprocessing",
            "hybrid_streaming",
            "global_optimization",
            "optimization",
            "workflow_steps",
            "validation_and_qc",
            "reporting",
            "visualization",
            "export",
            "batch",
            "advanced",
            "default_workflow",
            "memory_limit_gb",
            "workflows",
        }

        unknown_sections = set(config.keys()) - valid_sections
        if unknown_sections:
            # Warning but not error - allow unknown sections
            pass

        return True, None

    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {e}"


def merge_configs(
    base: SessionState, overlay: SessionState, fields: list[str] | None = None
) -> SessionState:
    """Merge two SessionState instances.

    Values from overlay override values in base, but only for specified fields
    or all fields if fields is None.

    Parameters
    ----------
    base : SessionState
        The base session state.
    overlay : SessionState
        The overlay session state with values to merge in.
    fields : list[str] or None, optional
        List of field names to merge. If None, all fields are merged.

    Returns
    -------
    SessionState
        A new SessionState with merged values.
    """
    result = base.copy()

    if fields is None:
        # Get all dataclass fields
        from dataclasses import fields as dc_fields

        fields = [f.name for f in dc_fields(SessionState)]

    for field_name in fields:
        if hasattr(overlay, field_name):
            overlay_value = getattr(overlay, field_name)
            # Only override if overlay value is not the default
            default_state = SessionState()
            default_value = getattr(default_state, field_name)

            if overlay_value != default_value:
                setattr(result, field_name, overlay_value)

    return result
