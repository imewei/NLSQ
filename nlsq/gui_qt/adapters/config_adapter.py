"""Workflow configuration adapter for the NLSQ GUI.

This module provides functions for importing and exporting workflow configurations
in YAML format, enabling interoperability between the GUI and CLI workflows.

The adapter maps all fields from workflow_config_template.yaml to SessionState
fields and vice versa.
"""

import contextlib
from typing import Any, TextIO

import numpy as np
import yaml

from nlsq.gui_qt.session_state import (
    VALID_WORKFLOWS,
    SessionState,
    initialize_state,
)

_SECTION_KEYS = (
    "model",
    "fitting",
    "hybrid_streaming",
    "batch",
    "global_optimization",
)


def _parse_model_section(config: dict[str, Any], state: SessionState) -> None:
    """Parse model section of YAML config into state."""
    if "model" not in config:
        return

    model = config["model"]
    state.model_type = model.get("type", state.model_type)
    state.model_name = model.get("name", state.model_name)
    state.auto_p0 = model.get("auto_p0", state.auto_p0)
    state.auto_bounds = model.get("auto_bounds", state.auto_bounds)
    state.polynomial_degree = model.get("degree", state.polynomial_degree)
    state.custom_file_path = model.get("path", state.custom_file_path)
    state.custom_function_name = model.get("function", state.custom_function_name)

    if "polynomial" in model:
        state.polynomial_degree = model["polynomial"].get(
            "degree",
            state.polynomial_degree,
        )

    if "custom" in model:
        custom = model["custom"]
        state.custom_file_path = custom.get("file") or None  # "" means unset
        state.custom_function_name = custom.get("function", "")
        state.custom_code = custom.get("code") or state.custom_code

    # The model page reads model_config, not the dedicated fields: rebuild it.
    if state.model_type == "polynomial":
        state.model_config = {"degree": state.polynomial_degree}
    elif state.model_type == "custom":
        state.model_config = {
            k: v
            for k, v in (
                ("code", state.custom_code),
                # The model page Apply sends {code, function}; a path next to
                # code would look like a config change and reset p0/bounds.
                ("path", None if state.custom_code else state.custom_file_path),
                ("function", state.custom_function_name),
            )
            if v
        }
    elif "name" in model:
        state.model_config = {"name": state.model_name}

    if "parameters" in model:
        _parse_model_parameters(model["parameters"], state)


def _parse_model_parameters(
    parameters: list[dict[str, Any]],
    state: SessionState,
) -> None:
    """Parse model parameters from YAML config."""
    p0_values = []
    lower_bounds = []
    upper_bounds = []
    transforms = {}

    for param in parameters:
        if "initial" in param:
            p0_values.append(
                _num(param["initial"], f"parameter {param.get('name', '?')!r} initial")
            )
        bounds = param.get("bounds")
        if bounds is None:
            bounds = [None, None]
        if len(bounds) != 2:
            import logging

            logging.getLogger(__name__).warning(
                "Parameter 'bounds' expected 2 elements [lower, upper], "
                "got %d: %r — malformed entries default to unbounded",
                len(bounds),
                bounds,
            )
        lower = bounds[0] if len(bounds) > 0 and bounds[0] is not None else None
        upper = bounds[1] if len(bounds) > 1 and bounds[1] is not None else None
        pname = f"parameter {param.get('name', '?')!r} bounds"
        lower_bounds.append(float("-inf") if lower is None else _num(lower, pname))
        upper_bounds.append(float("inf") if upper is None else _num(upper, pname))
        if "transform" in param and "name" in param:
            transforms[param["name"]] = param["transform"]

    if p0_values:
        if len(p0_values) != len(parameters):
            raise ValueError("Provide an initial value for every model parameter")
        state.p0 = p0_values
    if any("bounds" in param for param in parameters):
        state.bounds = (lower_bounds, upper_bounds)
    if transforms:
        state.transforms = transforms


def _num(value: Any, name: str) -> float:
    """Coerce a YAML scalar to float (PyYAML reads ``1e-3`` as a string)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r}") from None


def _nfev(value: Any, name: str) -> int | None:
    from nlsq.core._option_checks import is_integral_count

    if value is None:  # explicit null: let curve_fit pick its default
        return None

    if isinstance(value, str):  # hand-edited YAML often quotes numbers
        with contextlib.suppress(ValueError):
            value = float(value)
    if not is_integral_count(value, 1):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _parse_fitting_section(config: dict[str, Any], state: SessionState) -> None:
    """Parse fitting section of YAML config into state."""

    def _term(options: dict[str, Any], key: str, default: Any) -> Any:
        value = options.get(key)
        return default if value is None else value

    if "fitting" not in config:
        return

    fitting = config["fitting"]
    state.method = fitting.get("method", state.method)
    state.workflow = fitting.get("workflow", state.workflow)
    if state.workflow not in VALID_WORKFLOWS:
        raise ValueError(
            f"Unknown fitting.workflow {state.workflow!r}; "
            f"expected one of {sorted(VALID_WORKFLOWS)}"
        )
    state.loss = fitting.get("loss", state.loss)
    for name in ("gtol", "ftol", "xtol"):
        setattr(state, name, float(_term(fitting, name, getattr(state, name))))
    state.max_function_evals = _nfev(
        _term(fitting, "max_nfev", state.max_function_evals), "fitting.max_nfev"
    )
    p0 = fitting.get("p0")
    if isinstance(p0, (list, tuple)):
        state.p0 = [_num(v, "fitting.p0 entry") for v in p0]
        state.auto_p0 = False
    bounds = fitting.get("bounds")
    if isinstance(bounds, dict):
        bounds = (bounds.get("lower", -np.inf), bounds.get("upper", np.inf))
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        try:
            lower, upper = np.broadcast_arrays(
                np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"fitting.bounds must be numeric, got {bounds!r}"
            ) from None
        if state.p0 is not None:
            lower = np.broadcast_to(lower, (len(state.p0),))
            upper = np.broadcast_to(upper, (len(state.p0),))
        state.bounds = (np.atleast_1d(lower).tolist(), np.atleast_1d(upper).tolist())

    if "global_optimization" in fitting:
        go = fitting["global_optimization"]
        state.n_starts = go.get("n_starts", state.n_starts)
        state.sampler = go.get("sampler", state.sampler)
        state.center_on_p0 = go.get("center_on_p0", state.center_on_p0)
        state.scale_factor = float(go.get("scale_factor", state.scale_factor))

    if "objective" in fitting:
        state.loss = fitting["objective"].get("robust_loss", state.loss)

    if "termination" in fitting:
        term = fitting["termination"] or {}

        # Coerce like the flat gtol/ftol/xtol branch: YAML reads `1e-10` as str.
        state.gtol = float(_term(term, "gtol", state.gtol))
        state.ftol = float(_term(term, "ftol", state.ftol))
        state.xtol = float(_term(term, "xtol", state.xtol))
        state.max_iterations = int(_term(term, "max_iterations", state.max_iterations))
        state.max_function_evals = _nfev(
            _term(term, "max_function_evals", state.max_function_evals),
            "termination.max_function_evals",
        )

    if "multistart" in fitting:
        ms = fitting["multistart"]
        if ms.get("enabled", False):
            state.workflow = "auto_global"
        state.n_starts = ms.get("num_starts", state.n_starts)
        state.sampler = ms.get("sampler", state.sampler)
        state.center_on_p0 = ms.get("center_on_p0", state.center_on_p0)
        state.scale_factor = float(ms.get("scale_factor", state.scale_factor))


def _parse_hybrid_streaming_section(
    config: dict[str, Any],
    state: SessionState,
) -> None:
    """Parse hybrid streaming section of YAML config into state."""
    if "hybrid_streaming" not in config:
        return

    hs = config["hybrid_streaming"]
    state.normalize = hs.get("normalize", state.normalize)
    state.warmup_iterations = hs.get("warmup_iterations", state.warmup_iterations)
    state.max_warmup_iterations = hs.get(
        "max_warmup_iterations",
        state.max_warmup_iterations,
    )
    state.chunk_size = hs.get("chunk_size", state.chunk_size)
    state.enable_checkpoints = hs.get("enable_checkpoints", state.enable_checkpoints)
    state.checkpoint_dir = hs.get("checkpoint_dir", state.checkpoint_dir)

    if "defense_layers" in hs:
        _parse_defense_layers(hs["defense_layers"], state)


def _parse_defense_layers(dl: dict[str, Any], state: SessionState) -> None:
    """Parse defense layer settings from YAML config."""
    state.defense_preset = dl.get("preset", state.defense_preset)

    if "layer1_warm_start" in dl:
        l1 = dl["layer1_warm_start"]
        state.layer1_enabled = l1.get("enabled", state.layer1_enabled)
        state.layer1_threshold = l1.get("threshold", state.layer1_threshold)

    if "layer2_adaptive_lr" in dl:
        state.layer2_enabled = dl["layer2_adaptive_lr"].get(
            "enabled",
            state.layer2_enabled,
        )

    if "layer3_cost_guard" in dl:
        l3 = dl["layer3_cost_guard"]
        state.layer3_enabled = l3.get("enabled", state.layer3_enabled)
        state.layer3_tolerance = l3.get("tolerance", state.layer3_tolerance)

    if "layer4_step_clipping" in dl:
        l4 = dl["layer4_step_clipping"]
        state.layer4_enabled = l4.get("enabled", state.layer4_enabled)
        state.layer4_max_step = l4.get("max_step_size", state.layer4_max_step)


def _parse_batch_section(config: dict[str, Any], state: SessionState) -> None:
    """Parse batch section of YAML config into state."""
    if "batch" not in config:
        return

    batch = config["batch"]
    state.batch_max_workers = batch.get("max_workers", state.batch_max_workers)
    state.batch_continue_on_error = batch.get(
        "continue_on_error",
        state.batch_continue_on_error,
    )
    state.batch_summary_format = batch.get("summary_format", state.batch_summary_format)


def _parse_global_optimization_section(
    config: dict[str, Any],
    state: SessionState,
) -> None:
    """Parse global optimization section of YAML config into state."""
    if "global_optimization" not in config:
        return

    go = config["global_optimization"]
    if "n_starts" in go and go["n_starts"] > 0:
        state.workflow = "auto_global"
        state.n_starts = go["n_starts"]
    state.sampler = go.get("sampler", state.sampler)
    state.center_on_p0 = go.get("center_on_p0", state.center_on_p0)
    state.scale_factor = float(go.get("scale_factor", state.scale_factor))


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

    if not isinstance(config, dict):
        raise ValueError(
            "Invalid YAML config: expected a mapping at the top level, "
            f"got {type(config).__name__}",
        )
    for _section_key in _SECTION_KEYS:
        _section_value = config.get(_section_key)
        if _section_value is not None and not isinstance(_section_value, dict):
            raise ValueError(
                f"Invalid YAML config: '{_section_key}' section must be a "
                f"mapping, got {type(_section_value).__name__}",
            )

    # A heading whose children are all commented out parses as None; the CLI
    # treats that as an empty section, so the parsers below must too.
    config = {
        k: ({} if v is None and k in _SECTION_KEYS else v) for k, v in config.items()
    }

    state = initialize_state()

    _parse_model_section(config, state)
    _parse_fitting_section(config, state)
    _parse_hybrid_streaming_section(config, state)
    _parse_batch_section(config, state)
    _parse_global_optimization_section(config, state)

    return state


def export_yaml_config(state: SessionState) -> str:
    """Export a SessionState as a YAML configuration string.

    This function converts a SessionState into a YAML string in the GUI's own
    session schema (see ``get_current_config``). It is not runnable with
    ``nlsq fit``: it has no ``data`` section and uses nested fitting blocks
    that the CLI schema does not read.

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
    from nlsq.gui_qt.session_state import get_current_config

    config = get_current_config(state)

    yaml_str = yaml.safe_dump(
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
    base: SessionState,
    overlay: SessionState,
    fields: list[str] | None = None,
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

            # Plain `!=` raises on numpy arrays (xdata/ydata/sigma) since their
            # comparison returns an elementwise array, not a bool.
            if isinstance(overlay_value, np.ndarray) or isinstance(
                default_value,
                np.ndarray,
            ):
                changed = not np.array_equal(overlay_value, default_value)
            else:
                changed = overlay_value != default_value

            if changed:
                setattr(result, field_name, overlay_value)

    return result
