"""Parameter configuration component for NLSQ GUI.

This module provides the parameter configuration component for the Fitting Options
page, including dynamic input fields for model parameters, p0 estimation,
bounds configuration, and parameter transforms.

Functions
---------
get_param_names_from_model
    Extract parameter names from a model function.
validate_bounds
    Validate lower/upper bound constraints.
estimate_p0_for_model
    Estimate initial parameters using model's estimate_p0 method.
render_param_config
    Render the full parameter configuration UI.
render_single_param_row
    Render a single parameter's configuration row.
"""

import inspect
from collections.abc import Callable
from typing import Any

import streamlit as st

from nlsq.gui.state import SessionState

# Available transform options
TRANSFORM_OPTIONS = ["none", "log", "logit", "exp"]


def get_param_names_from_model(model: Callable) -> list[str]:
    """Extract parameter names from a model function.

    Parameters
    ----------
    model : Callable
        The model function to inspect.

    Returns
    -------
    list[str]
        List of parameter names (excluding the independent variable x).

    Examples
    --------
    >>> from nlsq.core.functions import gaussian
    >>> names = get_param_names_from_model(gaussian)
    >>> print(names)
    ['amp', 'mu', 'sigma']
    """
    param_names: list[str] = []
    has_var_positional = False

    try:
        sig = inspect.signature(model)
        params = list(sig.parameters.items())

        # Check if any parameter is VAR_POSITIONAL (*args)
        for name, param in params:
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                has_var_positional = True
                break

        if not has_var_positional:
            # Standard case: extract parameter names after x
            param_keys = [name for name, _ in params]
            if param_keys and param_keys[0] in ("x", "X", "xdata"):
                param_names = param_keys[1:]
            else:
                param_names = param_keys[1:] if len(param_keys) > 1 else []
    except (ValueError, TypeError):
        pass

    # For functions with *args (like polynomial), use estimate_p0 to get count
    if has_var_positional and hasattr(model, "estimate_p0"):
        try:
            # Try with dummy data
            p0 = model.estimate_p0([1, 2, 3], [1, 4, 9])
            param_names = [f"c{i}" for i in range(len(p0))]
        except Exception:
            pass

    return param_names


def validate_bounds(lower: float | None, upper: float | None) -> tuple[bool, str]:
    """Validate lower and upper bound constraints.

    Parameters
    ----------
    lower : float or None
        Lower bound value.
    upper : float or None
        Upper bound value.

    Returns
    -------
    tuple[bool, str]
        Tuple of (is_valid, error_message).
        If valid, error_message is empty string.

    Examples
    --------
    >>> validate_bounds(0.0, 10.0)
    (True, '')
    >>> validate_bounds(10.0, 0.0)
    (False, 'Lower bound must be <= upper bound')
    """
    # None bounds are always valid
    if lower is None or upper is None:
        return True, ""

    # Check lower <= upper
    if lower > upper:
        return False, "Lower bound must be <= upper bound"

    return True, ""


def get_default_p0_value() -> float:
    """Get the default initial parameter value.

    Returns
    -------
    float
        Default p0 value (1.0).
    """
    return 1.0


def create_param_config_dict(param_names: list[str]) -> dict[str, dict[str, Any]]:
    """Create a parameter configuration dictionary.

    Parameters
    ----------
    param_names : list[str]
        List of parameter names.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping parameter names to their configuration.
        Each config has keys: p0, lower, upper, transform, auto.
    """
    config: dict[str, dict[str, Any]] = {}

    for name in param_names:
        config[name] = {
            "p0": None,
            "lower": None,
            "upper": None,
            "transform": "none",
            "auto": True,
        }

    return config


def estimate_p0_for_model(
    model: Callable, xdata: Any, ydata: Any
) -> list[float] | None:
    """Estimate initial parameters using model's estimate_p0 method.

    Parameters
    ----------
    model : Callable
        The model function with optional estimate_p0 method.
    xdata : array-like
        Independent variable data.
    ydata : array-like
        Dependent variable data.

    Returns
    -------
    list[float] or None
        Estimated p0 values, or None if estimation not available.

    Examples
    --------
    >>> from nlsq.core.functions import linear
    >>> p0 = estimate_p0_for_model(linear, [0, 1, 2], [1, 3, 5])
    >>> len(p0)
    2
    """
    if not hasattr(model, "estimate_p0") or not callable(model.estimate_p0):
        return None

    try:
        p0 = model.estimate_p0(xdata, ydata)
        if p0 is not None:
            return list(p0)
    except Exception:
        pass

    return None


def format_p0_display(value: float | None) -> str:
    """Format a p0 value for display.

    Parameters
    ----------
    value : float or None
        The p0 value to format.

    Returns
    -------
    str
        Formatted display string.
    """
    if value is None:
        return "Auto"

    # Use scientific notation for very small or large values
    if abs(value) < 1e-3 or abs(value) > 1e6:
        return f"{value:.3e}"

    return f"{value:.6g}"


def validate_p0_input(text: str) -> tuple[bool, float | None, str]:
    """Validate a p0 input string.

    Parameters
    ----------
    text : str
        Input text to validate.

    Returns
    -------
    tuple[bool, float | None, str]
        Tuple of (is_valid, parsed_value, error_message).
        If empty string, returns (True, None, '') for auto mode.
    """
    text = text.strip()

    if not text:
        return True, None, ""

    try:
        value = float(text)
        return True, value, ""
    except ValueError:
        return False, None, f"Invalid number: {text}"


def render_param_config(
    state: SessionState,
    model: Callable,
    xdata: Any | None = None,
    ydata: Any | None = None,
) -> None:
    """Render the full parameter configuration UI.

    Parameters
    ----------
    state : SessionState
        The current session state.
    model : Callable
        The model function to configure parameters for.
    xdata : array-like or None
        Data for p0 estimation (optional).
    ydata : array-like or None
        Data for p0 estimation (optional).
    """
    st.subheader("Parameter Configuration")

    # Get parameter names
    param_names = get_param_names_from_model(model)

    if not param_names:
        st.warning("Could not determine model parameters")
        return

    # Check for auto_p0 capability
    has_auto_p0 = hasattr(model, "estimate_p0") and callable(
        getattr(model, "estimate_p0", None)
    )

    # Display auto_p0 status
    if has_auto_p0:
        st.info("This model supports automatic initial parameter estimation (auto p0)")
    else:
        st.caption(
            "This model does not have built-in p0 estimation. "
            "Please provide initial guesses."
        )

    # Global auto_p0 toggle
    auto_p0 = st.checkbox(
        "Use automatic p0 estimation",
        value=state.auto_p0,
        disabled=not has_auto_p0,
        help="Automatically estimate initial parameters from data",
    )
    state.auto_p0 = auto_p0

    # Get estimated p0 if available and data exists
    estimated_p0: list[float] | None = None
    if auto_p0 and has_auto_p0 and xdata is not None and ydata is not None:
        estimated_p0 = estimate_p0_for_model(model, xdata, ydata)

    st.divider()

    # Initialize p0 list if needed
    if state.p0 is None or len(state.p0) != len(param_names):
        state.p0 = [None] * len(param_names)

    # Initialize bounds if needed
    if state.bounds is None:
        state.bounds = ([None] * len(param_names), [None] * len(param_names))

    # Initialize transforms if needed
    if not state.transforms:
        state.transforms = {}

    # Render each parameter row
    for i, param_name in enumerate(param_names):
        render_single_param_row(
            param_name=param_name,
            index=i,
            state=state,
            estimated_p0=estimated_p0[i]
            if estimated_p0 and i < len(estimated_p0)
            else None,
            use_auto=auto_p0 and has_auto_p0,
        )

    # Show summary
    st.divider()
    render_param_summary(state, param_names)


def render_single_param_row(
    param_name: str,
    index: int,
    state: SessionState,
    estimated_p0: float | None = None,
    use_auto: bool = False,
) -> None:
    """Render a single parameter's configuration row.

    Parameters
    ----------
    param_name : str
        Name of the parameter.
    index : int
        Index of the parameter in the list.
    state : SessionState
        The current session state.
    estimated_p0 : float or None
        Estimated p0 value if available.
    use_auto : bool
        Whether auto mode is enabled.
    """
    with st.container():
        st.markdown(f"**{param_name}**")

        cols = st.columns([2, 1, 1, 1, 1])

        # p0 input
        with cols[0]:
            # Get current value
            current_p0 = state.p0[index] if state.p0 and index < len(state.p0) else None

            if use_auto and estimated_p0 is not None:
                # Show estimated value as placeholder
                p0_input = st.text_input(
                    "Initial (p0)",
                    value="" if current_p0 is None else str(current_p0),
                    placeholder=f"Auto: {format_p0_display(estimated_p0)}",
                    key=f"p0_{param_name}_{index}",
                    help=f"Initial guess for {param_name}. Leave empty for auto.",
                )
            else:
                p0_input = st.text_input(
                    "Initial (p0)",
                    value="" if current_p0 is None else str(current_p0),
                    placeholder="1.0",
                    key=f"p0_{param_name}_{index}",
                    help=f"Initial guess for {param_name}",
                )

            # Validate and store
            is_valid, parsed_value, error = validate_p0_input(p0_input)
            if not is_valid:
                st.error(error)
            else:
                if state.p0 is None:
                    state.p0 = [None] * (index + 1)
                while len(state.p0) <= index:
                    state.p0.append(None)
                state.p0[index] = parsed_value

        # Lower bound
        with cols[1]:
            current_lower = (
                state.bounds[0][index]
                if state.bounds and state.bounds[0] and index < len(state.bounds[0])
                else None
            )
            lower_input = st.text_input(
                "Lower",
                value="" if current_lower is None else str(current_lower),
                placeholder="-inf",
                key=f"lower_{param_name}_{index}",
                help="Lower bound (optional)",
            )

            if lower_input.strip():
                try:
                    lower_val = float(lower_input)
                    if state.bounds:
                        while len(state.bounds[0]) <= index:
                            state.bounds[0].append(None)
                        state.bounds[0][index] = lower_val
                except ValueError:
                    st.error("Invalid")

        # Upper bound
        with cols[2]:
            current_upper = (
                state.bounds[1][index]
                if state.bounds and state.bounds[1] and index < len(state.bounds[1])
                else None
            )
            upper_input = st.text_input(
                "Upper",
                value="" if current_upper is None else str(current_upper),
                placeholder="+inf",
                key=f"upper_{param_name}_{index}",
                help="Upper bound (optional)",
            )

            if upper_input.strip():
                try:
                    upper_val = float(upper_input)
                    if state.bounds:
                        while len(state.bounds[1]) <= index:
                            state.bounds[1].append(None)
                        state.bounds[1][index] = upper_val
                except ValueError:
                    st.error("Invalid")

        # Validate bounds
        if state.bounds:
            lower = state.bounds[0][index] if index < len(state.bounds[0]) else None
            upper = state.bounds[1][index] if index < len(state.bounds[1]) else None
            is_valid_bounds, bound_error = validate_bounds(lower, upper)
            if not is_valid_bounds:
                st.error(bound_error)

        # Transform dropdown
        with cols[3]:
            current_transform = state.transforms.get(param_name, "none")
            transform = st.selectbox(
                "Transform",
                options=TRANSFORM_OPTIONS,
                index=TRANSFORM_OPTIONS.index(current_transform)
                if current_transform in TRANSFORM_OPTIONS
                else 0,
                key=f"transform_{param_name}_{index}",
                help="Parameter transform",
            )
            state.transforms[param_name] = transform

        # Auto indicator
        with cols[4]:
            if use_auto and estimated_p0 is not None and state.p0[index] is None:
                st.success("Auto")
            elif state.p0[index] is not None:
                st.info("Set")
            else:
                st.warning("Unset")


def render_param_summary(state: SessionState, param_names: list[str]) -> None:
    """Render a summary of parameter configuration.

    Parameters
    ----------
    state : SessionState
        The current session state.
    param_names : list[str]
        List of parameter names.
    """
    st.caption("Configuration Summary")

    # Count configured parameters
    n_total = len(param_names)
    n_p0_set = sum(
        1
        for i in range(n_total)
        if state.p0 and i < len(state.p0) and state.p0[i] is not None
    )
    n_bounded = 0

    if state.bounds:
        for i in range(n_total):
            lower = state.bounds[0][i] if i < len(state.bounds[0]) else None
            upper = state.bounds[1][i] if i < len(state.bounds[1]) else None
            if lower is not None or upper is not None:
                n_bounded += 1

    n_transformed = sum(
        1 for name in param_names if state.transforms.get(name, "none") != "none"
    )

    # Display summary
    col1, col2, col3 = st.columns(3)

    with col1:
        if state.auto_p0:
            st.metric("Initial Values", f"Auto ({n_p0_set} custom)")
        else:
            st.metric("Initial Values", f"{n_p0_set}/{n_total} set")

    with col2:
        st.metric("Bounded", f"{n_bounded}/{n_total}")

    with col3:
        st.metric("Transformed", f"{n_transformed}/{n_total}")
