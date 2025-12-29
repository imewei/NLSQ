"""Model Selection Page for NLSQ GUI.

This page provides the interface for selecting the model function for curve fitting.
Users can choose from built-in models, polynomial models, or define custom models.

Model Types:
- Built-in: 7 predefined functions (linear, exponential_decay, etc.)
- Polynomial: Configurable degree 0-10
- Custom: User-defined Python code or uploaded .py file
"""

from typing import Any

import streamlit as st

from nlsq.gui.adapters.model_adapter import (
    get_model,
    get_model_info,
    list_builtin_models,
    list_functions_in_module,
)
from nlsq.gui.components.code_editor import (
    get_default_model_template,
    validate_code_syntax,
)
from nlsq.gui.components.model_preview import (
    format_parameter_list,
    get_model_summary,
    render_model_preview,
)
from nlsq.gui.state import SessionState, initialize_state
from nlsq.gui.utils.theme import apply_dark_theme_css

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Model Selection - NLSQ",
    page_icon="chart_with_upwards_trend",
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


def init_model_page_state() -> None:
    """Initialize model page specific state."""
    if "custom_code" not in st.session_state:
        st.session_state.custom_code = get_default_model_template()
    if "custom_function_name" not in st.session_state:
        st.session_state.custom_function_name = "model"
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False


# =============================================================================
# Model Loading Functions
# =============================================================================


def load_builtin_model(model_name: str) -> Any:
    """Load a built-in model by name.

    Parameters
    ----------
    model_name : str
        Name of the built-in model.

    Returns
    -------
    Callable
        The model function.
    """
    return get_model("builtin", {"name": model_name})


def load_polynomial_model(degree: int) -> Any:
    """Load a polynomial model of given degree.

    Parameters
    ----------
    degree : int
        Polynomial degree (0-10).

    Returns
    -------
    Callable
        The polynomial model function.
    """
    return get_model("polynomial", {"degree": degree})


def load_custom_model(code: str, function_name: str) -> Any | None:
    """Load a custom model from code string.

    Parameters
    ----------
    code : str
        Python source code containing the model function.
    function_name : str
        Name of the function to use.

    Returns
    -------
    Callable | None
        The model function, or None if loading fails.
    """
    try:
        return get_model("custom", {"code": code, "function": function_name})
    except Exception:
        return None


# =============================================================================
# UI Rendering Functions
# =============================================================================


def render_model_type_selector() -> str:
    """Render the model type selector.

    Returns
    -------
    str
        Selected model type: "Built-in", "Polynomial", or "Custom".
    """
    state = get_session_state()

    # Map internal type to display name
    type_mapping = {
        "builtin": "Built-in",
        "polynomial": "Polynomial",
        "custom": "Custom",
    }
    display_mapping = {v: k for k, v in type_mapping.items()}

    current_display = type_mapping.get(state.model_type, "Built-in")

    model_type = st.radio(
        "Model Type",
        options=["Built-in", "Polynomial", "Custom"],
        index=["Built-in", "Polynomial", "Custom"].index(current_display),
        horizontal=True,
        help="Select the type of model function to use for fitting",
    )

    # Update state with internal type
    state.model_type = display_mapping[model_type]

    return model_type


def render_builtin_model_selector() -> None:
    """Render the built-in model dropdown selector."""
    state = get_session_state()

    st.subheader("Built-in Models")

    # Get list of available models
    models = list_builtin_models()
    model_names = [m["name"] for m in models if m["name"] != "polynomial"]

    # Sort alphabetically for display
    model_names.sort()

    # Build options with parameter count info
    options_with_info = {
        name: f"{name} ({next((m['n_params'] for m in models if m['name'] == name), '?')} params)"
        for name in model_names
    }

    # Get current selection index
    current_name = state.model_name
    if current_name not in model_names:
        current_name = model_names[0] if model_names else "exponential_decay"

    selected_name = st.selectbox(
        "Select Model",
        options=model_names,
        index=model_names.index(current_name) if current_name in model_names else 0,
        format_func=lambda x: options_with_info.get(x, x),
        help="Choose from the available built-in model functions",
    )

    state.model_name = selected_name

    # Load and display model info
    try:
        model = load_builtin_model(selected_name)
        st.session_state.current_model = model
        st.session_state.model_loaded = True

        # Show model preview
        render_model_preview(model, "builtin", selected_name)

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.session_state.model_loaded = False


def render_polynomial_selector() -> None:
    """Render the polynomial degree selector."""
    state = get_session_state()

    st.subheader("Polynomial Model")

    degree = st.slider(
        "Polynomial Degree",
        min_value=0,
        max_value=10,
        value=state.polynomial_degree,
        help="Select the degree of the polynomial (number of terms = degree + 1)",
    )

    state.polynomial_degree = degree

    # Show degree explanation
    st.caption(
        f"A degree-{degree} polynomial has {degree + 1} parameters (coefficients)"
    )

    # Load and display model
    try:
        model = load_polynomial_model(degree)
        st.session_state.current_model = model
        st.session_state.model_loaded = True

        # Show model preview
        render_model_preview(model, "polynomial", polynomial_degree=degree)

    except Exception as e:
        st.error(f"Error generating polynomial: {e}")
        st.session_state.model_loaded = False


def render_custom_model_editor() -> None:
    """Render the custom model code editor."""
    state = get_session_state()

    st.subheader("Custom Model")

    # Tabs for code entry vs file upload
    tab1, tab2 = st.tabs(["Code Editor", "Upload File"])

    with tab1:
        render_custom_code_editor(state)

    with tab2:
        render_custom_file_upload(state)


def render_custom_code_editor(state: SessionState) -> None:
    """Render the code editor tab for custom models."""
    st.caption("Enter your model function using jax.numpy for JIT compatibility")

    # Get current code from state or session
    current_code = st.session_state.custom_code

    # Code editor
    updated_code = st.text_area(
        "Model Code",
        value=current_code,
        height=300,
        key="custom_code_input",
        help="Define your model function. First parameter should be x (independent variable).",
    )

    st.session_state.custom_code = updated_code
    state.custom_code = updated_code

    # Validate syntax
    if updated_code.strip():
        is_valid, error = validate_code_syntax(updated_code)

        if is_valid:
            st.success("Syntax is valid")

            # List functions found
            functions = list_functions_in_module(updated_code)

            if functions:
                # Function selector
                selected_func = st.selectbox(
                    "Select Function",
                    options=functions,
                    index=functions.index(st.session_state.custom_function_name)
                    if st.session_state.custom_function_name in functions
                    else 0,
                    help="Choose the function to use as the model",
                )

                st.session_state.custom_function_name = selected_func
                state.custom_function_name = selected_func

                # Load and validate model
                if st.button("Load Model", type="primary"):
                    try:
                        model = load_custom_model(updated_code, selected_func)
                        if model is not None:
                            st.session_state.current_model = model
                            st.session_state.model_loaded = True
                            st.success(f"Model '{selected_func}' loaded successfully")

                            # Show model info
                            info = get_model_info(model)
                            st.info(
                                f"Parameters: {format_parameter_list(info['param_names'])} "
                                f"({info['param_count']} total)"
                            )
                        else:
                            st.error("Failed to load model")
                            st.session_state.model_loaded = False
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        st.session_state.model_loaded = False
            else:
                st.warning("No functions found in code")
        else:
            st.error(f"Syntax error: {error}")


def render_custom_file_upload(state: SessionState) -> None:
    """Render the file upload tab for custom models."""
    st.caption("Upload a Python file containing your model function")

    uploaded_file = st.file_uploader(
        "Upload Model File",
        type=["py"],
        key="model_file_upload",
        help="Upload a .py file with your model function definition",
    )

    if uploaded_file is not None:
        try:
            # Read file content
            content = uploaded_file.read().decode("utf-8")

            # Validate syntax
            is_valid, error = validate_code_syntax(content)

            if not is_valid:
                st.error(f"File contains syntax errors: {error}")
                return

            # List functions
            functions = list_functions_in_module(content)

            if not functions:
                st.warning("No functions found in uploaded file")
                return

            st.success(f"File loaded: {uploaded_file.name}")

            # Function selector
            selected_func = st.selectbox(
                "Select Function",
                options=functions,
                key="uploaded_file_function",
                help="Choose the function to use as the model",
            )

            # Show function preview
            st.code(f"Selected: {selected_func}", language="python")

            # Load button
            if st.button("Use This Model", type="primary", key="load_uploaded"):
                try:
                    model = load_custom_model(content, selected_func)
                    if model is not None:
                        st.session_state.current_model = model
                        st.session_state.model_loaded = True
                        state.custom_code = content
                        state.custom_function_name = selected_func
                        state.custom_file_path = uploaded_file.name
                        st.success(
                            f"Model '{selected_func}' loaded from {uploaded_file.name}"
                        )

                        # Show model info
                        info = get_model_info(model)
                        st.info(
                            f"Parameters: {format_parameter_list(info['param_names'])} "
                            f"({info['param_count']} total)"
                        )
                    else:
                        st.error("Failed to load model from file")
                except Exception as e:
                    st.error(f"Error loading model: {e}")

        except Exception as e:
            st.error(f"Error reading file: {e}")


def render_model_summary_sidebar() -> None:
    """Render model summary in sidebar."""
    if not st.session_state.get("model_loaded", False):
        st.sidebar.info("No model selected")
        return

    model = st.session_state.get("current_model")
    if model is None:
        return

    state = get_session_state()

    st.sidebar.subheader("Current Model")

    if state.model_type == "builtin":
        st.sidebar.markdown("**Type:** Built-in")
        st.sidebar.markdown(f"**Name:** {state.model_name}")
    elif state.model_type == "polynomial":
        st.sidebar.markdown("**Type:** Polynomial")
        st.sidebar.markdown(f"**Degree:** {state.polynomial_degree}")
    else:
        st.sidebar.markdown("**Type:** Custom")
        st.sidebar.markdown(f"**Function:** {state.custom_function_name}")

    summary = get_model_summary(model)
    st.sidebar.markdown(f"**Parameters:** {summary['param_count']}")

    if summary["has_auto_p0"]:
        st.sidebar.success("Auto p0 available")
    if summary["has_auto_bounds"]:
        st.sidebar.success("Auto bounds available")


# =============================================================================
# Main Page
# =============================================================================


def main() -> None:
    """Main entry point for the model selection page."""
    apply_dark_theme_css()
    init_model_page_state()

    st.title("Model Selection")
    st.caption("Choose the model function for curve fitting")

    # Model type selector
    model_type = render_model_type_selector()

    st.divider()

    # Render appropriate selector based on type
    if model_type == "Built-in":
        render_builtin_model_selector()
    elif model_type == "Polynomial":
        render_polynomial_selector()
    else:  # Custom
        render_custom_model_editor()

    # Sidebar summary
    render_model_summary_sidebar()

    # Navigation hint
    st.divider()
    state = get_session_state()

    if st.session_state.get("model_loaded", False):
        st.success("Model is ready. Proceed to Fitting Options.")
    else:
        st.info("Select and load a model to continue")


if __name__ == "__main__":
    main()
