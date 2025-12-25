"""Code editor component for custom model entry in NLSQ GUI.

This module provides functions for rendering a code editor for custom model
functions, including syntax validation and template generation.

Functions
---------
get_default_model_template
    Get a default template for custom model functions.
validate_code_syntax
    Validate Python code syntax.
render_code_editor
    Render the code editor UI with Streamlit.
render_file_upload
    Render file upload for external .py files.
"""

import ast
from typing import Any

from nlsq.gui.adapters.model_adapter import (
    list_functions_in_module,
    validate_jit_compatibility,
)
from nlsq.gui.utils.theme import get_code_editor_theme, get_current_theme


def get_default_model_template() -> str:
    """Get a default template for custom model functions.

    Returns a template Python code string with a basic model function
    that users can modify for their custom fitting needs.

    Returns
    -------
    str
        A Python code template with a model function definition.

    Examples
    --------
    >>> template = get_default_model_template()
    >>> "def model" in template
    True
    """
    return '''import jax.numpy as jnp

def model(x, a, b, c):
    """Custom model function.

    Parameters
    ----------
    x : array-like
        Independent variable (input data).
    a, b, c : float
        Model parameters to be fitted.

    Returns
    -------
    array-like
        Model prediction y = f(x; a, b, c).
    """
    return a * jnp.exp(-b * x) + c
'''


def validate_code_syntax(code: str) -> tuple[bool, str]:
    """Validate Python code syntax.

    Parses the provided code string to check for syntax errors.
    Does not execute the code.

    Parameters
    ----------
    code : str
        Python source code to validate.

    Returns
    -------
    tuple[bool, str]
        Tuple of (is_valid, error_message).
        is_valid is True if syntax is correct, False otherwise.
        error_message contains the error description if invalid, empty string if valid.

    Examples
    --------
    >>> is_valid, error = validate_code_syntax("def f(x): return x")
    >>> is_valid
    True

    >>> is_valid, error = validate_code_syntax("def broken(x")
    >>> is_valid
    False
    """
    if not code.strip():
        return False, "Code is empty"

    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        return False, error_msg


def get_code_validation_status(code: str) -> dict[str, Any]:
    """Get comprehensive validation status for custom model code.

    Validates syntax and checks for JAX compatibility.

    Parameters
    ----------
    code : str
        Python source code to validate.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - is_valid: bool - Whether syntax is valid
        - error_message: str - Error message if invalid
        - is_jit_compatible: bool - Whether code appears JIT-compatible
        - functions: list[str] - List of function names found
        - warnings: list[str] - Non-fatal warnings

    Examples
    --------
    >>> status = get_code_validation_status("def f(x, a): return a * x")
    >>> status["is_valid"]
    True
    """
    result: dict[str, Any] = {
        "is_valid": False,
        "error_message": "",
        "is_jit_compatible": False,
        "functions": [],
        "warnings": [],
    }

    # Check syntax
    is_valid, error_msg = validate_code_syntax(code)
    result["is_valid"] = is_valid
    result["error_message"] = error_msg

    if not is_valid:
        return result

    # List functions in code
    result["functions"] = list_functions_in_module(code)

    if not result["functions"]:
        result["warnings"].append("No function definitions found in code")

    # Check JIT compatibility
    result["is_jit_compatible"] = validate_jit_compatibility(code)

    if not result["is_jit_compatible"]:
        result["warnings"].append(
            "Code may not be JIT-compatible. Consider using jax.numpy instead of numpy."
        )

    return result


def render_code_editor(
    code: str,
    key: str = "code_editor",
    height: int = 300,
) -> str:
    """Render the code editor UI with Streamlit.

    Displays a text area for code input with syntax validation feedback.
    This function requires Streamlit to be running. The editor theme
    automatically adjusts based on the current app theme.

    Parameters
    ----------
    code : str
        Initial code to display in the editor.
    key : str, optional
        Unique Streamlit widget key. Default is "code_editor".
    height : int, optional
        Height of the text area in pixels. Default is 300.

    Returns
    -------
    str
        The current code content from the editor.

    Note
    ----
    This function requires Streamlit to be running. It will import
    streamlit when called.
    """
    import streamlit as st

    # Get current theme and corresponding editor theme
    current_theme = get_current_theme()
    editor_theme = get_code_editor_theme(current_theme)

    # Display editor theme info
    theme_label = "Dark" if current_theme == "dark" else "Light"

    # Code input area
    # Note: st.text_area doesn't support Monaco themes directly,
    # but we can style it with CSS based on theme
    updated_code = st.text_area(
        "Model Code",
        value=code,
        height=height,
        key=key,
        help=f"Enter your custom model function. Use jax.numpy for JIT compatibility. (Editor: {theme_label} theme)",
    )

    # Apply theme-specific styling to the text area
    if current_theme == "dark":
        st.markdown(
            """
            <style>
            [data-testid="stTextArea"] textarea {
                background-color: #1e1e1e !important;
                color: #d4d4d4 !important;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            [data-testid="stTextArea"] textarea {
                background-color: #ffffff !important;
                color: #1e1e1e !important;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Validate and show status
    if updated_code.strip():
        status = get_code_validation_status(updated_code)

        if status["is_valid"]:
            st.success("Syntax is valid")

            # Show functions found
            if status["functions"]:
                st.info(f"Functions found: {', '.join(status['functions'])}")

            # Show warnings
            for warning in status["warnings"]:
                st.warning(warning)
        else:
            st.error(status["error_message"])

    return updated_code


def render_file_upload(key: str = "model_file") -> tuple[str | None, str | None]:
    """Render file upload for external .py files.

    Displays a file uploader and function selector for loading custom
    model functions from external Python files.

    Parameters
    ----------
    key : str, optional
        Unique Streamlit widget key prefix. Default is "model_file".

    Returns
    -------
    tuple[str | None, str | None]
        Tuple of (file_path, function_name).
        Both are None if no file is uploaded or no function selected.

    Note
    ----
    This function requires Streamlit to be running. It will import
    streamlit when called.
    """
    import streamlit as st

    uploaded_file = st.file_uploader(
        "Upload Model File",
        type=["py"],
        key=f"{key}_upload",
        help="Upload a Python file containing your model function",
    )

    if uploaded_file is None:
        return None, None

    # Read file content
    try:
        content = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None

    # Validate syntax
    is_valid, error = validate_code_syntax(content)
    if not is_valid:
        st.error(f"File contains syntax errors: {error}")
        return None, None

    # List functions in file
    functions = list_functions_in_module(content)

    if not functions:
        st.warning("No functions found in uploaded file")
        return None, None

    # Function selector
    selected_function = st.selectbox(
        "Select Function",
        options=functions,
        key=f"{key}_function",
        help="Choose the model function from the uploaded file",
    )

    # Store file path info (use uploaded file name)
    # Note: For actual file loading, we'll use the content, not a path
    st.session_state[f"{key}_content"] = content

    return uploaded_file.name, selected_function


def get_uploaded_file_content(key: str = "model_file") -> str | None:
    """Get the content of an uploaded model file.

    Retrieves the file content stored in session state by render_file_upload.

    Parameters
    ----------
    key : str, optional
        The key prefix used when uploading. Default is "model_file".

    Returns
    -------
    str | None
        The file content, or None if no file was uploaded.
    """
    import streamlit as st

    return st.session_state.get(f"{key}_content")
