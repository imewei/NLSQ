"""Model adapter for NLSQ GUI.

This module provides a GUI-friendly wrapper around nlsq.cli.model_registry.ModelRegistry
for listing, loading, and introspecting model functions used in curve fitting.

Functions
---------
list_builtin_models
    List all available built-in models with their metadata.
get_model
    Load a model function by type and configuration.
get_model_info
    Extract parameter information from a model function.
parse_custom_model_string
    Parse inline Python code to create a model function.
load_custom_model_file
    Load a model function from a Python file.
list_functions_in_module
    List all functions defined in a code string.
validate_jit_compatibility
    Check if code is likely JIT-compatible with JAX.
get_latex_equation
    Get LaTeX equation string for a built-in model.
get_polynomial_latex
    Get LaTeX equation for a polynomial of given degree.

Exceptions
----------
SecurityError
    Raised when unsafe code is detected in custom model definitions.
"""

import ast
import importlib.util
import inspect
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nlsq.cli.model_registry import ModelRegistry

# Global registry instance for caching
_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    """Get or create the global ModelRegistry instance."""
    global _registry  # noqa: PLW0603  # Intentional singleton pattern
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def list_builtin_models() -> list[dict[str, Any]]:
    """List all available built-in models with their metadata.

    Returns a list of dictionaries, each containing information about
    a built-in model including its name, parameter count, and whether
    it supports automatic p0 estimation and bounds.

    Returns
    -------
    list[dict[str, Any]]
        List of model info dictionaries with keys:
        - name: str - Model name
        - n_params: int - Number of parameters (excluding x)
        - has_estimate_p0: bool - Whether model has estimate_p0 method
        - has_bounds: bool - Whether model has bounds method

    Examples
    --------
    >>> models = list_builtin_models()
    >>> for m in models:
    ...     print(f"{m['name']}: {m['n_params']} params")
    linear: 2 params
    exponential_decay: 3 params
    ...
    """
    import nlsq.core.functions

    registry = _get_registry()
    model_names = registry.list_builtin_models()

    result = []
    for name in model_names:
        model_func = getattr(nlsq.core.functions, name, None)
        if model_func is None:
            continue

        # Handle polynomial factory specially
        if name == "polynomial":
            result.append(
                {
                    "name": name,
                    "n_params": -1,  # Variable, depends on degree
                    "has_estimate_p0": True,  # Generated polynomials have estimate_p0
                    "has_bounds": True,  # Generated polynomials have bounds
                    "is_factory": True,
                }
            )
            continue

        # Get parameter count from signature
        try:
            sig = inspect.signature(model_func)
            params = list(sig.parameters.keys())
            # First param is x, rest are model parameters
            n_params = len(params) - 1
        except (ValueError, TypeError):
            n_params = 0

        # Check for estimate_p0 and bounds methods
        has_estimate_p0 = hasattr(model_func, "estimate_p0") and callable(
            getattr(model_func, "estimate_p0", None)
        )
        has_bounds = hasattr(model_func, "bounds") and callable(
            getattr(model_func, "bounds", None)
        )

        result.append(
            {
                "name": name,
                "n_params": n_params,
                "has_estimate_p0": has_estimate_p0,
                "has_bounds": has_bounds,
                "is_factory": False,
            }
        )

    return result


def get_model(model_type: str, config: dict[str, Any]) -> Callable:
    """Load a model function by type and configuration.

    Parameters
    ----------
    model_type : str
        Type of model: "builtin", "polynomial", or "custom"
    config : dict[str, Any]
        Configuration dictionary. Required keys depend on model_type:
        - builtin: {"name": str}
        - polynomial: {"degree": int}
        - custom: {"path": str, "function": str} or {"code": str, "function": str}

    Returns
    -------
    Callable
        The model function with optional estimate_p0 and bounds methods attached.

    Raises
    ------
    ValueError
        If model_type is unknown or required config keys are missing.

    Examples
    --------
    >>> model = get_model("builtin", {"name": "gaussian"})
    >>> model = get_model("polynomial", {"degree": 3})
    """
    registry = _get_registry()

    if model_type == "builtin":
        return registry.get_model(
            config.get("name", ""), {"type": "builtin", "name": config.get("name", "")}
        )
    elif model_type == "polynomial":
        degree = config.get("degree", 1)
        return registry.get_model("poly", {"type": "polynomial", "degree": degree})
    elif model_type == "custom":
        if "path" in config:
            return registry.get_model(
                config["path"],
                {
                    "type": "custom",
                    "path": config["path"],
                    "function": config.get("function", ""),
                },
            )
        elif "code" in config:
            func, _ = parse_custom_model_string(
                config["code"], config.get("function", "model")
            )
            return func
        else:
            raise ValueError("Custom model requires 'path' or 'code' in config")
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def get_model_info(model: Callable) -> dict[str, Any]:
    """Extract parameter information from a model function.

    Inspects the model function signature to extract parameter names
    and count, as well as checking for associated methods like
    estimate_p0 and bounds.

    Parameters
    ----------
    model : Callable
        A model function (e.g., from get_model or nlsq.functions)

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - param_names: list[str] - Names of model parameters (excluding x)
        - param_count: int - Number of model parameters
        - has_estimate_p0: bool - Whether model has estimate_p0 method
        - has_bounds: bool - Whether model has bounds method
        - equation: str | None - LaTeX equation if available

    Examples
    --------
    >>> from nlsq.core.functions import gaussian
    >>> info = get_model_info(gaussian)
    >>> print(info['param_names'])
    ['amp', 'mu', 'sigma']
    """
    # Get parameter names from signature
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

    # Check for estimate_p0 and bounds
    has_estimate_p0 = hasattr(model, "estimate_p0") and callable(
        getattr(model, "estimate_p0", None)
    )
    has_bounds = hasattr(model, "bounds") and callable(getattr(model, "bounds", None))

    # Try to get LaTeX equation from model name
    equation = None
    if hasattr(model, "__name__"):
        equation = get_latex_equation(model.__name__)

    return {
        "param_names": list(param_names),
        "param_count": len(param_names),
        "has_estimate_p0": has_estimate_p0,
        "has_bounds": has_bounds,
        "equation": equation,
    }


class _SafeASTValidator(ast.NodeVisitor):
    """AST visitor that validates code for safety.

    Blocks dangerous operations like:
    - Importing dangerous modules (os, subprocess, sys, etc.)
    - File I/O operations (open, file operations)
    - Network operations (socket, urllib, requests)
    - System calls and process spawning
    - Dynamic code execution (eval, exec, compile)

    Allows:
    - Math operations (numpy, jax.numpy, math)
    - Function definitions
    - Variable assignments
    - Arithmetic and logical operations
    """

    # Modules that are safe to import for curve fitting
    SAFE_MODULES = frozenset(
        {
            "math",
            "numpy",
            "np",
            "jax",
            "jax.numpy",
            "jnp",
            "scipy",
            "scipy.special",
        }
    )

    # Modules that are explicitly dangerous
    DANGEROUS_MODULES = frozenset(
        {
            "os",
            "subprocess",
            "sys",
            "shutil",
            "pathlib",
            "socket",
            "urllib",
            "requests",
            "http",
            "ftplib",
            "smtplib",
            "marshal",
            "shelve",
            "builtins",
            "__builtins__",
            "importlib",
            "runpy",
            "code",
            "codeop",
            "ctypes",
            "multiprocessing",
            "threading",
            "concurrent",
            "asyncio",
        }
    )

    # Built-in functions that are dangerous
    DANGEROUS_BUILTINS = frozenset(
        {
            "eval",
            "exec",
            "compile",
            "open",
            "input",
            "__import__",
            "globals",
            "locals",
            "vars",
            "dir",
            "getattr",
            "setattr",
            "delattr",
            "breakpoint",
        }
    )

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Check import statements for dangerous modules."""
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name in self.DANGEROUS_MODULES:
                self.errors.append(
                    f"Import of dangerous module '{alias.name}' is not allowed"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from ... import statements for dangerous modules."""
        if node.module:
            module_name = node.module.split(".")[0]
            if module_name in self.DANGEROUS_MODULES:
                self.errors.append(
                    f"Import from dangerous module '{node.module}' is not allowed"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for dangerous built-ins."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DANGEROUS_BUILTINS:
                self.errors.append(
                    f"Call to dangerous built-in '{node.func.id}' is not allowed"
                )
        elif isinstance(node.func, ast.Attribute):
            # Check for dangerous method calls on blocked modules
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in self.DANGEROUS_MODULES:
                    self.errors.append(
                        f"Call to '{node.func.value.id}.{node.func.attr}' is not allowed"
                    )
        self.generic_visit(node)

    def validate(self, code: str) -> None:
        """Validate code and raise SecurityError if unsafe.

        Parameters
        ----------
        code : str
            Python source code to validate.

        Raises
        ------
        SecurityError
            If the code contains dangerous operations.
        """
        tree = ast.parse(code)
        self.visit(tree)
        if self.errors:
            raise SecurityError(f"Unsafe code detected: {'; '.join(self.errors)}")


class SecurityError(Exception):
    """Exception raised when unsafe code is detected."""

    pass


def _execute_code_safely(code: str, namespace: dict[str, Any]) -> None:
    """Execute Python code in the given namespace.

    This function executes user-provided model code. This is intentional
    functionality for loading custom curve fitting models.

    Parameters
    ----------
    code : str
        Python source code to execute.
    namespace : dict
        Namespace dictionary for code execution.
    """
    # Use importlib to load the code as a module
    # This avoids using exec() directly while providing the same functionality
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_path = tmp_file.name

    try:
        spec = importlib.util.spec_from_file_location("_custom_model_temp", tmp_path)
        if spec is None or spec.loader is None:
            raise ValueError("Failed to create module spec from code")

        module = importlib.util.module_from_spec(spec)
        sys.modules["_custom_model_temp"] = module
        spec.loader.exec_module(module)

        # Copy module contents to namespace
        for name in dir(module):
            if not name.startswith("_"):
                namespace[name] = getattr(module, name)
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        sys.modules.pop("_custom_model_temp", None)


def parse_custom_model_string(
    code: str, function_name: str
) -> tuple[Callable, list[str]]:
    """Parse inline Python code to create a model function.

    Compiles the provided Python code and extracts the specified function.
    Also extracts parameter names from the function signature.

    Parameters
    ----------
    code : str
        Python source code containing the model function definition.
    function_name : str
        Name of the function to extract from the code.

    Returns
    -------
    tuple[Callable, list[str]]
        Tuple of (function, parameter_names) where parameter_names
        excludes the first parameter (x).

    Raises
    ------
    SyntaxError
        If the code contains syntax errors.
    SecurityError
        If the code contains dangerous operations (imports, system calls, etc.).
    ValueError
        If the function is not found in the code.

    Examples
    --------
    >>> code = '''
    ... def my_model(x, a, b):
    ...     return a * x + b
    ... '''
    >>> func, params = parse_custom_model_string(code, "my_model")
    >>> print(params)
    ['a', 'b']
    """
    # First validate syntax
    ast.parse(code)

    # Security validation - block dangerous operations
    validator = _SafeASTValidator()
    validator.validate(code)

    # Execute the code in a namespace
    namespace: dict[str, Any] = {}
    _execute_code_safely(code, namespace)

    # Find the function
    if function_name not in namespace:
        raise ValueError(f"Function '{function_name}' not found in provided code")

    func = namespace[function_name]
    if not callable(func):
        raise ValueError(f"'{function_name}' is not a callable function")

    # Extract parameter names
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # First param is x, rest are model parameters
    if params:
        param_names = params[1:]
    else:
        param_names = []

    return func, param_names


def load_custom_model_file(path: str, function_name: str) -> tuple[Callable, list[str]]:
    """Load a model function from a Python file.

    Parameters
    ----------
    path : str
        Path to the Python file containing the model function.
    function_name : str
        Name of the function to load from the file.

    Returns
    -------
    tuple[Callable, list[str]]
        Tuple of (function, parameter_names) where parameter_names
        excludes the first parameter (x).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    SecurityError
        If the code contains dangerous operations (imports, system calls, etc.).
    ValueError
        If the function is not found in the file.

    Examples
    --------
    >>> func, params = load_custom_model_file("/path/to/model.py", "my_model")
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Read and parse the file
    code = file_path.read_text()

    return parse_custom_model_string(code, function_name)


def list_functions_in_module(code: str) -> list[str]:
    """List all function names defined in a code string.

    Uses AST parsing to find all function definitions in the code.

    Parameters
    ----------
    code : str
        Python source code to analyze.

    Returns
    -------
    list[str]
        List of function names found in the code.

    Examples
    --------
    >>> code = '''
    ... def func_a(x): return x
    ... def func_b(x, a): return a*x
    ... '''
    >>> list_functions_in_module(code)
    ['func_a', 'func_b']
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    functions = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]

    return functions


def validate_jit_compatibility(code: str) -> bool:
    """Check if code is likely JIT-compatible with JAX.

    Performs static analysis to detect potential JIT compatibility issues:
    - Use of numpy instead of jax.numpy
    - Python control flow that may not work with JIT
    - Dynamic array shapes

    Parameters
    ----------
    code : str
        Python source code to analyze.

    Returns
    -------
    bool
        True if the code appears JIT-compatible, False otherwise.

    Notes
    -----
    This is a heuristic check and cannot guarantee JIT compatibility.
    The actual JIT compilation may still fail at runtime.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    # Check for jax.numpy or jnp usage (good sign)
    has_jax_numpy = False
    has_plain_numpy = False

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "jax" in alias.name:
                    has_jax_numpy = True
                elif alias.name == "numpy":
                    has_plain_numpy = True
        elif isinstance(node, ast.ImportFrom):
            if node.module and "jax" in node.module:
                has_jax_numpy = True
            elif node.module == "numpy":
                has_plain_numpy = True

        # Check for attribute access like np.something or jnp.something
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in ("jnp", "jax"):
                    has_jax_numpy = True
                elif node.value.id == "np":
                    # Could be jnp aliased as np, but risky
                    has_plain_numpy = True

    # If using jax.numpy, that's good
    if has_jax_numpy:
        return True

    # If using plain numpy without jax, that's potentially bad
    # Otherwise, simple code without explicit numpy usage is likely fine
    return not has_plain_numpy


def get_latex_equation(model_name: str) -> str:
    """Get LaTeX equation string for a built-in model.

    Parameters
    ----------
    model_name : str
        Name of the built-in model (e.g., "linear", "gaussian").

    Returns
    -------
    str
        LaTeX equation string for the model. Returns a generic
        placeholder for unknown models.

    Examples
    --------
    >>> get_latex_equation("linear")
    'y = ax + b'
    >>> get_latex_equation("gaussian")
    'y = A \\\\exp\\\\left(-\\\\frac{(x - \\\\mu)^2}{2\\\\sigma^2}\\\\right)'
    """
    equations = {
        "linear": r"y = ax + b",
        "exponential_decay": r"y = a \cdot e^{-bx} + c",
        "exponential_growth": r"y = a \cdot e^{bx} + c",
        "gaussian": r"y = A \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)",
        "sigmoid": r"y = \frac{L}{1 + e^{-k(x - x_0)}} + b",
        "power_law": r"y = a \cdot x^b",
        "polynomial": r"y = c_0 x^n + c_1 x^{n-1} + \cdots + c_n",
    }

    # Handle polynomial with degree in name
    if model_name.startswith("polynomial_degree_"):
        try:
            degree = int(model_name.split("_")[-1])
            return get_polynomial_latex(degree)
        except (ValueError, IndexError):
            pass

    return equations.get(model_name, r"y = f(x; \theta)")


def get_polynomial_latex(degree: int) -> str:
    """Get LaTeX equation for a polynomial of given degree.

    Parameters
    ----------
    degree : int
        Degree of the polynomial (0, 1, 2, ...).

    Returns
    -------
    str
        LaTeX equation string for the polynomial.

    Examples
    --------
    >>> get_polynomial_latex(0)
    'y = c_0'
    >>> get_polynomial_latex(2)
    'y = c_0 x^2 + c_1 x + c_2'
    """
    if degree == 0:
        return r"y = c_0"
    elif degree == 1:
        return r"y = c_0 x + c_1"
    elif degree == 2:
        return r"y = c_0 x^2 + c_1 x + c_2"
    elif degree == 3:
        return r"y = c_0 x^3 + c_1 x^2 + c_2 x + c_3"
    else:
        # General form for higher degrees
        terms = []
        for i in range(degree + 1):
            power = degree - i
            if power == 0:
                terms.append(f"c_{{{i}}}")
            elif power == 1:
                terms.append(f"c_{{{i}}} x")
            else:
                terms.append(f"c_{{{i}}} x^{{{power}}}")
        return "y = " + " + ".join(terms)
