"""Model file validation for security.

This module provides security validation for custom model files loaded
through the NLSQ CLI. It inspects model files for dangerous patterns
that could lead to arbitrary code execution.

Security Features
-----------------
- AST-based pattern detection for dangerous operations
- Path traversal prevention for file operations
- Resource limits (timeout, memory) for model execution
- Audit logging for model loading attempts

Dangerous Patterns Blocked
--------------------------
- Code execution: exec, eval, compile, __import__
- System access: os.system, subprocess, popen
- File modification: open with write mode
- Network access: socket connections
- Memory manipulation: ctypes operations
"""

import ast
import logging
import os
import resource
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("nlsq.cli.security")

# Dangerous patterns that trigger blocking
# These patterns indicate operations that could lead to arbitrary code execution
DANGEROUS_PATTERNS: frozenset[str] = frozenset(
    {
        # Code execution
        "exec",
        "eval",
        "compile",
        "__import__",
        # System access
        "system",
        "popen",
        "spawn",
        "call",
        "run",
        "Popen",
        # Network access
        "socket",
        "urlopen",
        "request",
        # File operations (write mode detection handled separately)
        # Memory manipulation
        "ctypes",
        "cffi",
        # Module manipulation
        "importlib",
        "__loader__",
        "__spec__",
    }
)

# Dangerous module prefixes
DANGEROUS_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "http",
        "ftplib",
        "telnetlib",
        "smtplib",
        "ctypes",
        "cffi",
        "multiprocessing",
        "concurrent",
    }
)


@dataclass
class ModelValidationResult:
    """Result of model file validation.

    Attributes
    ----------
    path : Path
        Path to the validated model file.
    is_valid : bool
        True if the model passed all security checks.
    is_trusted : bool
        True if the model was loaded with explicit trust flag.
    violations : list[str]
        List of security violations found in the model.
    signature : str | None
        Optional cryptographic signature of the model file.
    """

    path: Path
    is_valid: bool
    is_trusted: bool
    violations: list[str]
    signature: str | None = None


class DangerousPatternVisitor(ast.NodeVisitor):
    """AST visitor that detects dangerous patterns in Python code."""

    def __init__(self):
        self.violations: list[str] = []

    def visit_Name(self, node: ast.Name) -> Any:
        """Check for dangerous name references."""
        if node.id in DANGEROUS_PATTERNS:
            self.violations.append(f"Dangerous name reference: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """Check for dangerous function calls."""
        # Check direct function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in DANGEROUS_PATTERNS:
                self.violations.append(f"Dangerous function call: {node.func.id}()")
            # Check for open() with write modes
            if node.func.id == "open":
                self._check_open_call(node)
        # Check attribute calls like os.system()
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in DANGEROUS_PATTERNS:
                self.violations.append(f"Dangerous method call: .{node.func.attr}()")
            # Check for open() with write modes (e.g., builtins.open)
            if node.func.attr == "open":
                self._check_open_call(node)

        self.generic_visit(node)

    def _check_open_call(self, node: ast.Call) -> None:
        """Check if open() is called with write mode."""
        # Check positional mode argument
        if len(node.args) >= 2:
            mode_arg = node.args[1]
            if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                if any(c in mode_arg.value for c in "wax"):
                    self.violations.append(
                        f"File write operation: open(..., '{mode_arg.value}')"
                    )
        # Check keyword mode argument
        for keyword in node.keywords:
            if keyword.arg == "mode":
                if isinstance(keyword.value, ast.Constant) and isinstance(
                    keyword.value.value, str
                ):
                    if any(c in keyword.value.value for c in "wax"):
                        self.violations.append(
                            f"File write operation: open(..., mode='{keyword.value.value}')"
                        )

    def visit_Import(self, node: ast.Import) -> Any:
        """Check for dangerous module imports."""
        for alias in node.names:
            module_root = alias.name.split(".")[0]
            if module_root in DANGEROUS_MODULES:
                self.violations.append(f"Dangerous import: import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Check for dangerous from...import statements."""
        if node.module:
            module_root = node.module.split(".")[0]
            if module_root in DANGEROUS_MODULES:
                self.violations.append(
                    f"Dangerous import: from {node.module} import ..."
                )
        self.generic_visit(node)


def validate_model(path: Path, trusted: bool = False) -> ModelValidationResult:
    """Validate a model file for security.

    Performs AST-based static analysis to detect dangerous patterns
    that could lead to arbitrary code execution.

    Parameters
    ----------
    path : Path
        Path to the model file to validate.
    trusted : bool, default=False
        If True, skip validation (user explicitly trusts the model).

    Returns
    -------
    ModelValidationResult
        Validation result with is_valid, violations, etc.

    Examples
    --------
    >>> result = validate_model(Path("model.py"))
    >>> if not result.is_valid:
    ...     print(f"Validation failed: {result.violations}")
    """
    violations: list[str] = []

    # Check file exists
    if not path.exists():
        return ModelValidationResult(
            path=path,
            is_valid=False,
            is_trusted=trusted,
            violations=["File does not exist"],
        )

    # Check file extension
    if path.suffix != ".py":
        violations.append(f"Unexpected file extension: {path.suffix}")

    # Parse and analyze AST
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return ModelValidationResult(
            path=path,
            is_valid=False,
            is_trusted=trusted,
            violations=[f"Syntax error: {e}"],
        )

    # Visit all nodes to find violations
    visitor = DangerousPatternVisitor()
    visitor.visit(tree)
    violations.extend(visitor.violations)

    return ModelValidationResult(
        path=path,
        is_valid=len(violations) == 0,
        is_trusted=trusted,
        violations=violations,
    )


def validate_path(path: Path, base_dir: Path | None = None) -> bool:
    """Validate path for traversal attacks.

    Prevents path traversal attacks by ensuring:
    1. Absolute paths: Must exist (user explicitly provided the path)
    2. Relative paths: Must stay within the base directory

    This allows users to explicitly specify absolute paths to model files
    they trust, while preventing relative path traversal attacks like
    `../../../etc/passwd`.

    Parameters
    ----------
    path : Path
        Path to validate.
    base_dir : Path | None, default=None
        Base directory that relative paths must stay within. If None, uses
        current working directory. Only applies to relative paths.

    Returns
    -------
    bool
        True if path is safe, False if path traversal detected.

    Examples
    --------
    >>> validate_path(Path("models/my_model.py"))
    True
    >>> validate_path(Path("../../../etc/passwd"))
    False
    >>> validate_path(Path("/tmp/my_model.py"))  # Absolute paths are allowed
    True
    """
    try:
        # Resolve the path to its canonical form
        resolved_path = path.resolve()

        # If the original path is absolute, allow it if it exists
        # (user explicitly provided an absolute path)
        if path.is_absolute():
            return resolved_path.exists()

        # For relative paths, ensure they stay within base_dir
        if base_dir is None:
            base_dir = Path.cwd()
        resolved_base = base_dir.resolve()

        # Check if the resolved path is within the base directory
        return resolved_path.is_relative_to(resolved_base)
    except (ValueError, OSError):
        return False


class ResourceLimitError(Exception):
    """Raised when a resource limit is exceeded during model execution."""


@contextmanager
def resource_limits(timeout: float = 10.0, memory_mb: int = 512):
    """Context manager for resource-limited execution.

    Provides timeout and memory limits for executing potentially
    untrusted model code.

    Parameters
    ----------
    timeout : float, default=10.0
        Maximum execution time in seconds.
    memory_mb : int, default=512
        Maximum memory usage in megabytes.

    Yields
    ------
    None

    Raises
    ------
    ResourceLimitError
        If timeout or memory limit is exceeded.

    Examples
    --------
    >>> with resource_limits(timeout=5.0, memory_mb=256):
    ...     # Execute potentially slow/memory-intensive code
    ...     result = execute_model(model, data)
    """
    # Store original limits
    original_mem = resource.getrlimit(resource.RLIMIT_AS)

    # Timer for timeout
    timer = None
    timeout_occurred = False

    def timeout_handler():
        nonlocal timeout_occurred
        timeout_occurred = True
        # Send signal to main thread
        os.kill(os.getpid(), signal.SIGALRM)

    def signal_handler(signum, frame):
        if timeout_occurred:
            raise ResourceLimitError(f"Execution timeout ({timeout}s exceeded)")

    try:
        # Set memory limit
        memory_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()

        yield

    except MemoryError as err:
        raise ResourceLimitError(f"Memory limit ({memory_mb}MB) exceeded") from err
    finally:
        # Cancel timer
        if timer is not None:
            timer.cancel()

        # Restore original limits
        resource.setrlimit(resource.RLIMIT_AS, original_mem)

        # Restore original signal handler
        signal.signal(signal.SIGALRM, old_handler)


class AuditLogger:
    """Audit logger for model loading attempts.

    Logs all model loading attempts with validation results, user
    identity, and timestamp for security auditing.

    Attributes
    ----------
    log_file : Path
        Path to the audit log file.
    max_size_bytes : int
        Maximum log file size before rotation (default: 10MB).
    retention_days : int
        Number of days to retain log files (default: 90).
    """

    def __init__(
        self,
        log_file: Path | None = None,
        max_size_bytes: int = 10 * 1024 * 1024,  # 10MB
        retention_days: int = 90,
    ):
        if log_file is None:
            # Default to ~/.nlsq/audit.log
            log_file = Path.home() / ".nlsq" / "audit.log"

        self.log_file = log_file
        self.max_size_bytes = max_size_bytes
        self.retention_days = retention_days

        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logger()

    def _setup_logger(self):
        """Set up the audit logger with rotation."""
        from logging.handlers import RotatingFileHandler

        self._logger = logging.getLogger("nlsq.audit")
        self._logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self._logger.handlers:
            handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_size_bytes,
                backupCount=self.retention_days,
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def log_load_attempt(
        self,
        path: Path,
        result: ModelValidationResult,
        user: str | None = None,
    ):
        """Log a model loading attempt.

        Parameters
        ----------
        path : Path
            Path to the model file.
        result : ModelValidationResult
            Result of model validation.
        user : str | None
            Username attempting the load (default: current user).
        """
        if user is None:
            user = os.getenv("USER", "unknown")

        status = "ALLOWED" if result.is_valid or result.is_trusted else "BLOCKED"
        trust_flag = " (trusted)" if result.is_trusted else ""

        message = f"{status}{trust_flag} - User: {user} - Path: {path}"
        if result.violations:
            message += f" - Violations: {', '.join(result.violations[:3])}"

        if status == "BLOCKED":
            self._logger.warning(message)
        else:
            self._logger.info(message)


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance.

    Returns
    -------
    AuditLogger
        The global audit logger.
    """
    global _audit_logger  # noqa: PLW0603 - singleton pattern
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
