"""Model file validation for security.

This module provides security validation for custom model files loaded
through the NLSQ CLI. It inspects model files for dangerous patterns
that could lead to arbitrary code execution.

Security Features
-----------------
- AST-based pattern detection for dangerous operations
- Path traversal prevention for file operations
- Resource limits (timeout, memory) around module *load* time only --
  ``resource_limits()`` wraps ``exec_module()``, not the many repeated
  calls to the model function during ``curve_fit()``'s optimization loop.
  An AST-clean model whose function body blocks or leaks memory only at
  call time (e.g. an infinite loop with no dangerous names) is not caught
  by this or by the AST validator; that is a known gap, not enforced here.
- Audit logging for model loading attempts

Dangerous Patterns Blocked
--------------------------
- Code execution: exec, eval, compile, __import__
- System access: os.system, subprocess, popen
- File modification: open with write mode
- Network access: socket connections
- Memory manipulation: ctypes operations
- Serialization code execution: pickle, marshal, shelve, dill
"""

import ast
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Platform-specific imports for resource limiting
# These modules are Unix-only and not available on Windows
_HAS_RESOURCE_LIMITS = sys.platform != "win32"

if _HAS_RESOURCE_LIMITS:
    import resource
    import signal

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
        # Reflection / sandbox escape
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "globals",
        "locals",
        "vars",
        "type",
        "__builtins__",
        "__subclasses__",
        # Dunder attribute chain traversal (used in sandbox escape: ().__class__.__bases__...)
        "__class__",
        "__bases__",
        "__mro__",
        "__dict__",
        "__globals__",
        "__code__",
        # Attribute-access bypass routes not covered by __class__ chain
        "__getattribute__",
        "__getattr__",
        "__base__",
        "mro",
        # Frame introspection (non-dunder equivalents of __globals__/__code__
        # that reach the same objects via sys._getframe() / traceback frames /
        # generator frames)
        "_getframe",
        "f_globals",
        "f_back",
        "f_locals",
        "f_code",
        "f_builtins",
        "gi_frame",
        "tb_frame",
        # Coroutine/async-generator frame introspection -- same technique as
        # gi_frame/tb_frame above, reached via a coroutine or async generator
        # object (e.g. `c = coro(); c.cr_frame.f_builtins["exec"](...)`)
        # instead of a plain generator or traceback. Not covered by banning
        # `asyncio`: these are built-in coroutine-object attributes, not
        # asyncio symbols, so they're reachable without importing asyncio.
        "cr_frame",
        "ag_frame",
        # Interactive debugger (can execute arbitrary code interactively)
        "breakpoint",
        # sys.modules dict-access bypass: reaches already-imported os/subprocess
        # etc. without a literal `import os` this visitor would otherwise catch
        "modules",
        # String-based getattr equivalents (operator.attrgetter/methodcaller,
        # pydoc.locate, pkgutil.resolve_name) resolve dotted names at runtime,
        # bypassing the Attribute/Name checks above
        "attrgetter",
        "methodcaller",
        "locate",
        "resolve_name",
        # asyncio subprocess spawners (bypass the subprocess/Popen name check)
        "create_subprocess_shell",
        "create_subprocess_exec",
        # pathlib file-mutation methods (bypass the open()-write-mode check;
        # no open()/os/shutil call needed to delete, rename, or repermission
        # a file through a Path object)
        "write_text",
        "write_bytes",
        "unlink",
        "rmdir",
        "rename",
        "replace",
        "chmod",
        "touch",
        "symlink_to",
        "hardlink_to",
    },
)

# Dangerous module prefixes that trigger blocking on import
DANGEROUS_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
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
        # Sandbox-escape via aliased symbol import: `from importlib import
        # import_module as im; im("subprocess")` or `from builtins import
        # eval as ev` previously bypassed detection entirely because only
        # the module root (not the imported symbol) was checked.
        "importlib",
        "builtins",
        # Bytecode/function construction — bypasses exec/eval detection by
        # building and running arbitrary code objects directly
        "types",
        # Serialization modules that execute arbitrary code during deserialization
        "pickle",
        "marshal",
        "shelve",
        "dill",
        "cloudpickle",
        # Introspection / sandbox-escape helpers
        "inspect",
        "dis",
        "code",
        "codeop",
        # Alternate code-execution entry points (bypass the exec/eval name check)
        "runpy",
        # String-based getattr equivalents
        "operator",
        "pydoc",
        "pkgutil",
        # timeit.timeit()/asyncio.create_subprocess_*() execute arbitrary code
        # strings / spawn processes without tripping exec/subprocess checks
        "timeit",
        "asyncio",
    },
)

# Dunder substrings that are dangerous even inside a plain string constant,
# because str.format()/str.format_map() resolve dotted attribute chains at
# runtime from a format-spec string (e.g. "{0.__class__.__bases__}".format(x)),
# which never appears as an ast.Attribute node for the visitor above to catch.
_DANGEROUS_STRING_SUBSTRINGS: tuple[str, ...] = (
    "__class__",
    "__base__",
    "__mro__",
    "__subclasses__",
    "__globals__",
    "__builtins__",
    "__code__",
    "__import__",
    "__loader__",
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
            # str.format()/str.format_map() can resolve dunder attribute
            # chains at runtime from a format-spec string, defeating the
            # literal-substring scan in visit_Constant when the string is
            # built via concatenation. Block the call itself.
            if node.func.attr in ("format", "format_map"):
                self.violations.append(
                    f"Dangerous method call: .{node.func.attr}() "
                    "(format-string escape risk)",
                )

        self.generic_visit(node)

    def _check_open_call(self, node: ast.Call) -> None:
        """Check if open() is called with write mode."""
        # Check positional mode argument
        if len(node.args) >= 2:
            mode_arg = node.args[1]
            if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                if any(c in mode_arg.value for c in "wax+"):
                    self.violations.append(
                        f"File write operation: open(..., '{mode_arg.value}')",
                    )
        # Check keyword mode argument
        for keyword in node.keywords:
            if keyword.arg == "mode":
                if isinstance(keyword.value, ast.Constant) and isinstance(
                    keyword.value.value,
                    str,
                ):
                    if any(c in keyword.value.value for c in "wax+"):
                        self.violations.append(
                            f"File write operation: open(..., mode='{keyword.value.value}')",
                        )

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Check string constants for dunder chains used in format-string escapes.

        `"{0.__class__.__bases__}".format(x)` resolves the dotted chain at
        runtime from inside the string itself, never producing an ast.Attribute
        node, so it must be caught here instead of in visit_Attribute.
        """
        if isinstance(node.value, str):
            for pattern in _DANGEROUS_STRING_SUBSTRINGS:
                if pattern in node.value:
                    self.violations.append(
                        f"Dangerous string content (format-string escape risk): {pattern}",
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Check for dangerous attribute accesses, including uncalled dunder chains.

        visit_Call only catches .attr() calls; this catches .attr access as a
        value (e.g. x.__subclasses__ stored in a variable, then called via [0]()).
        """
        if node.attr in DANGEROUS_PATTERNS:
            self.violations.append(f"Dangerous attribute access: .{node.attr}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        """Check for dangerous module imports."""
        for alias in node.names:
            module_root = alias.name.split(".")[0]
            if module_root in DANGEROUS_MODULES or module_root in DANGEROUS_PATTERNS:
                self.violations.append(f"Dangerous import: import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Check for dangerous from...import statements.

        Checks both the source module and each imported symbol name, since
        `from importlib import import_module as im` or `from builtins import
        eval as ev` rebind the dangerous symbol to a clean-looking alias --
        checking the module alone lets the aliased name evade every later
        Name/Attribute/Call check in this visitor.
        """
        module_root = node.module.split(".")[0] if node.module else None
        if module_root and (
            module_root in DANGEROUS_MODULES or module_root in DANGEROUS_PATTERNS
        ):
            self.violations.append(
                f"Dangerous import: from {node.module} import ...",
            )
        for alias in node.names:
            if alias.name in DANGEROUS_PATTERNS or alias.name in DANGEROUS_MODULES:
                self.violations.append(
                    f"Dangerous import: from {node.module} import {alias.name}",
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

    Prevents path traversal attacks by ensuring all paths (both absolute
    and relative) resolve to locations within the base directory.

    Parameters
    ----------
    path : Path
        Path to validate.
    base_dir : Path | None, default=None
        Base directory that paths must stay within. If None, uses
        current working directory.

    Returns
    -------
    bool
        True if path is safe (within base_dir), False if path traversal detected.

    Examples
    --------
    >>> validate_path(Path("models/my_model.py"))
    True
    >>> validate_path(Path("../../../etc/passwd"))
    False
    >>> validate_path(Path("/etc/passwd"))  # Outside base_dir
    False
    """
    try:
        # Resolve the path to its canonical form
        resolved_path = path.resolve()

        # Set base directory (defaults to CWD)
        if base_dir is None:
            base_dir = Path.cwd()
        resolved_base = base_dir.resolve()

        # Check if the resolved path is within the base directory
        # This applies to both absolute and relative paths
        return resolved_path.is_relative_to(resolved_base)
    except (ValueError, OSError):
        return False


class ResourceLimitError(Exception):
    """Raised when a resource limit is exceeded during model execution."""


def _accelerator_backend_active() -> bool:
    """Return True if JAX is already initialized on a GPU/TPU backend.

    Used to decide whether enforcing an ``RLIMIT_AS`` (virtual address-space)
    memory cap is safe. CUDA and other accelerator runtimes reserve huge
    amounts of virtual address space, so an AS cap is both ineffective and
    crash-prone around accelerator code. We only probe an *already-imported*
    JAX module (``sys.modules``) so this check never forces JAX initialization
    or imports JAX as a hard dependency of the validator.
    """
    jax_mod = sys.modules.get("jax")
    if jax_mod is None:
        return False
    try:
        return jax_mod.default_backend() in ("gpu", "tpu")
    except Exception:
        # Any failure probing the backend -> conservatively report no
        # accelerator so the (safe) CPU memory cap path is taken.
        return False


def _current_virtual_memory_bytes() -> int | None:
    """Current virtual address-space size (VmSize) of this process, in bytes.

    Used as the baseline for the RLIMIT_AS cap in ``resource_limits`` so the
    cap reflects headroom above what the interpreter already holds, rather
    than an absolute ceiling that ignores it (a process that has already
    imported JAX/NumPy routinely sits well above a few hundred MB before any
    model code runs).

    Returns None when VSZ cannot be measured (e.g. no /proc, as on macOS).
    Peak RSS is NOT an acceptable substitute here: it can sit far below VSZ
    for a JAX process, which would under-set the cap and reproduce the very
    "next allocation fails" failure this baseline exists to avoid.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    # Format: "VmSize:\t  123456 kB"
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


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

    Notes
    -----
    Resource limits (memory, signals) are only enforced on Unix-like systems.
    On Windows, this context manager yields without enforcing limits.

    Examples
    --------
    >>> with resource_limits(timeout=5.0, memory_mb=256):
    ...     # Execute potentially slow/memory-intensive code
    ...     result = execute_model(model, data)
    """
    # On Windows, resource limits are not available
    # Yield without enforcing any limits
    if not _HAS_RESOURCE_LIMITS:
        logger.debug(
            "Resource limits not available on this platform (Windows), "
            "skipping enforcement",
        )
        yield
        return

    # Unix-specific resource limiting
    # signal.signal() can only be called from the main thread
    if threading.current_thread() is not threading.main_thread():
        logger.debug(
            "Resource limits not available from non-main thread, skipping enforcement",
        )
        yield
        return

    # Timer for timeout (uses threading.Timer + SIGALRM, not signal.alarm())
    timer = None
    timeout_occurred = False
    old_handler = signal.SIG_DFL
    signal_installed = False

    def timeout_handler():
        nonlocal timeout_occurred
        timeout_occurred = True
        # Send signal to main thread
        os.kill(os.getpid(), signal.SIGALRM)

    def signal_handler(signum, frame):
        if timeout_occurred:
            raise ResourceLimitError(f"Execution timeout ({timeout}s exceeded)")
        # Not our timeout — forward to displaced handler
        if callable(old_handler):
            old_handler(signum, frame)

    # Enforce memory limit via RLIMIT_AS (virtual address space).
    #
    # IMPORTANT: RLIMIT_AS caps *virtual* address space, which GPU/TPU runtimes
    # (CUDA especially) reserve in enormous amounts — tens of GB — regardless of
    # actual physical usage. Capping AS around accelerator code makes the limit
    # both ineffective and crash-prone: CUDA initialization aborts hard (often
    # SIGABRT / cudaErrorMemoryAllocation) instead of raising a catchable
    # MemoryError, which would take down the whole process rather than sandbox
    # it. So when a JAX accelerator backend is already active we skip the AS cap
    # and rely on the timeout alone.
    accelerator_active = _accelerator_backend_active()
    current_vsize = None if accelerator_active else _current_virtual_memory_bytes()
    enforce_mem = current_vsize is not None
    # Placeholders (ints) so mypy sees a concrete tuple type for setrlimit; they
    # are always reassigned from getrlimit before use when enforce_mem is True,
    # and only read inside `if enforce_mem` guards.
    old_soft = old_hard = 0
    effective_limit = 0
    if enforce_mem:
        assert current_vsize is not None  # guaranteed by enforce_mem's definition
        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
        limit_bytes = memory_mb * 1024 * 1024
        # memory_mb is a budget ON TOP OF whatever virtual address space the
        # interpreter already holds, not an absolute ceiling. A process that
        # has already imported JAX/NumPy routinely sits at several hundred MB
        # of VSZ before any model code runs; capping RLIMIT_AS at a bare
        # memory_mb would leave zero headroom and fail the very next
        # allocation (even a new thread stack) instead of the model's own
        # memory use.
        effective_limit = current_vsize + limit_bytes
        # Only tighten the limit, never loosen it
        if old_hard != resource.RLIM_INFINITY:
            effective_limit = min(effective_limit, old_hard)
        if old_soft != resource.RLIM_INFINITY:
            effective_limit = min(effective_limit, old_soft)
    elif accelerator_active:
        logger.warning(
            "JAX accelerator backend active; skipping RLIMIT_AS memory cap "
            "(virtual address-space limits are incompatible with GPU/TPU "
            "runtimes and crash CUDA initialization). Timeout enforcement still "
            "applies.",
        )
    else:
        logger.debug(
            "Could not determine current virtual memory size on this platform; "
            "skipping RLIMIT_AS memory cap to avoid under-setting it below "
            "already-committed address space. Timeout enforcement still applies.",
        )

    try:
        if enforce_mem:
            resource.setrlimit(resource.RLIMIT_AS, (effective_limit, old_hard))
        old_handler = signal.signal(signal.SIGALRM, signal_handler)  # type: ignore[assignment]
        signal_installed = True
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()

        yield

    except MemoryError as err:
        raise ResourceLimitError(f"Memory limit ({memory_mb}MB) exceeded") from err
    finally:
        # Cancel timer first to prevent spurious SIGALRM after cleanup
        if timer is not None:
            timer.cancel()
        # Restore original SIGALRM handler (only if we installed ours)
        # Do NOT call signal.alarm(0) — we use threading.Timer, not signal.alarm(),
        # so clearing alarm() would cancel the caller's pre-existing alarm instead.
        if signal_installed:
            signal.signal(signal.SIGALRM, old_handler)  # type: ignore[arg-type]
        # Restore memory limit last (only if we set it)
        if enforce_mem:
            resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))


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
                backupCount=min(self.retention_days, 10),
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
_audit_logger_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance.

    Returns
    -------
    AuditLogger
        The global audit logger.
    """
    global _audit_logger  # noqa: PLW0603 - singleton pattern
    if _audit_logger is None:
        with _audit_logger_lock:
            if _audit_logger is None:
                _audit_logger = AuditLogger()
    return _audit_logger
