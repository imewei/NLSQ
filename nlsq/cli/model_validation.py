"""Model file validation for security.

This module provides security validation for custom model files loaded
through the NLSQ CLI. It inspects model files for dangerous patterns
that could lead to arbitrary code execution.

.. warning::
   This is a best-effort static **denylist**, NOT a sandbox. Python is too
   dynamic for name-based AST checks to be complete, so a passing validation
   does not prove a model file is safe. Do not load model files from
   untrusted sources; a model file is Python code executed with the
   privileges of the calling process.

An opt-in **strict allowlist** mode (``validate_model(..., strict=True)``,
``strict: true`` in the model config, or ``NLSQ_STRICT_MODELS=1``) adds a
positive import/attribute allowlist on top of the denylist. It narrows the
attack surface but is likewise not a sandbox.

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
- File modification: open with write mode, or with a mode that is not a
  string literal (fail closed); io.FileIO; numpy file writers
- Process creation under any module name: posix_spawn, exec*, fork*, spawn*
  (including via the ``posix``/``nt`` aliases of ``os``)
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
        # os process-creation/exec names, matched by attribute/symbol name so a
        # renamed module (`posix`, `nt`, `import x as y`) does not hide them
        "posix_spawn",
        "posix_spawnp",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "fork",
        "forkpty",
        "vfork",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "startfile",
        "fdopen",
        # File creation without open(): io.FileIO and numpy file writers
        # (pickle-capable `allow_pickle` is blocked as a keyword, see visitor)
        "FileIO",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "tofile",
        "dump",
        "memmap",
        "open_memmap",
        "write_array",
        "write_array_header_1_0",
        "write_array_header_2_0",
        "ctypeslib",
        "f2py",
        # Deserialisation that executes code (PyYAML python tags, pickle)
        "unsafe_load",
        "unsafe_load_all",
        "UnsafeLoader",
        "load_extension",
        "savemat",
        "mmwrite",  # scipy.io.mmio file writers
        "hb_write",
        "save_npz",
        "netcdf_file",
        "FortranFile",
        # String annotations are eval()'d by singledispatch.register,
        # typing.get_type_hints and friends; the visitor also rejects any
        # string constant in an annotation position.
        "ForwardRef",
        "get_type_hints",
        "_eval_type",
        "evaluate_forward_ref",
        "get_annotations",
        "__annotations__",
        "__annotate__",
        # Pickle-capable numpy readers (allow_pickle is positional: 2nd param
        # of read_array, 3rd of NpzFile) and header readers
        "read_array",
        "NpzFile",
        "read_array_header_1_0",
        "read_array_header_2_0",
        # Deserialisers of compiled/pickled artefacts and compiler-invoking gadgets
        "deserialize",
        "deserialize_and_load",
        "serialize_executable",
        "array_serialization",
        "pytree_serialization",
        "colocated_python",
        "compilation_cache",
        "extbuild",
        "build_and_import_extension",
        # Network / file side effects reachable from numeric packages
        "DataSource",
        "distributed",
        "start_trace",
        "start_server",
        "save_device_memory_profile",
        # Private implementation modules of numeric packages
        "_src",
        "_core",
        # string.Formatter resolves dotted attribute chains from runtime strings
        "get_field",
        "vformat",
        "Formatter",
        "format_field",
        "convert_field",
        # closure / alias handles onto otherwise blocked objects
        "cell_contents",
        "bltns",  # enum's alias for the builtins module
        # jax.profiler.trace(dir) writes trace files
        "profiler",
        "FullLoader",
        "read_pickle",
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
        # yaml.unsafe_load("!!python/object/apply:...") and
        # pandas.read_pickle() execute arbitrary code on load
        "yaml",
        "pandas",
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
        # `os` under another name (posix on Unix, nt on Windows) and the C
        # accelerator modules behind subprocess/io
        "posix",
        "nt",
        "_posixsubprocess",
        "_io",
        "pty",
        # Stdlib modules whose .run()/.runctx()/.run_docstring_examples()
        # compile+exec a code string
        "cProfile",
        "profile",
        "trace",
        "bdb",
        "pdb",
        "doctest",
        "unittest",  # unittest.mock.patch("pkg.attr") imports by dotted name
        "compileall",
        "py_compile",
        "tkinter",  # Tcl interpreter: tk.eval()/tk.call("exec", ...)
        "turtle",
        "idlelib",
        # Process/thread spawning, interpreter-state and timeout tampering
        # (signal/resource can disarm the loader's own resource_limits)
        "webbrowser",
        "threading",
        "_thread",
        "sched",
        "atexit",
        "signal",
        "resource",
        "gc",
        "fcntl",
        "venv",
        "ensurepip",
        # Import-machinery / site hooks (.pth lines are exec'd)
        "site",
        "zipimport",
        "zipapp",
        "imp",
        "setuptools",
        "pkg_resources",
        "distutils",
        "pip",
        # File creation / config-driven callables / native extension loading
        "sqlite3",
        "dbm",
        "tempfile",
        "zipfile",
        "tarfile",
        "fileinput",
        "mmap",
        "logging",
        # Network (urllib/http/socket already blocked)
        "ssl",
        "selectors",
        "socketserver",
        "xmlrpc",
        "wsgiref",
        "imaplib",
        "poplib",
        "nntplib",
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "websockets",
        # Third-party: unsafe deserialisation, string evaluation, process pools
        "joblib",
        "sympy",  # sympify()/parse_expr() eval strings
        "numexpr",
        "h5py",
        "torch",  # torch.load unpickles
        "dask",
        "ray",
        "annotationlib",
        "string",
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
    # PyYAML python tags (consumed by an unsafe loader)
    "!!python/",
)


_NUMERIC_ROOTS: frozenset[str] = frozenset({"numpy", "scipy", "jax"})
# Dunders that legitimate models read and that expose nothing dangerous.
_VETTED_DUNDERS: frozenset[str] = frozenset({"__name__", "__doc__", "__version__"})
_COLLIDING_SYMBOLS: frozenset[str] = frozenset(
    {"signal", "trace", "profile", "gc", "resource", "logging"}
)


# --- Strict allowlist mode (opt-in; see AllowlistVisitor) -------------------
# A module path is allowed iff it equals one of these entries or lies below
# one (``numpy.linalg`` is below ``numpy``). Extend here to widen the mode.
ALLOWLIST_MODULES: frozenset[str] = frozenset(
    {
        "__future__",
        "math",
        "cmath",
        "functools",
        "typing",
        "collections.abc",
        "dataclasses",
        "numpy",
        "jax",
        "jax.numpy",
        "jax.scipy",
        "jax.lax",
        "scipy.special",
        "scipy.stats",
        "scipy.constants",
    }
)

# Dotted paths (and everything below them) that stay blocked even though their
# root is allowlisted: file/pickle I/O, native-code loaders, runtime config and
# profiler/debug/export hooks of numpy and jax.
ALLOWLIST_BLOCKED_ATTRS: frozenset[str] = frozenset(
    {
        "numpy.ctypeslib",
        "numpy.f2py",
        "numpy.testing",
        "numpy.distutils",
        "numpy.lib.format",
        "numpy.lib.npyio",
        "numpy.lib.NpzFile",
        "numpy.load",
        "numpy.loadtxt",
        "numpy.genfromtxt",
        "numpy.fromregex",
        "numpy.fromfile",
        "numpy.save",
        "numpy.savez",
        "numpy.savez_compressed",
        "numpy.savetxt",
        "numpy.tofile",
        "numpy.memmap",
        "jax.numpy.load",
        "jax.config",
        "jax.experimental",
        "jax.profiler",
        "jax.export",
        "jax.debug",
        "jax.extend",
        "jax.ffi",
        "jax.lib",
        "jax.interpreters",
    }
)

# Names that may not appear as ANY Name reference in strict mode (not just as
# call targets), so aliasing (`o = open`) and passing them around is rejected.
ALLOWLIST_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
    }
)

# Last component of each blocked path, denied on ANY base object. This catches
# module objects that reached a local name by a route alias tracking misses
# (`m = [jax][0]; m.config`). `lib` is too common (`np.lib.stride_tricks`).
_ALLOWLIST_BLOCKED_TERMINALS: frozenset[str] = frozenset(
    p.rsplit(".", 1)[-1] for p in ALLOWLIST_BLOCKED_ATTRS
) - {"lib"}


def _under(path: str, prefixes: frozenset[str]) -> bool:
    """True if ``path`` equals or lies below any dotted prefix."""
    return any(path == p or path.startswith(p + ".") for p in prefixes)


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
        # ids of nodes used directly as a call target; an `open` reference
        # anywhere else (`o = open; o(p, "w")`) escapes the mode check
        self._callees: set[int] = set()
        # local name -> dotted module path, from import statements seen so far
        self._aliases: dict[str, str] = {}

    def visit_Name(self, node: ast.Name) -> Any:
        """Check for dangerous name references."""
        if node.id in DANGEROUS_PATTERNS:
            self.violations.append(f"Dangerous name reference: {node.id}")
        if node.id == "open" and id(node) not in self._callees:
            self.violations.append("Dangerous name reference: open (aliased)")
        if (
            node.id == "load"
            and isinstance(node.ctx, ast.Load)
            and id(node) not in self._callees
        ):
            self.violations.append("Dangerous name reference: load (aliased)")
        self.generic_visit(node)

    def _check_annotation(self, annotation: ast.expr | None) -> None:
        """Reject string constants in annotations (they are eval()'d on demand)."""
        if annotation is None:
            return
        for sub in ast.walk(annotation):
            if (
                isinstance(sub, ast.Constant)
                and isinstance(sub.value, str)
                and not self._is_plain_type_string(sub.value)
            ):
                self.violations.append(
                    "Dangerous string annotation (evaluated by get_type_hints/register)"
                )
                return

    @staticmethod
    def _is_plain_type_string(text: str) -> bool:
        """True for forward references made only of names/attributes/subscripts
        ("jax.Array", "list[int] | None"): evaluating them cannot call anything."""
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError:
            return False
        allowed = (
            ast.Expression, ast.Name, ast.Attribute, ast.Subscript, ast.Tuple,
            ast.BitOr, ast.BinOp, ast.Load,
        )  # fmt: skip
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Constant) and (
                sub.value is None or sub.value is Ellipsis
            ):
                continue
            if not isinstance(sub, allowed):
                return False
            if isinstance(sub, ast.BinOp) and not isinstance(sub.op, ast.BitOr):
                return False
            if isinstance(sub, ast.Name) and sub.id in DANGEROUS_PATTERNS:
                return False
            if isinstance(sub, ast.Attribute) and (
                sub.attr in DANGEROUS_PATTERNS or sub.attr.startswith("_")
            ):
                return False
        return True

    def visit_arg(self, node: ast.arg) -> Any:
        """Check parameter annotations."""
        self._check_annotation(node.annotation)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Check return annotations."""
        self._check_annotation(node.returns)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        """Check variable annotations."""
        self._check_annotation(node.annotation)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """Check for dangerous function calls."""
        self._callees.add(id(node.func))
        if any(k.arg == "eval_str" for k in node.keywords):
            self.violations.append("Dangerous keyword argument: eval_str")
        # numpy.load(..., allow_pickle=...) executes pickle payloads
        if any(k.arg == "allow_pickle" for k in node.keywords):
            self.violations.append("Dangerous keyword argument: allow_pickle")
        # np.load's allow_pickle is its THIRD POSITIONAL parameter; fail closed
        # on any call shape that can supply it (or hide arguments).
        callee = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        if callee == "load" and (
            len(node.args) >= 3
            or any(isinstance(a, ast.Starred) for a in node.args)
            or any(k.arg is None for k in node.keywords)
        ):
            self.violations.append(
                "Dangerous call: load() with allow_pickle-capable arguments"
            )
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
        """Check open() calls; fail closed unless the mode is a literal read mode."""
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            self.violations.append("File operation: open() with unpacked arguments")
            return
        modes = list(node.args[1:2]) + [
            k.value for k in node.keywords if k.arg == "mode"
        ]
        # pathlib.Path.open(mode) / io.open(mode=...) put the mode first
        if isinstance(node.func, ast.Attribute) and node.args:
            first = node.args[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value
                and set(first.value) <= set("rwxabt+U")
            ):
                modes.append(first)
        for mode_arg in modes:
            if not (
                isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str)
            ):
                # a variable/expression mode cannot be proven read-only
                self.violations.append("File write risk: open() mode is not a literal")
            elif any(c in mode_arg.value for c in "wax+"):
                self.violations.append(
                    f"File write operation: open(..., '{mode_arg.value}')",
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
        else:
            reason = self._attr_violation(node.attr)
            if reason:
                self.violations.append(
                    f"Dangerous attribute access: .{node.attr} ({reason})"
                )
        if node.attr == "io":
            root: ast.expr = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if (
                isinstance(root, ast.Name)
                and self._aliases.get(root.id, root.id).split(".")[0] == "scipy"
            ):
                self.violations.append(
                    "Dangerous attribute access: scipy.io (file writers)"
                )
        if node.attr == "open" and id(node) not in self._callees:
            self.violations.append("Dangerous attribute access: .open (aliased)")
        if node.attr == "load" and id(node) not in self._callees:
            self.violations.append("Dangerous attribute access: .load (aliased)")
        self.generic_visit(node)

    @staticmethod
    def _attr_violation(attr: str) -> str | None:
        """Reason an attribute NAME is unsafe regardless of its base, else None.

        Reachability rule: public chains such as ``typing.sys`` or
        ``jax.version.subprocess`` hand out blocked modules, and private or
        dunder attributes (``len.__self__``, ``f.__closure__``) reach arbitrary
        objects. Names that merely collide with a blocked module but are public
        numeric APIs (jnp.trace, scipy.signal) are exempt.
        """
        if attr.startswith("_") and attr not in _VETTED_DUNDERS:
            return "private attribute"
        if attr in DANGEROUS_PATTERNS:
            return "dangerous name"
        if attr in DANGEROUS_MODULES and attr not in _COLLIDING_SYMBOLS:
            return "reaches a blocked module"
        return None

    def visit_MatchClass(self, node: ast.MatchClass) -> Any:
        """``case object(__self__=m)``: kwd_attrs are plain strings, not Attribute nodes."""
        for attr in node.kwd_attrs:
            reason = self._attr_violation(attr)
            if reason:
                self.violations.append(
                    f"Dangerous match class attribute: {attr} ({reason})"
                )
        self.generic_visit(node)

    def _check_capture(self, name: str | None) -> None:
        if name and (name in DANGEROUS_PATTERNS or name in DANGEROUS_MODULES):
            self.violations.append(f"Dangerous match capture name: {name}")

    def visit_MatchAs(self, node: ast.MatchAs) -> Any:
        """Check capture names."""
        self._check_capture(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> Any:
        """Check capture names."""
        self._check_capture(node.name)
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> Any:
        """Check the ``**rest`` capture name."""
        self._check_capture(node.rest)
        self.generic_visit(node)

    @staticmethod
    def _bad_module_path(dotted: str) -> bool:
        """True for a dangerous submodule (jax.experimental.serialize_executable)
        or a private module of a numeric package (jax._src.*, numpy.lib._npyio_impl)."""
        parts = dotted.split(".")
        if parts[:2] == ["scipy", "io"]:  # wavfile.write, savemat, netcdf, ...
            return True
        if any(p in DANGEROUS_PATTERNS for p in parts[1:]):
            return True
        return parts[0] in _NUMERIC_ROOTS and any(p.startswith("_") for p in parts[1:])

    def visit_Import(self, node: ast.Import) -> Any:
        """Check for dangerous module imports."""
        for alias in node.names:
            module_root = alias.name.split(".")[0]
            if alias.asname:
                self._aliases[alias.asname] = alias.name
            else:
                self._aliases[module_root] = module_root
            if (
                module_root in DANGEROUS_MODULES
                or module_root in DANGEROUS_PATTERNS
                or self._bad_module_path(alias.name)
            ):
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
        # Numeric-library symbols that merely share a name with a blocked
        # module (scipy.signal, jnp.trace, ...). Private submodules
        # (jax._src.config.logging) and non-numeric roots stay blocked.
        exempt_root = module_root in _NUMERIC_ROOTS and not any(
            part.startswith("_") for part in (node.module or "").split(".")
        )
        if node.module and self._bad_module_path(node.module):
            self.violations.append(
                f"Dangerous import: from {node.module} import ...",
            )
        for alias in node.names:
            self._aliases[alias.asname or alias.name] = (
                f"{node.module}.{alias.name}" if node.module else alias.name
            )
            if node.module == "scipy" and alias.name == "io":
                self.violations.append("Dangerous import: from scipy import io")
            if module_root in _NUMERIC_ROOTS and alias.name.startswith("_"):
                self.violations.append(
                    f"Dangerous import: from {node.module} import {alias.name} (private)",
                )
            if alias.name == "load" and alias.asname not in (None, "load"):
                self.violations.append(
                    f"Dangerous import: from {node.module} import load as {alias.asname}",
                )
            if alias.name in DANGEROUS_PATTERNS or (
                alias.name in DANGEROUS_MODULES
                and not (exempt_root and alias.name in _COLLIDING_SYMBOLS)
            ):
                self.violations.append(
                    f"Dangerous import: from {node.module} import {alias.name}",
                )
        self.generic_visit(node)


class AllowlistVisitor(DangerousPatternVisitor):
    """Opt-in strict mode: the denylist checks plus a positive allowlist.

    Every ``DangerousPatternVisitor`` check still runs (each override calls
    ``super()``). On top of that:

    * imports must fall under ``ALLOWLIST_MODULES`` (relative and wildcard
      imports are rejected);
    * every attribute chain rooted at an import alias is resolved to its full
      dotted path and must fall under the allowlist and outside
      ``ALLOWLIST_BLOCKED_ATTRS`` (so ``import jax; jax.config`` and
      ``from scipy import special`` + ``special.x`` are judged on the real path);
    * aliases are followed through simple assignment; a target bound to a
      blocked callable or non-allowlisted path is tainted and any use of it is
      a violation;
    * ``ALLOWLIST_FORBIDDEN_NAMES`` are rejected as any Name reference;
    * a module object used as a bare value (``f(np)``, ``x = jax``) is rejected
      because it can hand an unchecked module to code the visitor cannot follow.

    This narrows the attack surface; like the denylist it is NOT a sandbox.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tainted: set[str] = set()
        # ids of Name/Attribute nodes that are the base of a longer chain
        self._inner: set[int] = set()

    def _flag(self, msg: str) -> None:
        self.violations.append(f"strict allowlist: {msg}")

    @staticmethod
    def _path_problem(path: str) -> str | None:
        if _under(path, ALLOWLIST_BLOCKED_ATTRS):
            return "blocked path"
        if not _under(path, ALLOWLIST_MODULES):
            return "not an allowlisted module"
        return None

    @staticmethod
    def _is_module_path(path: str) -> bool:
        """An allowlisted module itself, or a package that only contains one."""
        return path in ALLOWLIST_MODULES or any(
            m.startswith(path + ".") for m in ALLOWLIST_MODULES
        )

    def _resolve(self, node: ast.expr) -> str | None:
        """Dotted path of a Name/Attribute chain rooted at an import alias."""
        if isinstance(node, ast.Name):
            return self._aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _check_import_path(self, dotted: str) -> None:
        problem = self._path_problem(dotted)
        if problem:
            self._flag(f"import of '{dotted}' ({problem})")

    def visit_Import(self, node: ast.Import) -> Any:
        super().visit_Import(node)
        for alias in node.names:
            self._check_import_path(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        super().visit_ImportFrom(node)
        if node.level:
            self._flag("relative import")
            return
        for alias in node.names:
            if alias.name == "*":
                self._flag("wildcard import")
            else:
                self._check_import_path(f"{node.module}.{alias.name}")

    def visit_Name(self, node: ast.Name) -> Any:
        super().visit_Name(node)
        if node.id in ALLOWLIST_FORBIDDEN_NAMES:
            self._flag(f"forbidden name '{node.id}'")
        if isinstance(node.ctx, ast.Load):
            if node.id in self._tainted:
                self._flag(f"use of tainted alias '{node.id}'")
            if id(node) not in self._inner:
                path = self._aliases.get(node.id)
                if path and self._is_module_path(path):
                    self._flag(f"module '{path}' used as a value")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr in _ALLOWLIST_BLOCKED_TERMINALS:
            self._flag(f"blocked attribute '.{node.attr}'")
        if id(node) not in self._inner:
            # outermost node of a chain: mark the rest so each chain is judged once
            inner: ast.expr = node.value
            while True:
                self._inner.add(id(inner))
                if not isinstance(inner, ast.Attribute):
                    break
                inner = inner.value
            path = self._resolve(node)
            if path:
                problem = self._path_problem(path)
                if problem:
                    self._flag(f"attribute chain '{path}' ({problem})")
                elif self._is_module_path(path):
                    self._flag(f"module '{path}' used as a value")
        super().visit_Attribute(node)

    def visit_MatchClass(self, node: ast.MatchClass) -> Any:
        for attr in node.kwd_attrs:
            if attr in _ALLOWLIST_BLOCKED_TERMINALS:
                self._flag(f"blocked match attribute '{attr}'")
        super().visit_MatchClass(node)

    def _bind(self, target: ast.expr, value: ast.expr | None) -> None:
        """Follow ``target = value`` for alias resolution and tainting."""
        if isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(
                target.elts
            ):
                for t, v in zip(target.elts, value.elts, strict=True):
                    self._bind(t, v)
            return
        if not isinstance(target, ast.Name) or value is None:
            return
        if isinstance(value, ast.Name) and (
            value.id in self._tainted or value.id in ALLOWLIST_FORBIDDEN_NAMES
        ):
            self._tainted.add(target.id)
            return
        path = self._resolve(value)
        if path is None:
            return
        if self._path_problem(path):
            self._tainted.add(target.id)
        else:
            self._aliases[target.id] = path

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            self._bind(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._bind(node.target, node.value)
        super().visit_AnnAssign(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> Any:
        self._bind(node.target, node.value)
        self.generic_visit(node)


def strict_models_enabled() -> bool:
    """True when ``NLSQ_STRICT_MODELS`` requests the strict allowlist everywhere."""
    return os.environ.get("NLSQ_STRICT_MODELS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def validate_model(
    path: Path,
    trusted: bool = False,
    source: bytes | None = None,
    strict: bool = False,
) -> ModelValidationResult:
    """Validate a model file for security.

    Performs AST-based static analysis to detect dangerous patterns
    that could lead to arbitrary code execution.

    Parameters
    ----------
    path : Path
        Path to the model file to validate.
    trusted : bool, default=False
        If True, skip validation (user explicitly trusts the model).
    source : bytes, optional
        Raw file bytes to validate instead of reading ``path``. Callers that
        go on to execute the file pass the same bytes to the executor so the
        validated and executed code are identical.
    strict : bool, default=False
        If True, additionally enforce the opt-in strict allowlist
        (:class:`AllowlistVisitor`). The default denylist behaviour is
        unchanged when False.

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
        # Parse raw bytes so a PEP 263 coding cookie is honoured exactly as
        # the interpreter honours it at execution time.
        if source is None:
            source = path.read_bytes()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return ModelValidationResult(
            path=path,
            is_valid=False,
            is_trusted=trusted,
            violations=[f"Syntax error: {e}"],
        )

    # Visit all nodes to find violations
    visitor = AllowlistVisitor() if strict else DangerousPatternVisitor()
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
        Only caps the number of rotated backups (``min(retention_days, 10)``);
        there is no time-based expiry (default: 90).
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
