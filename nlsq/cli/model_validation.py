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
        # site/pydoc builtins: help("mod") imports any module by string
        # (help("antigravity") opens a browser); the rest read files or exit
        "help",
        "license",
        "credits",
        "copyright",
        "exit",
        "quit",
        # numpy readers: URL/DataSource access and a CWD cache write
        "genfromtxt",
        "loadtxt",
        "fromregex",
        "fromfile",
        "fromtextfile",  # numpy.ma.mrecords: reads any text file
        "mrecords",  # numpy.ma.mrecords.openfile opens any file
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
        "dump_stats",  # pstats.Stats / cProfile.Profile: writes to a path
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
        "mkdir",
        # Python 3.14 pathlib mutation methods (``copy`` itself collides with
        # ``np.copy``/``x.copy()`` and is judged in visit_Call)
        "copy_into",
        "move",
        "move_into",
        "lchmod",
        # matplotlib / Qt file writers and process spawners
        "savefig",
        "startDetached",
        "QProcess",
        "QSaveFile",
        "QFile",
        "QDir",
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
        "antigravity",  # opens a web browser on import
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
        "_ctypes",
        "_imp",
        "_frozen_importlib",
        "_frozen_importlib_external",
        "_socket",
        "_winapi",
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
        "bz2",
        "gzip",
        "lzma",
        "fileinput",
        # GUI toolkits / plotting: QProcess spawns processes, QSaveFile and
        # savefig write files
        "PySide6",
        "shiboken6",
        "PyQt5",
        "PyQt6",
        "pyqtgraph",
        "matplotlib",
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
        # Runtime dependencies that are not numeric libraries: evosax unpickles
        # in-memory bytes (load_pkl_object), pytest imports and runs any file
        # (numpy.test/scipy.test delegate to it), psutil can lift the loader's
        # RLIMIT_AS and signal other processes, tqdm.contrib posts to remote
        # services, jaxlib exposes the native XLA client.
        "evosax",
        "pytest",
        "psutil",
        "tqdm",
        "jaxlib",
        "torch",  # torch.load unpickles
        "dask",
        "ray",
        "annotationlib",
        "string",
        # Implicit file writes / URL fetches / terminal or OS-state changes:
        # xml.etree .write(path), xml.sax.parse(url) and xml.dom.pulldom fetch
        # through urllib; pstats .dump_stats(path); audio/mailbox writers;
        # readline history files; cgitb log files; pipes runs shell commands
        "xml",
        "pstats",
        "mailbox",
        "wave",
        "aifc",
        "sunau",
        "ossaudiodev",
        "termios",
        "tty",
        "curses",
        "readline",
        "rlcompleter",
        "netrc",
        "syslog",
        "cgi",
        "cgitb",
        "pipes",
        "winreg",
        "msvcrt",
        "lib2to3",
        "asyncore",
        "asynchat",
        "smtpd",
    },
)


def _private_root(dotted: str) -> bool:
    """True when the ROOT module is underscore-prefixed (``_ctypes``, ``_imp``,
    ``_pickle``): C accelerator/implementation modules that bypass the public
    denylist names. ``__future__`` is the only legitimate one."""
    root = dotted.split(".", maxsplit=1)[0]
    return root.startswith("_") and root != "__future__"


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
    # jax.jit(compiler_options={"xla_dump_to": DIR}) / XLA_FLAGS write files
    "xla_dump",
    # PyYAML python tags (consumed by an unsafe loader)
    "!!python/",
)


_NUMERIC_ROOTS: frozenset[str] = frozenset({"numpy", "scipy", "jax"})
# Dunders that legitimate models read and that expose nothing dangerous.
_VETTED_DUNDERS: frozenset[str] = frozenset({"__name__", "__doc__", "__version__"})
# Also plausible user-object field names (``C.wave``, ``net.pipes``); the
# module imports stay blocked via DANGEROUS_MODULES.
_COLLIDING_SYMBOLS: frozenset[str] = frozenset(
    {
        "signal",
        "trace",
        "profile",
        "gc",
        "resource",
        "readline",
        "wave",
        "pipes",
        "xml",
    }
)

# Numeric-package submodule paths the DEFAULT visitor rejects (strict mode
# blocks these and more via ALLOWLIST_BLOCKED_ATTRS). Each hands out a code
# string compiler (numpy.testing.measure), file/pickle I/O, native loaders or
# interpreter hooks. Only paths whose root is in _NUMERIC_ROOTS are judged.
_DEFAULT_BLOCKED_PATHS: frozenset[str] = frozenset(
    {
        "numpy.testing",
        "numpy.ctypeslib",
        "numpy.f2py",
        "numpy.distutils",
        "numpy.lib.format",
        "numpy.lib.npyio",
        "jax.debug",
        "jax.export",
        "jax.extend",
        "jax.ffi",
        "jax.interpreters",
        "jax.lib",
        "jax.profiler",
        "jax.tools",
    }
)
# jax.experimental is an allowlist: its submodules re-export stdlib modules
# (mosaic.gpu.utils.logging) and jax._src modules (topologies.xb), so anything
# not listed here is rejected.
_JAX_EXPERIMENTAL_OK: frozenset[str] = frozenset(
    {"ode", "sparse", "jet", "checkify", "shard_map", "enable_x64", "disable_x64"}
)
# stdlib pathlib names; Path.copy/move/... mutate files (Python 3.14)
_PATHISH_ATTRS: frozenset[str] = frozenset(
    {
        "pathlib",
        "Path",
        "PurePath",
        "PosixPath",
        "WindowsPath",
        "PurePosixPath",
        "PureWindowsPath",
    }
)
# Attribute names that reach a blocked numeric submodule or a private ``_src``
# module through a public re-export. Rejected on ANY base, so value aliases of
# the numeric modules (``m = [np][0]``) cannot hide them.
_REEXPORT_ATTRS: frozenset[str] = frozenset(
    {
        "testing",
        "ctypeslib",
        "f2py",
        "measure",
        "xb",
        "xla_bridge",
        "jtu",
        "test_util",
        "register_plugin",
        # public numeric modules re-export stdlib pathlib (jax.version.pathlib,
        # numpy.conftest.Path) and codecs (numpy.f2py.crackfortran.codecs); their
        # test packages re-export tempfile/shutil/multiprocessing/importlib
        *_PATHISH_ATTRS,
        "codecs",
        "tests",
        "conftest",
        # numpy.test / scipy.test / np.linalg.test ... are PytestTester objects
        # that call pytest.main(extra_argv, tests=[any .py file])
        "test",
    }
)


# The nlsq package re-exports its own model functions (``from nlsq.core.functions
# import gaussian``); everything else under ``nlsq`` (cli.model_registry loads and
# executes files, cli.result_exporter writes any path, curve_fit checkpoints) is
# off limits to a model file.
_NLSQ_ALLOWED: tuple[str, ...] = ("nlsq.core.functions", "nlsq.functions")


def _nlsq_import_ok(dotted: str) -> bool:
    return any(dotted == a or dotted.startswith(a + ".") for a in _NLSQ_ALLOWED)


def _blocked_numeric_path(dotted: str) -> bool:
    """True when ``dotted`` (rooted at numpy/scipy/jax) is a blocked submodule."""
    parts = dotted.split(".")
    if parts[0] not in _NUMERIC_ROOTS:
        return False
    if _under(dotted, _DEFAULT_BLOCKED_PATHS):
        return True
    if parts[:2] == ["jax", "experimental"] and len(parts) > 2:
        if parts[2] not in _JAX_EXPERIMENTAL_OK:
            return True
    return any(p in _REEXPORT_ATTRS for p in parts[1:])


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
        "numpy.test",  # PytestTester: pytest.main on an arbitrary file
        "numpy.distutils",
        "numpy.lib.format",
        "numpy.lib.npyio",
        "numpy.lib.NpzFile",
        "numpy.load",
        "numpy.loadtxt",
        "numpy.genfromtxt",
        "numpy.fromregex",
        "numpy.fromfile",
        "numpy.ma.mrecords",  # fromtextfile reads any text file
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
        "help",
        "license",
        "credits",
        "copyright",
        "exit",
        "quit",
    }
)

# Last component of each blocked path, denied on ANY base object. This catches
# module objects that reached a local name by a route alias tracking misses
# (`m = [jax][0]; m.config`). `lib` is too common (`np.lib.stride_tricks`).
_ALLOWLIST_BLOCKED_TERMINALS: frozenset[str] = frozenset(
    p.rsplit(".", 1)[-1] for p in ALLOWLIST_BLOCKED_ATTRS
) - {"lib"}


# site builtins that are ordinary words (`def m(x, entry, exit)`): a Name of one is
# safe only when it resolves to a function-local binding, never to the builtin.
_SHADOWABLE_BUILTINS: frozenset[str] = frozenset(
    {"help", "license", "credits", "copyright", "exit", "quit"}
)

_NESTED_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _function_scope(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[set, set]:
    """(locals, globals) of a function body. Nested scopes are not entered, so a
    missed binding kind only makes the check stricter, never looser."""
    a = node.args
    bound = {
        x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg) if x
    }
    declared_global: set[str] = set()
    nonlocal_: set[str] = set()
    stack: list[ast.AST] = list(node.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)  # body, decorators, defaults: not walked
            continue
        if isinstance(n, _NESTED_SCOPES):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.alias):
            bound.add((n.asname or n.name).split(".")[0])
        elif (isinstance(n, ast.ExceptHandler) and n.name) or (
            isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name
        ):
            bound.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            bound.add(n.rest)
        elif isinstance(n, ast.Global):
            declared_global.update(n.names)
        elif isinstance(n, ast.Nonlocal):
            nonlocal_.update(n.names)
        stack.extend(ast.iter_child_nodes(n))
    return bound - declared_global - nonlocal_, declared_global


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
        # (kind, locals, globals) of the enclosing function/class scopes
        self._scopes: list[tuple[str, set[str], set[str]]] = []

    def visit_Name(self, node: ast.Name) -> Any:
        """Check for dangerous name references."""
        if node.id in DANGEROUS_PATTERNS and not self._shadowed(node.id):
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
        """Check return annotations; the body is visited in the function's scope."""
        self._check_annotation(node.returns)
        # decorators, defaults and annotations evaluate in the enclosing scope
        for child in (
            *node.decorator_list,
            node.args,
            node.returns,
            *getattr(node, "type_params", ()),
        ):
            if child is not None:
                self.visit(child)
        self._scopes.append(("function", *_function_scope(node)))
        for stmt in node.body:
            self.visit(stmt)
        self._scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        """A class body runs top to bottom with builtins as fallback: no shadowing."""
        for child in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(child)
        for tp in getattr(node, "type_params", ()):
            self.visit(tp)
        self._scopes.append(("class", set(), set()))
        for stmt in node.body:
            self.visit(stmt)
        self._scopes.pop()

    def _shadowed(self, name: str) -> bool:
        """True if ``name`` is a shadowable builtin bound by an enclosing function,
        so it cannot reach the builtin. Module/class bindings never count: code
        there runs in order and a Load before the binding hits the builtin."""
        if name not in _SHADOWABLE_BUILTINS:
            return False
        for i, (kind, bound, declared_global) in enumerate(reversed(self._scopes)):
            if kind == "class":
                if i == 0:
                    return False
                continue  # methods skip enclosing class scopes
            if name in declared_global:
                return False
            if name in bound:
                return True
        return False

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
        # jax.jit(..., compiler_options={"xla_dump_to": DIR}) writes files
        if any(k.arg == "compiler_options" for k in node.keywords):
            self.violations.append("Dangerous keyword argument: compiler_options")
        # `jax.jit(f, **opts)`: the compiler_options keyword can be built at runtime
        if (
            isinstance(node.func, (ast.Name, ast.Attribute))
            and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr)
            in ("jit", "pmap")
            and any(k.arg is None for k in node.keywords)
        ):
            self.violations.append(
                "Dangerous call: jit/pmap with ** keyword unpacking "
                "(compiler_options can write files)"
            )
        # np.load(file, mmap_mode, allow_pickle, ...): mmap_mode "r+"/"w+"
        # writes or truncates the file, allow_pickle unpickles. Fail closed on
        # any call shape that can supply either (or hide arguments); only a
        # literal read-only mmap_mode (None, "r", "c") is accepted.
        callee = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        if callee == "load":
            mmap_modes = node.args[1:2] + [
                k.value for k in node.keywords if k.arg == "mmap_mode"
            ]
            if (
                len(node.args) >= 3
                or any(isinstance(a, ast.Starred) for a in node.args)
                or any(k.arg is None for k in node.keywords)
                or any(
                    not (isinstance(m, ast.Constant) and m.value in (None, "r", "c"))
                    for m in mmap_modes
                )
            ):
                self.violations.append(
                    "Dangerous call: load() with writable mmap_mode or "
                    "allow_pickle-capable arguments"
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
            # Python 3.14 Path.copy(target=...): the receiver type is unknowable,
            # so judge by argument shape. ndarray.copy takes only ``order``;
            # accept no arguments or one literal order (positional or keyword).
            if (
                node.func.attr == "copy"
                and not self._numeric_root(node.func)
                and not self._is_array_copy_call(node)
            ):
                self.violations.append(
                    "Dangerous method call: .copy(...) (pathlib.Path.copy risk; "
                    "pass a literal order 'C'/'F'/'A'/'K' or use trusted: true)"
                )
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

    def _numeric_root(self, node: ast.expr) -> bool:
        """True for ``np.copy``-style chains: rooted at an imported numpy/scipy/jax
        alias, with no pathlib re-export and no class (capitalised) segment, so a
        re-exported ``Path`` can never inherit the numeric ``.copy`` exemption."""
        parts = (self._resolve(node) or "").split(".")
        return parts[0] in _NUMERIC_ROOTS and not any(
            p in _REEXPORT_ATTRS or p[:1].isupper() for p in parts[1:-1]
        )

    @staticmethod
    def _is_array_copy_call(node: ast.Call) -> bool:
        """``x.copy()`` / ``x.copy('C')`` / ``x.copy(order='F')``, literal order only."""
        args = [*node.args, *(k.value for k in node.keywords)]
        if any(k.arg not in (None, "order") for k in node.keywords) or any(
            k.arg is None for k in node.keywords
        ):
            return False
        if len(args) > 1 or any(isinstance(a, ast.Starred) for a in args):
            return False
        return not args or (
            isinstance(args[0], ast.Constant) and args[0].value in ("C", "F", "A", "K")
        )

    def _resolve(self, node: ast.expr) -> str | None:
        """Dotted path of a Name/Attribute chain rooted at an import alias."""
        if isinstance(node, ast.Name):
            return self._aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

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
        chain = self._resolve(node)
        if chain and _blocked_numeric_path(chain):
            self.violations.append(
                f"Dangerous attribute access: {chain} (blocked numeric submodule)"
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
        # Any `.config` off a base other than self/cls: value aliases of the jax
        # module (`J = jax`, default args, walrus, lambdas) defeat import-alias
        # tracking, and no numeric model needs another object's `.config`.
        if node.attr == "config" and not (
            isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")
        ):
            self.violations.append(
                "Dangerous attribute access: .config (jax.config: runtime config, file writes)"
            )
        if node.attr == "open" and id(node) not in self._callees:
            self.violations.append("Dangerous attribute access: .open (aliased)")
        if (
            node.attr == "copy"
            and id(node) not in self._callees
            and not self._numeric_root(node)
        ):  # `c = Path(p).copy; c(dst)` escapes the call-shape check
            self.violations.append("Dangerous attribute access: .copy (aliased)")
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
        if attr in _REEXPORT_ATTRS:
            return "reaches a blocked numeric submodule"
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
            elif attr == "config":  # same any-base rule as visit_Attribute
                self.violations.append(
                    "Dangerous match class attribute: config (jax.config)"
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
        if parts[:2] in (["scipy", "io"], ["jax", "config"]):  # file writers / config
            return True
        if parts[:3] == ["numpy", "ma", "mrecords"]:  # fromtextfile reads any file
            return True
        if _blocked_numeric_path(dotted):
            return True
        if any(p in DANGEROUS_PATTERNS for p in parts[1:]):
            return True
        return parts[0] in _NUMERIC_ROOTS and any(p.startswith("_") for p in parts[1:])

    def visit_Import(self, node: ast.Import) -> Any:
        """Check for dangerous module imports."""
        for alias in node.names:
            module_root = alias.name.split(".")[0]
            # `import nlsq` / `import nlsq.core.functions` bind the package root
            if module_root == "nlsq" and not (
                alias.asname and _nlsq_import_ok(alias.name)
            ):
                self.violations.append(
                    f"Dangerous import: import {alias.name} (only "
                    "`import nlsq.core.functions as name` is allowed from nlsq)"
                )
            if alias.asname:
                self._aliases[alias.asname] = alias.name
            else:
                self._aliases[module_root] = module_root
            if (
                module_root in DANGEROUS_MODULES
                or module_root in DANGEROUS_PATTERNS
                or _private_root(alias.name)
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
            module_root in DANGEROUS_MODULES
            or module_root in DANGEROUS_PATTERNS
            or _private_root(module_root)
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
            if module_root == "nlsq" and not node.level:
                if alias.name == "*" or not _nlsq_import_ok(
                    f"{node.module}.{alias.name}"
                ):
                    self.violations.append(
                        f"Dangerous import: from {node.module} import {alias.name} "
                        "(only nlsq.core.functions is allowed from nlsq)"
                    )
            self._aliases[alias.asname or alias.name] = (
                f"{node.module}.{alias.name}" if node.module else alias.name
            )
            if node.module and _blocked_numeric_path(f"{node.module}.{alias.name}"):
                self.violations.append(
                    f"Dangerous import: from {node.module} import {alias.name} "
                    "(blocked numeric submodule)"
                )
            if module_root == "jax" and alias.name == "config":
                self.violations.append("Dangerous import: from jax import config")
            if node.module == "jax" and alias.name == "*":  # binds bare `config`
                self.violations.append("Dangerous import: from jax import *")
            if node.module == "scipy" and alias.name == "io":
                self.violations.append("Dangerous import: from scipy import io")
            if not node.module and _private_root(alias.name):
                # `from . import _ctypes`: relative form of an underscore module
                self.violations.append(
                    f"Dangerous import: from {'.' * node.level} import {alias.name}",
                )
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
        if node.id in ALLOWLIST_FORBIDDEN_NAMES and not self._shadowed(node.id):
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
            value.id in self._tainted
            or (value.id in ALLOWLIST_FORBIDDEN_NAMES and not self._shadowed(value.id))
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
        # ast.parse defers placement checks (top-level return/yield, ...) to the
        # compiler; the registry executes the file, so reject what it cannot compile.
        compile(tree, str(path), "exec", dont_inherit=True)
    except SyntaxError as e:
        return ModelValidationResult(
            path=path,
            is_valid=False,
            is_trusted=trusted,
            violations=[f"Syntax error: {e}"],
        )

    violations.extend(_find_protocol_hook_violations(tree))
    violations.extend(_find_module_machinery_violations(tree))

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


# Hooks that let a model fetch an attribute named by a runtime-built string
# ("__glo" + "bals__"), which no literal check sees: a class-pattern
# `case C(g)` reads the names in __match_args__ (reached through a
# __subclasshook__ that makes any object an instance), and update_wrapper copies
# the attributes named in `assigned`.
_PROTOCOL_HOOK_NAMES: frozenset[str] = frozenset(
    {
        "__match_args__",
        "__subclasshook__",
        "__instancecheck__",
        "__subclasscheck__",
        "update_wrapper",
    }
)
_PROTOCOL_HOOK_KEYWORDS: frozenset[str] = frozenset({"assigned", "updated"})


def _local_hook_keyword_targets(tree: ast.AST) -> dict[str, set[str]]:
    """Hook-keyword names (``updated``) that each file-local function declares as
    its own parameter, for functions whose name is bound exactly once. A call
    ``scale(x, updated=a)`` to such a function is an ordinary numeric argument."""
    params: dict[str, set[str]] = {}
    bindings: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            own = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
            params[node.name] = own & _PROTOCOL_HOOK_KEYWORDS
            bindings[node.name] = bindings.get(node.name, 0) + 1
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bindings[node.id] = bindings.get(node.id, 0) + 1
        elif isinstance(node, (ast.ClassDef, ast.alias)):
            n = (
                node.name
                if isinstance(node, ast.ClassDef)
                else node.asname or node.name
            )
            bindings[n.split(".")[0]] = bindings.get(n.split(".")[0], 0) + 1
    return {n: p for n, p in params.items() if p and bindings.get(n) == 1}


def _find_protocol_hook_violations(tree: ast.AST) -> list[str]:
    """Flag names and keywords that fetch attributes by runtime string."""
    found: list[str] = []
    local_ok = _local_hook_keyword_targets(tree)
    exempt_kw: set[int] = set()  # id() of keyword nodes passed to a local function
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            allowed = local_ok.get(node.func.id, set())
            exempt_kw.update(id(k) for k in node.keywords if k.arg in allowed)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.alias):
            names.append(node.name.rsplit(".", 1)[-1])
        elif (
            isinstance(node, ast.keyword)
            and node.arg in _PROTOCOL_HOOK_KEYWORDS
            and id(node) not in exempt_kw
        ):
            found.append(f"Dangerous keyword argument: {node.arg}=")
        found.extend(
            f"Dangerous attribute-by-name hook: {n}"
            for n in names
            if n in _PROTOCOL_HOOK_NAMES
        )
    return found


# Module-machinery dunders. Assigning ``__package__`` makes a relative import
# resolve against any package (``__package__ = "os"; from . import system``), which
# skips every module-name check, so none of these may be bound by a model file.
_MODULE_MACHINERY_NAMES: frozenset[str] = frozenset(
    {
        "__package__",
        "__spec__",
        "__path__",
        "__loader__",
        "__builtins__",
        "__name__",
        "__file__",
        "__cached__",
        "__import__",
    }
)


def _find_module_machinery_violations(tree: ast.AST) -> list[str]:
    """Flag relative imports and any binding of module-machinery dunders.

    A model file runs as a standalone module, so a relative import is never
    legitimate; and rebinding ``__package__`` & co. would redirect one.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        bound: list[str | None] = []
        if isinstance(node, ast.ImportFrom) and node.level:
            found.append(
                f"Dangerous import: relative import (from {'.' * node.level}"
                f"{node.module or ''} import ...)"
            )
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.append(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            bound.append(node.attr)
        elif isinstance(node, ast.alias):
            bound.append(node.asname)
        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.ExceptHandler,
                ast.MatchAs,
                ast.MatchStar,
            ),
        ):
            bound.append(node.name)
        elif isinstance(node, ast.MatchMapping):
            bound.append(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.extend(node.names)
        found.extend(
            f"Dangerous module attribute binding: {n}"
            for n in bound
            if n in _MODULE_MACHINERY_NAMES
        )
    return found


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


def _warm_jax_cpu_runtime() -> None:
    """Compile one trivial op on an already-imported JAX before the AS baseline.

    XLA's CPU runtime maps compiler arenas and worker-thread stacks on first
    compile. Taking the VmSize baseline before that makes a module-level
    ``jnp``/``jax.jit`` call in a model exceed the cap, and XLA aborts the whole
    process (SIGABRT) instead of raising ``MemoryError``.
    """
    jax_mod = sys.modules.get("jax")
    if jax_mod is None:
        return
    try:
        jax_mod.jit(lambda v: v + 1)(1.0).block_until_ready()
    except Exception:  # best effort: the cap is still applied from the old baseline
        logger.debug("JAX CPU warm-up before RLIMIT_AS baseline failed", exc_info=True)


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
    if not accelerator_active:
        _warm_jax_cpu_runtime()
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

        # Set up logging. An unwritable audit location (read-only HOME) must not
        # stop every custom model from loading.
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._setup_logger()
        except OSError as e:
            logger.warning("audit log disabled, cannot write %s: %s", self.log_file, e)
            self._logger = logging.getLogger("nlsq.audit")
            if not self._logger.handlers:
                self._logger.addHandler(logging.NullHandler())

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
