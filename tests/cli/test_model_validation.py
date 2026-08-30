"""Tests for model file security validation.

This module tests the security validation for custom model files loaded
through the NLSQ CLI, ensuring dangerous patterns are blocked and audit
logging works correctly.
"""

import tempfile
from pathlib import Path

import pytest

from nlsq.cli.model_validation import (
    DANGEROUS_MODULES,
    DANGEROUS_PATTERNS,
    DangerousPatternVisitor,
    ModelValidationResult,
    ResourceLimitError,
    resource_limits,
    validate_model,
    validate_path,
)


class TestDangerousPatterns:
    """Test that dangerous patterns are correctly detected."""

    def test_dangerous_patterns_frozen(self):
        """DANGEROUS_PATTERNS should be immutable."""
        assert isinstance(DANGEROUS_PATTERNS, frozenset)

    def test_dangerous_modules_frozen(self):
        """DANGEROUS_MODULES should be immutable."""
        assert isinstance(DANGEROUS_MODULES, frozenset)

    def test_exec_is_dangerous(self):
        """exec() should be in dangerous patterns."""
        assert "exec" in DANGEROUS_PATTERNS

    def test_eval_is_dangerous(self):
        """eval() should be in dangerous patterns."""
        assert "eval" in DANGEROUS_PATTERNS

    def test_subprocess_is_dangerous(self):
        """subprocess should be in dangerous modules."""
        assert "subprocess" in DANGEROUS_MODULES

    def test_os_is_dangerous(self):
        """os should be in dangerous modules."""
        assert "os" in DANGEROUS_MODULES

    def test_socket_is_dangerous(self):
        """socket should be in dangerous patterns and modules."""
        assert "socket" in DANGEROUS_PATTERNS
        assert "socket" in DANGEROUS_MODULES


class TestValidateModel:
    """Test model validation function."""

    def test_valid_model_passes(self, tmp_path: Path):
        """A safe model file should pass validation."""
        model_file = tmp_path / "safe_model.py"
        model_file.write_text("""
import jax.numpy as jnp

def model(x, a, b):
    return a * jnp.exp(-b * x)

def estimate_p0(xdata, ydata):
    return [1.0, 0.1]
""")

        result = validate_model(model_file)

        assert result.is_valid
        assert result.violations == []
        assert result.path == model_file

    def test_exec_blocked(self, tmp_path: Path):
        """Model with exec() should be blocked."""
        model_file = tmp_path / "malicious_exec.py"
        model_file.write_text("""
def model(x, a):
    exec("import os; os.system('rm -rf /')")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("exec" in v for v in result.violations)

    def test_eval_blocked(self, tmp_path: Path):
        """Model with eval() should be blocked."""
        model_file = tmp_path / "malicious_eval.py"
        model_file.write_text("""
def model(x, a):
    return eval("a * x")
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("eval" in v for v in result.violations)

    def test_os_system_blocked(self, tmp_path: Path):
        """Model with os.system() should be blocked."""
        model_file = tmp_path / "malicious_os.py"
        model_file.write_text("""
import os

def model(x, a):
    os.system("echo pwned")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        # Should flag both the import and the function call
        assert any("os" in v.lower() for v in result.violations)

    def test_subprocess_import_blocked(self, tmp_path: Path):
        """Model with subprocess import should be blocked."""
        model_file = tmp_path / "malicious_subprocess.py"
        model_file.write_text("""
import subprocess

def model(x, a):
    subprocess.run(["echo", "pwned"])
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("subprocess" in v for v in result.violations)

    def test_file_write_blocked(self, tmp_path: Path):
        """Model with file write operations should be blocked."""
        model_file = tmp_path / "malicious_write.py"
        model_file.write_text("""
def model(x, a):
    with open("malicious.txt", "w") as f:
        f.write("pwned")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("write" in v.lower() for v in result.violations)

    def test_socket_blocked(self, tmp_path: Path):
        """Model with socket operations should be blocked."""
        model_file = tmp_path / "malicious_socket.py"
        model_file.write_text("""
import socket

def model(x, a):
    s = socket.socket()
    s.connect(("evil.com", 80))
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("socket" in v for v in result.violations)

    def test_ctypes_blocked(self, tmp_path: Path):
        """Model with ctypes should be blocked."""
        model_file = tmp_path / "malicious_ctypes.py"
        model_file.write_text("""
import ctypes

def model(x, a):
    libc = ctypes.CDLL("libc.so.6")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("ctypes" in v for v in result.violations)

    def test_aliased_importlib_import_module_blocked(self, tmp_path: Path):
        """Aliasing importlib.import_module must not bypass detection.

        Regression test for a sandbox-escape hole: aliasing the imported
        symbol (`as im`) rebinds it to a name absent from DANGEROUS_PATTERNS,
        so only checking the module root let this through undetected.
        """
        model_file = tmp_path / "malicious_importlib_alias.py"
        model_file.write_text("""
from importlib import import_module as im

def model(x, a):
    im("subprocess").run(["echo", "pwned"])
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("importlib" in v.lower() for v in result.violations)

    def test_aliased_builtins_eval_blocked(self, tmp_path: Path):
        """Aliasing builtins.eval must not bypass detection."""
        model_file = tmp_path / "malicious_builtins_alias.py"
        model_file.write_text("""
from builtins import eval as ev

def model(x, a):
    ev("__import__('os').system('pwned')")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("builtins" in v.lower() for v in result.violations)

    def test_bare_dunder_import_blocked(self, tmp_path: Path):
        """Bare __import__() (no explicit import statement) must be blocked."""
        model_file = tmp_path / "malicious_dunder_import.py"
        model_file.write_text("""
def model(x, a):
    __import__("os").system("pwned")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("__import__" in v for v in result.violations)

    def test_dotted_submodule_import_blocked(self, tmp_path: Path):
        """`import os.path as p` must be blocked via the module root."""
        model_file = tmp_path / "malicious_dotted_import.py"
        model_file.write_text("""
import os.path as p

def model(x, a):
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("os.path" in v for v in result.violations)

    def test_generator_frame_builtins_blocked(self, tmp_path: Path):
        """gi_frame/f_builtins must be blocked: reaches exec() via a generator
        frame instead of a name-based check on exec/eval/__builtins__ itself."""
        model_file = tmp_path / "malicious_gi_frame.py"
        model_file.write_text("""
def model(x, a):
    gen = (i for i in [1])
    b = gen.gi_frame.f_builtins
    b["exec"]("import os; os.system('pwned')")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("gi_frame" in v for v in result.violations)
        assert any("f_builtins" in v for v in result.violations)

    def test_pkgutil_resolve_name_blocked(self, tmp_path: Path):
        """pkgutil.resolve_name resolves a dotted string path without an import."""
        model_file = tmp_path / "malicious_pkgutil.py"
        model_file.write_text("""
import pkgutil

def model(x, a):
    pkgutil.resolve_name("posix:system")("pwned")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("pkgutil" in v.lower() for v in result.violations)

    def test_aliased_pkgutil_resolve_name_blocked(self, tmp_path: Path):
        """Aliasing pkgutil.resolve_name must not bypass detection -- this is
        the actual bypass shape the `resolve_name` DANGEROUS_PATTERNS entry
        exists for; a bare `import pkgutil` block alone wouldn't catch this."""
        model_file = tmp_path / "malicious_pkgutil_alias.py"
        model_file.write_text("""
from pkgutil import resolve_name as r

def model(x, a):
    r("posix:system")("pwned")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("resolve_name" in v for v in result.violations)

    def test_timeit_code_string_blocked(self, tmp_path: Path):
        """timeit.timeit() executes an arbitrary code string."""
        model_file = tmp_path / "malicious_timeit.py"
        model_file.write_text("""
import timeit

def model(x, a):
    timeit.timeit("import os; os.system('pwned')", number=1)
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("timeit" in v.lower() for v in result.violations)

    def test_pathlib_write_text_blocked(self, tmp_path: Path):
        """Path.write_text() must be blocked: mutates files via a pathlib
        method call instead of a name-based check on open()/os/shutil."""
        model_file = tmp_path / "malicious_pathlib_write.py"
        model_file.write_text("""
from pathlib import Path

def model(x, a):
    Path("/tmp/pwned.txt").write_text("arbitrary write")
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("write_text" in v for v in result.violations)

    @pytest.mark.parametrize(
        ("source", "expected_substring"),
        [
            pytest.param(
                "from pathlib import Path\n\ndef model(x, a):\n"
                '    Path("/tmp/p.txt").write_bytes(b"pwned")\n    return a * x\n',
                "write_bytes",
                id="pathlib-write_bytes",
            ),
            pytest.param(
                "import asyncio\n\ndef model(x, a):\n    return a * x\n",
                "asyncio",
                id="asyncio-module-import",
            ),
            pytest.param(
                "import asyncio\n\nasync def model(x, a):\n"
                '    await asyncio.create_subprocess_shell("echo pwned")\n'
                "    return a * x\n",
                "create_subprocess_shell",
                id="asyncio-create_subprocess_shell",
            ),
            pytest.param(
                "import asyncio\n\nasync def model(x, a):\n"
                '    await asyncio.create_subprocess_exec("echo", "pwned")\n'
                "    return a * x\n",
                "create_subprocess_exec",
                id="asyncio-create_subprocess_exec",
            ),
            pytest.param(
                "import sys\n\ndef model(x, a):\n"
                "    tb = sys.exc_info()[2]\n"
                '    tb.tb_frame.f_builtins["exec"]("pwned")\n'
                "    return a * x\n",
                "tb_frame",
                id="traceback-tb_frame",
            ),
        ],
    )
    def test_remaining_denylist_entries_blocked(
        self,
        tmp_path: Path,
        source: str,
        expected_substring: str,
    ):
        """Every denylist entry added in this bypass-closing pass must be
        independently pinned -- closes a coverage gap where 5 of 11 new
        entries (tb_frame, create_subprocess_shell/exec, write_bytes,
        asyncio) had no dedicated test despite being claimed as fixed."""
        model_file = tmp_path / "malicious_remaining.py"
        model_file.write_text(source)

        result = validate_model(model_file)

        assert not result.is_valid
        assert any(expected_substring in v for v in result.violations)

    def test_coroutine_frame_builtins_blocked(self, tmp_path: Path):
        """cr_frame/f_builtins: same technique as gi_frame, via a coroutine
        object instead of a generator. Not covered by banning asyncio, since
        cr_frame is a built-in coroutine-object attribute."""
        model_file = tmp_path / "malicious_cr_frame.py"
        model_file.write_text("""
async def _coro():
    pass

def model(x, a):
    c = _coro()
    b = c.cr_frame.f_builtins
    b["exec"]("import os; os.system('pwned')")
    c.close()
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("cr_frame" in v for v in result.violations)

    def test_pathlib_unlink_blocked(self, tmp_path: Path):
        """Path.unlink() deletes files -- same bypass class as write_text,
        adjacent destructive pathlib method not covered by the open() check."""
        model_file = tmp_path / "malicious_pathlib_unlink.py"
        model_file.write_text("""
from pathlib import Path

def model(x, a):
    Path("/tmp/important.txt").unlink()
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("unlink" in v for v in result.violations)

    def test_trusted_bypasses_validation(self, tmp_path: Path):
        """trusted=True should bypass validation."""
        model_file = tmp_path / "malicious_but_trusted.py"
        model_file.write_text("""
import os

def model(x, a):
    os.system("echo trusted")
    return a * x
""")

        result = validate_model(model_file, trusted=True)

        # Even with violations, trusted models are marked valid
        assert result.is_trusted
        # Violations are still recorded for audit logging
        assert len(result.violations) > 0

    def test_nonexistent_file(self, tmp_path: Path):
        """Non-existent file should fail validation."""
        model_file = tmp_path / "nonexistent.py"

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("not exist" in v.lower() for v in result.violations)

    def test_syntax_error(self, tmp_path: Path):
        """File with syntax error should fail validation."""
        model_file = tmp_path / "syntax_error.py"
        model_file.write_text("""
def model(x, a):
    return a * x
    # Missing closing parenthesis
    print(
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("syntax" in v.lower() for v in result.violations)

    def test_non_python_extension_warning(self, tmp_path: Path):
        """Non-.py extension should add a violation."""
        model_file = tmp_path / "model.txt"
        model_file.write_text("""
def model(x, a):
    return a * x
""")

        result = validate_model(model_file)

        assert not result.is_valid
        assert any("extension" in v.lower() for v in result.violations)


class TestValidatePath:
    """Test path traversal prevention."""

    def test_relative_path_in_cwd(self, tmp_path: Path, monkeypatch):
        """Relative path within cwd should be valid."""
        monkeypatch.chdir(tmp_path)

        model_file = tmp_path / "model.py"
        model_file.touch()

        assert validate_path(Path("model.py"))

    def test_absolute_path_in_cwd(self, tmp_path: Path, monkeypatch):
        """Absolute path within cwd should be valid."""
        monkeypatch.chdir(tmp_path)

        model_file = tmp_path / "model.py"
        model_file.touch()

        assert validate_path(model_file)

    def test_subdirectory_path_valid(self, tmp_path: Path, monkeypatch):
        """Path in subdirectory should be valid."""
        monkeypatch.chdir(tmp_path)

        subdir = tmp_path / "models"
        subdir.mkdir()
        model_file = subdir / "model.py"
        model_file.touch()

        assert validate_path(Path("models/model.py"))

    def test_parent_traversal_blocked(self, tmp_path: Path, monkeypatch):
        """Parent directory traversal should be blocked."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        # Try to access parent directory
        assert not validate_path(Path("../etc/passwd"))

    def test_absolute_outside_cwd_blocked(self, tmp_path: Path, monkeypatch):
        """Absolute path outside cwd should be blocked."""
        monkeypatch.chdir(tmp_path)

        # /etc/passwd is outside any normal project directory
        assert not validate_path(Path("/etc/passwd"))

    def test_symlink_outside_blocked(self, tmp_path: Path, monkeypatch):
        """Symlink pointing outside cwd should be blocked."""
        monkeypatch.chdir(tmp_path)

        # Create a symlink pointing outside
        link = tmp_path / "evil_link.py"
        try:
            link.symlink_to("/etc/passwd")
            assert not validate_path(link)
        except OSError:
            # Skip if symlink creation not supported
            pytest.skip("Symlink creation not supported")

    def test_custom_base_dir(self, tmp_path: Path):
        """Custom base_dir should be respected."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        model_file = models_dir / "model.py"
        model_file.touch()

        # Path is valid relative to models_dir
        assert validate_path(model_file, base_dir=models_dir)

        # Path outside models_dir should be blocked
        other_file = tmp_path / "other.py"
        other_file.touch()
        assert not validate_path(other_file, base_dir=models_dir)


class TestResourceLimits:
    """Test resource limit context manager.

    Note: Memory limit tests are skipped because setting RLIMIT_AS to
    a restrictive value can crash the Python interpreter, especially
    when JAX is loaded and uses significant virtual address space.
    """

    def test_context_manager_exists(self):
        """Verify resource_limits is importable and callable."""
        assert callable(resource_limits)

    def test_resource_limit_error_exists(self):
        """Verify ResourceLimitError is defined."""
        assert issubclass(ResourceLimitError, Exception)


class TestModelValidationResult:
    """Test ModelValidationResult dataclass."""

    def test_valid_result(self, tmp_path: Path):
        """Test a valid result."""
        result = ModelValidationResult(
            path=tmp_path / "model.py",
            is_valid=True,
            is_trusted=False,
            violations=[],
        )

        assert result.is_valid
        assert not result.is_trusted
        assert result.violations == []

    def test_invalid_result_with_violations(self, tmp_path: Path):
        """Test an invalid result with violations."""
        violations = ["Dangerous function: exec", "Dangerous import: os"]
        result = ModelValidationResult(
            path=tmp_path / "model.py",
            is_valid=False,
            is_trusted=False,
            violations=violations,
        )

        assert not result.is_valid
        assert result.violations == violations

    def test_trusted_result(self, tmp_path: Path):
        """Test a trusted result (bypasses validation)."""
        result = ModelValidationResult(
            path=tmp_path / "model.py",
            is_valid=False,  # Would be invalid
            is_trusted=True,  # But trusted
            violations=["Dangerous function: exec"],
        )

        assert not result.is_valid
        assert result.is_trusted


class TestDangerousPatternVisitor:
    """Test the AST visitor for dangerous patterns."""

    def test_detects_exec_name(self):
        """Visitor should detect exec as a name reference."""
        import ast

        source = "x = exec"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) >= 1
        assert any("exec" in v for v in visitor.violations)

    def test_detects_eval_call(self):
        """Visitor should detect eval() call."""
        import ast

        source = "result = eval('1 + 2')"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) >= 1
        assert any("eval" in v for v in visitor.violations)

    def test_detects_method_call(self):
        """Visitor should detect dangerous method calls like os.system()."""
        import ast

        source = "os.system('echo')"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) >= 1
        assert any("system" in v for v in visitor.violations)

    def test_detects_import(self):
        """Visitor should detect dangerous imports."""
        import ast

        source = "import subprocess"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) >= 1
        assert any("subprocess" in v for v in visitor.violations)

    def test_detects_from_import(self):
        """Visitor should detect dangerous from...import statements."""
        import ast

        source = "from os import system"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) >= 1
        assert any("os" in v for v in visitor.violations)

    def test_detects_file_write_mode(self):
        """Visitor should detect open() with write mode."""
        import ast

        source = "open('file.txt', 'w')"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert len(visitor.violations) >= 1
        assert any("write" in v.lower() for v in visitor.violations)

    def test_allows_file_read_mode(self):
        """Visitor should allow open() with read mode."""
        import ast

        source = "open('file.txt', 'r')"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        # Read mode should not add violations
        assert not any("write" in v.lower() for v in visitor.violations)

    def test_safe_code_no_violations(self):
        """Safe code should have no violations."""
        import ast

        source = """
import jax.numpy as jnp
import numpy as np

def model(x, a, b):
    return a * jnp.exp(-b * x)

def estimate_p0(xdata, ydata):
    return [np.max(ydata), 0.1]
"""
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert visitor.violations == []

    def test_detects_sys_modules_bypass(self):
        """sys.modules dict-access must not bypass the os/subprocess check.

        Regression for a three-brain-review finding: previously neither
        `sys` nor `modules` was in the blocklist, so `sys.modules["os"]`
        reached an already-imported dangerous module without a literal
        `import os` for visit_Import to catch.
        """
        import ast

        source = """
import sys
def f(x, a):
    sys.modules["os"].remove("/tmp/whatever")
    return a * x
"""
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("modules" in v for v in visitor.violations)

    def test_detects_format_string_dunder_escape(self):
        """A dunder chain hidden inside a format-spec string must be caught.

        Regression for a three-brain-review finding: `"{0.__class__...}"
        .format(x)` resolves the chain at runtime from string data, never
        producing an ast.Attribute node for visit_Attribute to see.
        """
        import ast

        source = (
            'template = "{0.__class__.__bases__[0].__subclasses__}"\n'
            "result = template.format(x)\n"
        )
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("__class__" in v for v in visitor.violations)

    def test_detects_operator_attrgetter_bypass(self):
        """operator.attrgetter is a string-mediated getattr equivalent."""
        import ast

        source = """
import operator
getter = operator.attrgetter("system")
"""
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("attrgetter" in v for v in visitor.violations)

    def test_detects_pydoc_locate_bypass(self):
        """pydoc.locate is a string-mediated import/getattr equivalent."""
        import ast

        source = """
import pydoc
system = pydoc.locate("os.system")
"""
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("locate" in v for v in visitor.violations)

    def test_detects_types_module_bypass(self):
        """types module can construct/run arbitrary bytecode, bypassing exec/eval."""
        import ast

        source = "import types\nf = types.FunctionType\n"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("types" in v for v in visitor.violations)

    def test_detects_sys_module_import(self):
        """`import sys` alone must be blocked (frame/stack introspection
        surface)."""
        import ast

        source = "import sys\n"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("sys" in v.lower() for v in visitor.violations)

    def test_detects_getframe_attribute_pattern_independent_of_sys_import(self):
        """The `_getframe` attribute-pattern fix must be caught on its own,
        not merely as a side effect of the separate `sys` module block --
        this source never imports sys, isolating the two."""
        import ast

        source = "frame = obj._getframe()\n"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("_getframe" in v for v in visitor.violations)

    def test_detects_type_builtin(self):
        """type() reaches the same sandbox-escape surface as __class__."""
        import ast

        source = "t = type(1)\n"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("type" in v for v in visitor.violations)

    def test_detects_setattr_delattr_hasattr(self):
        """setattr/delattr/hasattr are getattr's blocked siblings."""
        import ast

        source = "setattr(obj, 'x', 1)\ndelattr(obj, 'x')\nhasattr(obj, 'x')\n"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        found = set(visitor.violations)
        assert any("setattr" in v for v in found)
        assert any("delattr" in v for v in found)
        assert any("hasattr" in v for v in found)

    def test_detects_open_read_write_plus_mode(self):
        """open(..., 'r+') is write-capable and must not bypass the w/a/x check."""
        import ast

        source = "open('file.txt', 'r+')\n"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("write" in v.lower() for v in visitor.violations)

    def test_detects_format_call_on_concatenated_string(self):
        """Concatenated string literals defeat the substring scan; block .format() itself."""
        import ast

        source = "template = '{0.__cla' + 'ss__}'\nresult = template.format(x)\n"
        tree = ast.parse(source)

        visitor = DangerousPatternVisitor()
        visitor.visit(tree)

        assert any("format" in v for v in visitor.violations)
