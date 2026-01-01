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
