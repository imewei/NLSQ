"""Import-time hazards of ``import nlsq`` (each case runs in a fresh interpreter)."""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.serial


def _run(code: str, env_extra: dict | None = None, cwd=None):
    env = {**os.environ, "NLSQ_SKIP_GPU_CHECK": "1", **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize(
    "stderr_expr", ["None", "io.StringIO()"], ids=["none", "no-fileno"]
)
def test_import_with_unusable_stderr(stderr_expr):
    """AD-1: faulthandler.enable() must not abort import without a real stderr."""
    r = _run(f"import io, sys; sys.stderr = {stderr_expr}; import nlsq; print('ok')")
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_import_preserves_user_jax_cache_config(tmp_path):
    """GS-2: a cache dir configured via jax.config before import is kept."""
    user_dir = tmp_path / "userdir"
    r = _run(
        f"""
        import jax
        jax.config.update('jax_compilation_cache_dir', {str(user_dir)!r})
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 5.0)
        import nlsq
        print(jax.config.jax_compilation_cache_dir,
              jax.config.jax_persistent_cache_min_compile_time_secs)
        """,
        {"JAX_COMPILATION_CACHE_DIR": ""},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [str(user_dir), "5.0"]


def test_darwin_block_keeps_user_env():
    """GS-3: macOS defaults must not override user-set env vars."""
    r = _run(
        """
        import os, sys
        sys.platform = 'darwin'
        try:
            import nlsq
        except Exception:
            pass  # not really macOS; the env writes under test happen first
        print(os.environ['MPLBACKEND'], os.environ['OMP_NUM_THREADS'],
              os.environ['QT_API'], os.environ['PYQTGRAPH_QT_LIB'])
        """,
        {
            "MPLBACKEND": "MacOSX",
            "OMP_NUM_THREADS": "8",
            "QT_API": "pyqt6",
            "PYQTGRAPH_QT_LIB": "PyQt6",
        },
    )
    assert r.stdout.split() == ["MacOSX", "8", "pyqt6", "PyQt6"], r.stderr


def test_nlsq_debug_unwritable_log_dir(tmp_path):
    """GS-4: NLSQ_DEBUG=1 with an unwritable log dir must not crash import."""
    blocker = tmp_path / "file"
    blocker.write_text("x")  # mkdir under a regular file -> OSError, even as root
    r = _run(
        "import nlsq; print('ok')",
        {"NLSQ_DEBUG": "1", "NLSQ_LOG_DIR": str(blocker / "logs")},
        cwd=tmp_path,
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
    assert "file log disabled" in r.stderr


def test_makefile_debug_modules_imports():
    """AD-4: the modules named by `make debug-modules` must import."""
    import re
    from pathlib import Path

    mk = (Path(__file__).parents[2] / "Makefile").read_text()
    m = re.search(r"debug-modules:.*?-c \"([^\"]+)\"", mk, re.DOTALL)
    assert m, "debug-modules python snippet not found"
    r = _run(m.group(1).replace("$(PACKAGE_NAME)", "nlsq"))
    assert r.returncode == 0, r.stderr
