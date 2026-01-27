"""
Smoke tests for GUI startup on macOS.

Validates that the SIGBUS-prevention hotfixes are in place and that
the Qt application can initialize without crashing.  These tests
run on all platforms but specifically guard against regressions
that caused SIGBUS on macOS (see RCA: OpenGL/Metal conflict,
matplotlib backend conflict, JAX Metal backend conflict).
"""

from __future__ import annotations

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# 1. Import-order guards (no Qt/GUI imports at module level)
# ---------------------------------------------------------------------------


class TestImportGuards:
    """Verify defensive environment variables are set before JAX loads."""

    def test_jax_cpu_enforcement_on_non_linux(self):
        """nlsq.__init__ must set JAX CPU vars on non-Linux platforms."""
        if sys.platform == "linux":
            pytest.skip("Non-Linux guard (macOS/Windows only)")

        # Importing nlsq triggers the hotfix
        import nlsq

        assert os.environ.get("JAX_PLATFORM_NAME") == "cpu"
        assert os.environ.get("JAX_PLATFORMS") == "cpu"
        assert os.environ.get("NLSQ_FORCE_CPU") == "1"

    def test_macos_specific_env_vars(self):
        """nlsq.__init__ must set macOS-specific stability vars on Darwin."""
        if sys.platform != "darwin":
            pytest.skip("macOS-only guard")

        import nlsq

        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("MPLBACKEND") == "Agg"
        assert os.environ.get("QT_MAC_WANTS_LAYER") == "1"
        assert os.environ.get("QT_API") == "pyside6"
        assert os.environ.get("PYQTGRAPH_QT_LIB") == "PySide6"
        assert os.environ.get("QT_OPENGL") == "software"
        assert os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1"

    def test_jax_backend_is_cpu(self):
        """JAX must resolve to CPU backend in the test environment."""
        import jax

        assert jax.default_backend() == "cpu"

    def test_matplotlib_agg_backend_after_gui_import(self):
        """gui_qt must force matplotlib to the Agg backend."""
        import matplotlib

        from nlsq.gui_qt import run_desktop

        assert matplotlib.get_backend().lower() == "agg"

    def test_pyqtgraph_opengl_disabled_on_macos(self, qtbot):
        """pyqtgraph must default to useOpenGL=False on macOS."""
        if sys.platform != "darwin":
            pytest.skip("macOS-only guard")

        from nlsq.gui_qt.plots import _configure_pyqtgraph, _pg_state

        _pg_state["configured"] = False
        _configure_pyqtgraph()

        import pyqtgraph as pg

        assert pg.getConfigOption("useOpenGL") is False

    def test_pyqtgraph_opengl_env_override(self, qtbot, monkeypatch):
        """NLSQ_GUI_USE_OPENGL=1 must force OpenGL on."""
        monkeypatch.setenv("NLSQ_GUI_USE_OPENGL", "1")

        from nlsq.gui_qt.plots import _configure_pyqtgraph, _pg_state

        _pg_state["configured"] = False
        _configure_pyqtgraph()

        import pyqtgraph as pg

        assert pg.getConfigOption("useOpenGL") is True

        # Reset so other tests aren't affected
        _pg_state["configured"] = False
        pg.setConfigOptions(useOpenGL=False)

    def test_pyqtgraph_configured_on_package_import(self, qtbot):
        """pyqtgraph must be configured eagerly when the plots package loads.

        Direct submodule imports (e.g. ``from nlsq.gui_qt.plots.fit_plot import ...``)
        bypass ``__getattr__``, so the guard must also run at module level.
        Verify by re-importing in a subprocess to avoid cross-test state leaks.
        """
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from nlsq.gui_qt.plots import _pg_state; "
                "assert _pg_state['configured'] is True, "
                "'pyqtgraph not configured at package load time'",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (
            f"pyqtgraph not eagerly configured on import: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# 2. Application startup smoke test
# ---------------------------------------------------------------------------


class TestApplicationStartup:
    """Verify the GUI can initialize without SIGBUS."""

    def test_qapplication_creation(self, qtbot):
        """QApplication must be creatable."""
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        assert app is not None

    def test_app_state_creation(self, qtbot):
        """AppState must initialize without error."""
        from nlsq.gui_qt.app_state import AppState

        state = AppState()
        assert state is not None

    def test_main_window_creation(self, qtbot):
        """MainWindow must initialize without error (the SIGBUS crash site)."""
        from nlsq.gui_qt.app_state import AppState
        from nlsq.gui_qt.main_window import MainWindow

        state = AppState()
        window = MainWindow(state)
        qtbot.addWidget(window)
        assert window is not None


# ---------------------------------------------------------------------------
# 3. Exception hook integrity
# ---------------------------------------------------------------------------


class TestExceptionHook:
    """Verify the exception hook formats messages correctly."""

    def test_exception_hook_newlines_not_escaped(self):
        """_exception_hook must use real newlines, not escaped \\n literals."""
        import inspect

        from nlsq.gui_qt import _exception_hook

        source = inspect.getsource(_exception_hook)

        # The source should contain \n (in f-strings) but NOT \\n
        # We check the actual string literals in the source code
        assert '\\\\n' not in source, (
            "_exception_hook contains double-escaped newlines (\\\\n). "
            "These render as literal backslash-n instead of actual newlines."
        )
