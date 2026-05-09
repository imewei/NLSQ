from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import nlsq.gui_qt.adapters.fit_adapter as fit_adapter_module
from nlsq.gui_qt.adapters.fit_adapter import FitConfig, execute_fit


def test_execute_fit_auto_global_fallback():
    """Test that execute_fit falls back to 'auto' workflow when 'auto_global' has no bounds."""
    # Setup data
    xdata = np.array([1, 2, 3])
    ydata = np.array([2, 4, 6])
    model = lambda x, a: a * x

    # Config with auto_global but NO bounds
    config = FitConfig(p0=None, bounds=None, workflow="auto_global", n_starts=5)

    # Mock minpack.fit using patch.object for more reliable patching
    # This patches the 'fit' attribute on the module object directly
    with patch.object(fit_adapter_module, "fit") as mock_fit:
        # Mock result
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, key: False  # For aborted check
        mock_fit.return_value = mock_result

        # Execute
        execute_fit(xdata, ydata, None, model, config, None)

        # Verify mock was actually called (catch flaky test issues early)
        assert mock_fit.called, (
            "Mock fit() was not called - possible import caching issue"
        )

        # Verify call args
        _args, kwargs = mock_fit.call_args

        # Check that workflow was changed to "auto"
        assert kwargs["workflow"] == "auto", (
            "Should downgrade to 'auto' when bounds are missing"
        )

        # Verify bounds passed are effectively infinite (default for None in config)
        bounds_passed = kwargs["bounds"]
        assert bounds_passed[0] == -float("inf")
        assert bounds_passed[1] == float("inf")


def test_execute_fit_auto_global_with_bounds_preserved():
    """Test that execute_fit keeps 'auto_global' workflow when bounds ARE provided."""
    # Setup data
    xdata = np.array([1, 2, 3])
    ydata = np.array([2, 4, 6])
    model = lambda x, a: a * x

    # Config with auto_global AND bounds
    config = FitConfig(
        p0=None, bounds=([-10], [10]), workflow="auto_global", n_starts=5
    )

    # Mock minpack.fit using patch.object for more reliable patching
    with patch.object(fit_adapter_module, "fit") as mock_fit:
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, key: False
        mock_fit.return_value = mock_result

        execute_fit(xdata, ydata, None, model, config, None)

        # Verify mock was actually called (catch flaky test issues early)
        assert mock_fit.called, (
            "Mock fit() was not called - possible import caching issue"
        )

        _args, kwargs = mock_fit.call_args
        assert kwargs["workflow"] == "auto_global", (
            "Should keep 'auto_global' when bounds are present"
        )


class TestSnapshotForFit:
    """Tests for SessionState.snapshot_for_fit() deep-copy guarantee."""

    def _make_state(self):
        from nlsq.gui_qt.session_state import SessionState

        s = SessionState()
        s.xdata = np.array([1.0, 2.0, 3.0])
        s.ydata = np.array([2.0, 4.0, 6.0])
        s.sigma = np.array([0.1, 0.1, 0.1])
        s.p0 = [1.0, 0.0]
        s.bounds = ([0.0, -1.0], [10.0, 1.0])
        s.model_config = {"name": "linear", "params": [1]}
        return s

    def test_snapshot_arrays_are_independent_copies(self):
        """Mutating original arrays after snapshot must not affect snapshot."""
        state = self._make_state()
        snap = state.snapshot_for_fit()

        original_x = snap.xdata.copy()
        state.xdata[0] = 999.0  # mutate original in-place

        np.testing.assert_array_equal(snap.xdata, original_x)

    def test_snapshot_model_config_is_deep_copy(self):
        """Mutating model_config after snapshot must not affect snapshot."""
        state = self._make_state()
        snap = state.snapshot_for_fit()

        state.model_config["name"] = "changed"
        assert snap.model_config["name"] == "linear"

    def test_snapshot_p0_is_independent_list(self):
        state = self._make_state()
        snap = state.snapshot_for_fit()

        state.p0[0] = 999.0
        assert snap.p0[0] == 1.0

    def test_snapshot_bounds_are_independent_lists(self):
        state = self._make_state()
        snap = state.snapshot_for_fit()

        state.bounds[0][0] = 999.0
        assert snap.bounds[0][0] == 0.0

    def test_snapshot_clears_fit_result(self):
        """fit_result is not needed by the worker — snapshot must be None."""
        state = self._make_state()
        state.fit_result = {"popt": np.array([1.0, 2.0]), "pcov": np.eye(2)}
        snap = state.snapshot_for_fit()

        assert snap.fit_result is None

    def test_snapshot_sigma_none_stays_none(self):
        state = self._make_state()
        state.sigma = None
        snap = state.snapshot_for_fit()

        assert snap.sigma is None


class TestPendingThreadsDeferred:
    """Smoke tests for _pending_threads deferred cleanup in FittingOptionsPage.

    Uses __new__ + MagicMock to bypass Qt widget construction entirely,
    so these run without a QApplication and without @pytest.mark.serial.
    """

    def _make_page_stub(self):
        from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

        page = FittingOptionsPage.__new__(FittingOptionsPage)
        page._pending_threads = set()
        return page

    def test_thread_added_to_pending_on_timeout(self):
        """When wait(100) times out, the thread ref must be in _pending_threads."""
        from unittest.mock import MagicMock

        page = self._make_page_stub()
        mock_thread = MagicMock()
        mock_thread.wait.return_value = False  # simulate timeout — thread still running

        page._fit_thread = mock_thread
        page._fit_worker = MagicMock()

        page._cleanup_thread()

        assert mock_thread in page._pending_threads, (
            "_pending_threads must hold the thread ref until finished fires"
        )

    def test_fit_thread_ref_cleared_while_thread_still_pending(self):
        """_fit_thread / _fit_worker must be None after deferred cleanup
        so that run_fit() can start a new fit immediately."""
        from unittest.mock import MagicMock

        page = self._make_page_stub()
        mock_thread = MagicMock()
        mock_thread.wait.return_value = False

        page._fit_thread = mock_thread
        page._fit_worker = MagicMock()

        page._cleanup_thread()

        assert page._fit_thread is None
        assert page._fit_worker is None

    def test_thread_removed_from_pending_when_finished_fires(self):
        """Calling _deferred_delete (via thread.finished) must remove the ref
        from _pending_threads — the Python strong reference is released."""
        from unittest.mock import MagicMock

        page = self._make_page_stub()
        mock_thread = MagicMock()
        mock_thread.wait.return_value = False

        page._fit_thread = mock_thread
        page._fit_worker = MagicMock()

        # Capture the slot that gets connected to thread.finished
        connected_slot = None

        def capture_connect(slot):
            nonlocal connected_slot
            connected_slot = slot

        mock_thread.finished.connect.side_effect = capture_connect

        page._cleanup_thread()

        assert mock_thread in page._pending_threads
        assert connected_slot is not None, "A slot must be connected to thread.finished"

        # Simulate the C++ thread finishing → fires thread.finished → calls slot
        connected_slot()

        assert mock_thread not in page._pending_threads, (
            "_pending_threads must be empty after _deferred_delete fires"
        )

    def test_double_fire_is_safe(self):
        """Calling _deferred_delete twice (double-connection edge case) must not crash."""
        from unittest.mock import MagicMock

        page = self._make_page_stub()
        mock_thread = MagicMock()
        mock_thread.wait.return_value = False

        page._fit_thread = mock_thread
        page._fit_worker = MagicMock()

        connected_slot = None

        def capture_connect(slot):
            nonlocal connected_slot
            connected_slot = slot

        mock_thread.finished.connect.side_effect = capture_connect

        page._cleanup_thread()

        # Fire twice — must not raise
        connected_slot()
        connected_slot()  # second call after discard — safe no-op

        assert mock_thread not in page._pending_threads
