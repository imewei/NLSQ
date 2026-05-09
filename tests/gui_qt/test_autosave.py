"""Tests for AutosaveManager — restore validation, sigma=inf, atomic write."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestRestoreStateValidation:
    """Tests for _restore_state input validation in AutosaveManager."""

    def _make_manager(self, tmp_path):
        from nlsq.gui_qt.autosave import AutosaveManager

        mock_state = MagicMock()
        mock_state.state.xdata = None
        mock_state.state.ydata = None
        mock_state.state.sigma = None
        mock_state.state.model_config = None
        mock_state.state.p0 = None
        mock_state.state.bounds = None
        mock_state.state.auto_p0 = True

        manager = AutosaveManager.__new__(AutosaveManager)
        manager._app_state = mock_state
        return manager, mock_state

    def test_nan_xdata_rejected(self, tmp_path):
        """Arrays with NaN must not be restored into state."""
        manager, mock_state = self._make_manager(tmp_path)
        data = {"xdata": [1.0, float("nan"), 3.0], "ydata": [1.0, 2.0, 3.0]}

        manager._restore_state(data)

        assert mock_state.state.xdata is None

    def test_inf_xdata_rejected(self, tmp_path):
        """Arrays with Inf in xdata must not be restored (x must be finite)."""
        manager, mock_state = self._make_manager(tmp_path)
        data = {"xdata": [1.0, float("inf"), 3.0], "ydata": [1.0, 2.0, 3.0]}

        manager._restore_state(data)

        assert mock_state.state.xdata is None

    def test_valid_arrays_restored(self, tmp_path):
        """Finite, 1-D arrays must be accepted."""
        manager, mock_state = self._make_manager(tmp_path)
        data = {"xdata": [1.0, 2.0, 3.0], "ydata": [4.0, 5.0, 6.0]}

        manager._restore_state(data)

        np.testing.assert_array_equal(mock_state.state.xdata, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(mock_state.state.ydata, [4.0, 5.0, 6.0])

    def test_sigma_positive_finite_accepted(self, tmp_path):
        """Sigma with positive finite values must be accepted."""
        manager, mock_state = self._make_manager(tmp_path)
        data = {"sigma": [0.1, 0.2, 0.3]}

        manager._restore_state(data)

        np.testing.assert_array_almost_equal(mock_state.state.sigma, [0.1, 0.2, 0.3])

    def test_sigma_inf_accepted(self, tmp_path):
        """sigma=inf is the SciPy convention for 'no weight' — must be accepted."""
        manager, mock_state = self._make_manager(tmp_path)
        data = {"sigma": [float("inf"), float("inf"), float("inf")]}

        manager._restore_state(data)

        assert mock_state.state.sigma is not None
        assert np.all(np.isinf(mock_state.state.sigma))

    def test_sigma_nan_rejected(self, tmp_path):
        """NaN sigma is invalid and must be rejected."""
        manager, mock_state = self._make_manager(tmp_path)
        data = {"sigma": [0.1, float("nan"), 0.3]}

        manager._restore_state(data)

        assert mock_state.state.sigma is None

    def test_sigma_zero_rejected(self, tmp_path):
        """sigma=0 is invalid (division by zero) and must be rejected."""
        manager, mock_state = self._make_manager(tmp_path)
        data = {"sigma": [0.0, 0.1, 0.2]}

        manager._restore_state(data)

        assert mock_state.state.sigma is None

    def test_bounds_numpy_floats_restored(self, tmp_path):
        """Bounds saved as Python floats must round-trip cleanly."""
        manager, mock_state = self._make_manager(tmp_path)
        data = {"bounds": [[0.0, -1.0], [10.0, 1.0]]}

        manager._restore_state(data)

        lb, ub = mock_state.state.bounds
        assert lb == [0.0, -1.0]
        assert ub == [10.0, 1.0]


class TestAtomicWrite:
    """Tests for atomic autosave write behaviour."""

    def test_tmp_file_renamed_to_final(self, tmp_path):
        """After _do_autosave, the .tmp file must be gone and the .json present."""
        from nlsq.gui_qt.autosave import AUTOSAVE_DIR, AUTOSAVE_FILE, AutosaveManager

        mock_state = MagicMock()
        mock_state.state.xdata = np.array([1.0, 2.0])
        mock_state.state.ydata = np.array([3.0, 4.0])
        mock_state.state.sigma = None
        mock_state.state.model_config = None
        mock_state.state.p0 = None
        mock_state.state.bounds = None
        mock_state.state.auto_p0 = True

        with (
            patch("nlsq.gui_qt.autosave.AUTOSAVE_DIR", tmp_path),
            patch(
                "nlsq.gui_qt.autosave.AUTOSAVE_FILE",
                tmp_path / "session_autosave.json",
            ),
        ):
            manager = AutosaveManager.__new__(AutosaveManager)
            manager._app_state = mock_state
            manager._timer = MagicMock()
            manager.autosave_completed = MagicMock()

            manager._do_autosave()

            final = tmp_path / "session_autosave.json"
            tmp = tmp_path / "session_autosave.tmp"

            assert final.exists(), "Final autosave file must exist after save"
            assert not tmp.exists(), ".tmp file must be cleaned up after atomic rename"

            # Verify the content is valid JSON
            with open(final) as f:
                data = json.load(f)
            assert "timestamp" in data
