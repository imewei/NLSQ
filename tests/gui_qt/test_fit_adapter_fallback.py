from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nlsq.gui_qt.adapters.fit_adapter import FitConfig, execute_fit


def test_execute_fit_auto_global_fallback():
    """Test that execute_fit falls back to 'auto' workflow when 'auto_global' has no bounds."""
    # Setup data
    xdata = np.array([1, 2, 3])
    ydata = np.array([2, 4, 6])
    model = lambda x, a: a * x

    # Config with auto_global but NO bounds
    config = FitConfig(p0=None, bounds=None, workflow="auto_global", n_starts=5)

    # Mock minpack.fit to verification
    with patch("nlsq.gui_qt.adapters.fit_adapter.fit") as mock_fit:
        # Mock result
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, key: False  # For aborted check
        mock_fit.return_value = mock_result

        # Execute
        execute_fit(xdata, ydata, None, model, config, None)

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

    # Mock minpack.fit
    with patch("nlsq.gui_qt.adapters.fit_adapter.fit") as mock_fit:
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, key: False
        mock_fit.return_value = mock_result

        execute_fit(xdata, ydata, None, model, config, None)

        _args, kwargs = mock_fit.call_args
        assert kwargs["workflow"] == "auto_global", (
            "Should keep 'auto_global' when bounds are present"
        )
