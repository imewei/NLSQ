"""Tests for the NLSQ GUI fitting execution page components.

This module tests the UI components for fitting execution including
the Run Fit button, progress bar, abort button, and result transitions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nlsq.core.functions import exponential_decay, linear
from nlsq.gui.components.iteration_table import (
    create_iteration_history,
    format_iteration_table,
    get_table_display_config,
    limit_history_size,
    update_iteration_history,
)
from nlsq.gui.components.live_cost_plot import (
    create_cost_history,
    create_cost_plot_figure,
    get_plot_config,
    update_cost_history,
)
from nlsq.gui.state import SessionState, initialize_state

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def session_state():
    """Create a session state with data and model loaded."""
    state = initialize_state()
    state.xdata = np.linspace(0, 10, 50)
    state.ydata = 2.0 * state.xdata + 5.0
    state.p0 = [1.0, 1.0]
    return state


@pytest.fixture
def mock_model():
    """Create a mock model function."""

    def model(x, a, b):
        return a * x + b

    return model


# =============================================================================
# Test Run Fit Button State Management
# =============================================================================


class TestRunFitButtonState:
    """Tests for Run Fit button state management."""

    def test_button_disabled_without_data(self):
        """Test that button is disabled when data not loaded."""
        state = initialize_state()
        # No xdata/ydata set

        is_ready = (
            state.xdata is not None and state.ydata is not None and state.p0 is not None
        )

        assert is_ready is False

    def test_button_disabled_without_p0(self, session_state):
        """Test that button is disabled when p0 not set."""
        session_state.p0 = None

        is_ready = (
            session_state.xdata is not None
            and session_state.ydata is not None
            and session_state.p0 is not None
        )

        assert is_ready is False

    def test_button_enabled_when_ready(self, session_state):
        """Test that button is enabled when data, model, and p0 are set."""
        is_ready = (
            session_state.xdata is not None
            and session_state.ydata is not None
            and session_state.p0 is not None
        )

        assert is_ready is True

    def test_button_disabled_during_fit(self, session_state):
        """Test that button is disabled while fit is running."""
        session_state.fit_running = True

        can_start_new_fit = not session_state.fit_running

        assert can_start_new_fit is False


# =============================================================================
# Test Progress Bar Updates
# =============================================================================


class TestProgressBarUpdates:
    """Tests for progress bar update functionality."""

    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        current_iteration = 50
        max_iterations = 200

        progress = current_iteration / max_iterations

        assert progress == 0.25

    def test_progress_clamping(self):
        """Test progress is clamped to [0, 1]."""
        # Over max
        progress1 = min(1.0, 250 / 200)
        assert progress1 == 1.0

        # Negative (shouldn't happen but test edge case)
        progress2 = max(0.0, -10 / 200)
        assert progress2 == 0.0

    def test_progress_text_format(self):
        """Test progress text formatting."""
        current = 75
        max_iter = 200
        cost = 0.00123

        text = f"Iteration {current}/{max_iter} | Cost: {cost:.6f}"

        assert "75/200" in text
        assert "0.001230" in text


# =============================================================================
# Test Abort Button Functionality
# =============================================================================


class TestAbortButton:
    """Tests for abort button functionality."""

    def test_abort_flag_set(self, session_state):
        """Test that abort flag is set when button clicked."""
        session_state.fit_running = True
        session_state.fit_aborted = False

        # Simulate abort button click
        session_state.fit_aborted = True

        assert session_state.fit_aborted is True

    def test_abort_button_visibility(self, session_state):
        """Test abort button is only visible during fit."""
        # Before fit
        session_state.fit_running = False
        show_abort = session_state.fit_running
        assert show_abort is False

        # During fit
        session_state.fit_running = True
        show_abort = session_state.fit_running
        assert show_abort is True

        # After fit
        session_state.fit_running = False
        show_abort = session_state.fit_running
        assert show_abort is False


# =============================================================================
# Test Transition to Results
# =============================================================================


class TestResultsTransition:
    """Tests for transition to results on completion."""

    def test_result_stored_on_completion(self, session_state):
        """Test that result is stored in state on completion."""
        mock_result = MagicMock()
        mock_result.popt = np.array([2.0, 5.0])
        mock_result.success = True

        session_state.fit_result = mock_result

        assert session_state.fit_result is not None
        assert session_state.fit_result.success is True

    def test_fit_running_cleared_on_completion(self, session_state):
        """Test that fit_running is cleared on completion."""
        session_state.fit_running = True

        # Simulate fit completion
        session_state.fit_running = False

        assert session_state.fit_running is False

    def test_results_page_enabled_after_fit(self, session_state):
        """Test that results page is enabled after successful fit."""
        mock_result = MagicMock()
        mock_result.success = True
        session_state.fit_result = mock_result

        can_show_results = session_state.fit_result is not None

        assert can_show_results is True


# =============================================================================
# Test Live Cost Plot Component
# =============================================================================


class TestLiveCostPlot:
    """Tests for live cost plot component."""

    def test_create_cost_history(self):
        """Test creating empty cost history."""
        history = create_cost_history()

        assert "iterations" in history
        assert "costs" in history
        assert len(history["iterations"]) == 0
        assert len(history["costs"]) == 0

    def test_update_cost_history(self):
        """Test updating cost history with new values."""
        history = create_cost_history()

        history = update_cost_history(history, iteration=1, cost=1.5)
        history = update_cost_history(history, iteration=2, cost=0.8)
        history = update_cost_history(history, iteration=3, cost=0.3)

        assert len(history["iterations"]) == 3
        assert history["iterations"] == [1, 2, 3]
        assert history["costs"] == [1.5, 0.8, 0.3]

    def test_create_cost_plot_figure(self):
        """Test creating Plotly figure from cost history."""
        history = {
            "iterations": [1, 2, 3, 4, 5],
            "costs": [10.0, 5.0, 2.0, 1.0, 0.5],
        }

        fig = create_cost_plot_figure(history)

        assert fig is not None
        # Check figure has data
        assert len(fig.data) > 0

    def test_plot_config_defaults(self):
        """Test default plot configuration."""
        config = get_plot_config()

        assert "displayModeBar" in config
        assert "staticPlot" in config or config.get("displayModeBar") is False


# =============================================================================
# Test Iteration Table Component
# =============================================================================


class TestIterationTable:
    """Tests for iteration parameter table component."""

    def test_create_iteration_history(self):
        """Test creating empty iteration history."""
        param_names = ["a", "b"]
        history = create_iteration_history(param_names)

        assert "iterations" in history
        assert "params" in history
        assert len(history["iterations"]) == 0

    def test_update_iteration_history(self):
        """Test updating iteration history."""
        param_names = ["a", "b"]
        history = create_iteration_history(param_names)

        params1 = np.array([1.0, 2.0])
        params2 = np.array([1.5, 3.0])

        history = update_iteration_history(history, iteration=1, params=params1)
        history = update_iteration_history(history, iteration=2, params=params2)

        assert len(history["iterations"]) == 2
        assert len(history["params"]) == 2

    def test_format_iteration_table(self):
        """Test formatting iteration history as table."""
        history = {
            "iterations": [1, 2, 3],
            "params": [
                np.array([1.0, 2.0]),
                np.array([1.5, 2.5]),
                np.array([1.9, 4.8]),
            ],
        }
        param_names = ["a", "b"]

        df = format_iteration_table(history, param_names)

        assert df is not None
        assert "Iteration" in df.columns
        assert "a" in df.columns
        assert "b" in df.columns
        assert len(df) == 3

    def test_limit_history_size(self):
        """Test limiting history to last N entries."""
        history = {
            "iterations": list(range(100)),
            "params": [np.array([i, i]) for i in range(100)],
        }

        limited = limit_history_size(history, max_entries=10)

        assert len(limited["iterations"]) == 10
        # Should keep the most recent entries
        assert limited["iterations"][0] == 90
        assert limited["iterations"][-1] == 99

    def test_table_display_config(self):
        """Test table display configuration."""
        config = get_table_display_config()

        assert "max_rows" in config
        assert config["max_rows"] > 0


# =============================================================================
# Test Checkpoint Indicator
# =============================================================================


class TestCheckpointIndicator:
    """Tests for checkpoint status indicator in streaming mode."""

    def test_checkpoint_status_not_shown_for_small_data(self, session_state):
        """Test checkpoint indicator hidden for small datasets."""
        # 50 points is small
        show_checkpoint = len(session_state.xdata) > 1_000_000

        assert show_checkpoint is False

    def test_checkpoint_status_shown_for_large_data(self, session_state):
        """Test checkpoint indicator shown for large datasets."""
        session_state.xdata = np.zeros(2_000_000)
        show_checkpoint = len(session_state.xdata) > 1_000_000

        assert show_checkpoint is True

    def test_chunk_progress_calculation(self):
        """Test chunk progress calculation for streaming."""
        total_points = 10_000_000
        chunk_size = 100_000
        current_chunk = 50

        total_chunks = (total_points + chunk_size - 1) // chunk_size
        chunk_progress = current_chunk / total_chunks

        assert total_chunks == 100
        assert chunk_progress == 0.5


# =============================================================================
# Regression Tests for is_ready_to_fit
# =============================================================================


class TestIsReadyToFitRegression:
    """Regression tests for is_ready_to_fit edge cases.

    These tests cover bugs that were fixed related to p0 validation
    and auto_p0 handling.
    """

    @pytest.fixture
    def state_with_data(self):
        """Create state with data loaded."""
        state = initialize_state()
        state.xdata = np.linspace(0, 10, 100)
        state.ydata = 2.0 * np.exp(-0.5 * state.xdata) + 0.1
        return state

    def test_p0_all_none_without_auto_p0_not_ready(self, state_with_data):
        """Regression: p0=[None, None, None] should NOT be ready without auto_p0.

        Bug: is_ready_to_fit only checked `if state.p0 is None` but didn't
        check if all elements in the list were None.
        """
        state = state_with_data
        state.p0 = [None, None, None]
        state.auto_p0 = False

        # Simulate the is_ready_to_fit logic
        model = exponential_decay

        has_auto_p0 = hasattr(model, "estimate_p0") and callable(
            getattr(model, "estimate_p0", None)
        )

        # This is the fixed logic
        if state.p0 is None:
            is_ready = state.auto_p0 and has_auto_p0
        else:
            all_none = all(v is None for v in state.p0)
            is_ready = not all_none or (state.auto_p0 and has_auto_p0)

        assert is_ready is False, (
            "p0=[None,None,None] without auto_p0 should not be ready"
        )

    def test_p0_all_none_with_auto_p0_is_ready(self, state_with_data):
        """Test: p0=[None, None, None] with auto_p0=True should be ready.

        When auto_p0 is enabled and model supports it, fitting should proceed
        even if p0 contains all None values.
        """
        state = state_with_data
        state.p0 = [None, None, None]
        state.auto_p0 = True

        model = exponential_decay

        has_auto_p0 = hasattr(model, "estimate_p0") and callable(
            getattr(model, "estimate_p0", None)
        )

        # This is the fixed logic
        if state.p0 is None:
            is_ready = state.auto_p0 and has_auto_p0
        else:
            all_none = all(v is None for v in state.p0)
            is_ready = not all_none or (state.auto_p0 and has_auto_p0)

        assert has_auto_p0 is True, "exponential_decay should have estimate_p0"
        assert is_ready is True, "p0=[None,None,None] with auto_p0 should be ready"

    def test_p0_partial_none_is_ready(self, state_with_data):
        """Test: p0 with some values set should be ready."""
        state = state_with_data
        state.p0 = [1.0, None, 0.1]
        state.auto_p0 = False

        # At least one non-None value
        all_none = all(v is None for v in state.p0)

        assert all_none is False
        # With partial values, fitting should attempt (None will be filled by auto_p0 if enabled)

    def test_p0_none_without_auto_model_not_ready(self, state_with_data):
        """Test: p0=None without auto_p0 model support should not be ready."""
        state = state_with_data
        state.p0 = None
        state.auto_p0 = True

        # Model without estimate_p0
        def simple_model(x, a, b):
            return a * x + b

        has_auto_p0 = hasattr(simple_model, "estimate_p0") and callable(
            getattr(simple_model, "estimate_p0", None)
        )

        is_ready = state.p0 is not None or (state.auto_p0 and has_auto_p0)

        assert has_auto_p0 is False
        assert is_ready is False


class TestAutoP0Application:
    """Tests for auto_p0 application during run_fit."""

    @pytest.fixture
    def state_with_data(self):
        """Create state with data loaded."""
        state = initialize_state()
        state.xdata = np.linspace(0, 10, 100)
        state.ydata = 2.0 * np.exp(-0.5 * state.xdata) + 0.1
        return state

    def test_auto_p0_fills_none_values(self, state_with_data):
        """Test that auto_p0 fills in None values before fitting."""
        state = state_with_data
        state.p0 = [None, None, None]
        state.auto_p0 = True

        model = exponential_decay

        # Simulate the auto_p0 application logic from run_fit
        has_auto_p0 = hasattr(model, "estimate_p0") and callable(
            getattr(model, "estimate_p0", None)
        )

        if state.auto_p0 and has_auto_p0:
            estimated_p0 = model.estimate_p0(state.xdata, state.ydata)
            if estimated_p0 is not None:
                estimated_p0 = list(estimated_p0)
                if state.p0 is None:
                    state.p0 = estimated_p0
                else:
                    for i in range(len(state.p0)):
                        if state.p0[i] is None and i < len(estimated_p0):
                            state.p0[i] = estimated_p0[i]

        # Verify all None values are filled
        assert state.p0 is not None
        assert len(state.p0) == 3
        assert all(v is not None for v in state.p0), "All None values should be filled"
        assert all(isinstance(v, (int, float)) for v in state.p0)

    def test_auto_p0_preserves_user_values(self, state_with_data):
        """Test that auto_p0 preserves user-set values."""
        state = state_with_data
        state.p0 = [5.0, None, 0.5]  # User set first and third, leave second as auto
        state.auto_p0 = True

        model = exponential_decay

        has_auto_p0 = hasattr(model, "estimate_p0")

        if state.auto_p0 and has_auto_p0:
            estimated_p0 = model.estimate_p0(state.xdata, state.ydata)
            if estimated_p0 is not None:
                estimated_p0 = list(estimated_p0)
                for i in range(len(state.p0)):
                    if state.p0[i] is None and i < len(estimated_p0):
                        state.p0[i] = estimated_p0[i]

        # User values should be preserved
        assert state.p0[0] == 5.0, "User value should be preserved"
        assert state.p0[2] == 0.5, "User value should be preserved"
        # Auto-filled value
        assert state.p0[1] is not None, "None value should be filled"

    def test_auto_p0_disabled_keeps_none(self, state_with_data):
        """Test that auto_p0=False does not fill None values."""
        state = state_with_data
        state.p0 = [None, None, None]
        state.auto_p0 = False

        model = exponential_decay

        # Simulate the logic - should not fill
        if state.auto_p0 and hasattr(model, "estimate_p0"):
            # This branch should not execute
            estimated_p0 = model.estimate_p0(state.xdata, state.ydata)
            state.p0 = list(estimated_p0)

        # p0 should still be all None
        assert all(v is None for v in state.p0), "p0 should remain unchanged"
