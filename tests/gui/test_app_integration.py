"""Tests for NLSQ GUI main application integration.

This module tests the main app entry point, including session state initialization,
sidebar status panel, navigation guards, page navigation, and workflow end-to-end.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class MockSessionState(dict):
    """Mock class that allows both dict-like and attribute access."""

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class TestAppLaunches:
    """Tests for app launch behavior."""

    def test_configure_page_runs_without_error(self) -> None:
        """Test that configure_page can be called without error."""
        from nlsq.gui.app import configure_page

        # Mock st.set_page_config since it can only be called once
        with patch("nlsq.gui.app.st") as mock_st:
            configure_page()
            mock_st.set_page_config.assert_called_once()

    def test_initialize_session_state_runs_without_error(self) -> None:
        """Test that initialize_session_state properly initializes state."""
        from nlsq.gui.app import initialize_session_state

        with patch("nlsq.gui.app.st") as mock_st:
            # Mock session_state as a MockSessionState that supports attribute access
            mock_st.session_state = MockSessionState()
            initialize_session_state()
            # State should now have nlsq_state key
            assert "nlsq_state" in mock_st.session_state

    def test_initialize_session_state_preserves_existing_state(self) -> None:
        """Test that initialize_session_state does not overwrite existing state."""
        from nlsq.gui.app import initialize_session_state
        from nlsq.gui.state import SessionState

        with patch("nlsq.gui.app.st") as mock_st:
            # Pre-populate session state with existing nlsq_state
            existing_state = SessionState(model_name="gaussian")
            mock_st.session_state = MockSessionState({"nlsq_state": existing_state})
            initialize_session_state()
            # Should preserve existing state
            assert mock_st.session_state["nlsq_state"].model_name == "gaussian"

    def test_main_function_exists_and_callable(self) -> None:
        """Test that main entry point function exists."""
        from nlsq.gui.app import main

        assert callable(main)


class TestPageNavigation:
    """Tests for page navigation and multi-page structure."""

    def test_get_page_status_data_not_loaded(self) -> None:
        """Test page status when data is not loaded."""
        from nlsq.gui.app import get_page_status
        from nlsq.gui.state import SessionState

        # Create a fresh state without data
        state = SessionState()
        # Clear model_type to simulate fresh app start
        state.model_type = ""
        status = get_page_status(state)

        assert status["data"]["loaded"] is False
        assert status["model"]["selected"] is False
        assert status["fit"]["complete"] is False

    def test_get_page_status_data_loaded(self) -> None:
        """Test page status when data is loaded."""
        from nlsq.gui.app import get_page_status
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])

        status = get_page_status(state)

        assert status["data"]["loaded"] is True
        assert status["data"]["point_count"] == 3

    def test_get_page_status_model_selected(self) -> None:
        """Test page status when model is selected."""
        from nlsq.gui.app import get_page_status
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        state.model_name = "gaussian"
        state.model_type = "builtin"

        status = get_page_status(state)

        assert status["model"]["selected"] is True
        assert status["model"]["name"] == "gaussian"

    def test_get_page_status_fit_complete(self) -> None:
        """Test page status when fit is complete."""
        from nlsq.gui.app import get_page_status
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        state.model_name = "gaussian"
        state.fit_result = {"popt": [1.0, 2.0], "success": True}

        status = get_page_status(state)

        assert status["fit"]["complete"] is True


class TestNavigationGuards:
    """Tests for navigation guards that disable pages based on workflow state."""

    def test_can_access_data_page_always(self) -> None:
        """Test that Data Loading page is always accessible."""
        from nlsq.gui.app import can_access_page
        from nlsq.gui.state import SessionState

        state = SessionState()
        assert can_access_page("data", state) is True

    def test_cannot_access_model_page_without_data(self) -> None:
        """Test that Model page is disabled without data."""
        from nlsq.gui.app import can_access_page
        from nlsq.gui.state import SessionState

        state = SessionState()
        assert can_access_page("model", state) is False

    def test_can_access_model_page_with_data(self) -> None:
        """Test that Model page is enabled with data."""
        from nlsq.gui.app import can_access_page
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        assert can_access_page("model", state) is True

    def test_cannot_access_fitting_page_without_model(self) -> None:
        """Test that Fitting page is disabled without model selection."""
        from nlsq.gui.app import can_access_page
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        # Simulate no model selected by clearing model_type
        state.model_type = ""
        assert can_access_page("fitting", state) is False

    def test_can_access_fitting_page_with_model(self) -> None:
        """Test that Fitting page is enabled with model selection."""
        from nlsq.gui.app import can_access_page
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        state.model_type = "builtin"
        state.model_name = "gaussian"
        assert can_access_page("fitting", state) is True

    def test_cannot_access_results_page_without_fit(self) -> None:
        """Test that Results page is disabled without fit results."""
        from nlsq.gui.app import can_access_page
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        state.model_type = "builtin"
        state.model_name = "gaussian"
        assert can_access_page("results", state) is False

    def test_can_access_results_page_with_fit(self) -> None:
        """Test that Results page is enabled with fit results."""
        from nlsq.gui.app import can_access_page
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        state.model_type = "builtin"
        state.model_name = "gaussian"
        state.fit_result = {"popt": [1.0, 2.0], "success": True}
        assert can_access_page("results", state) is True

    def test_cannot_access_export_page_without_fit(self) -> None:
        """Test that Export page is disabled without fit results."""
        from nlsq.gui.app import can_access_page
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        state.model_type = "builtin"
        state.model_name = "gaussian"
        assert can_access_page("export", state) is False


class TestStatePersistence:
    """Tests for state persistence across pages."""

    def test_state_persists_data_fields(self) -> None:
        """Test that data fields persist in session state."""
        from nlsq.gui.state import initialize_state

        state = initialize_state()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        state.data_file_name = "test.csv"

        # Verify persistence
        assert state.xdata is not None
        assert len(state.xdata) == 3
        assert state.data_file_name == "test.csv"

    def test_state_persists_model_fields(self) -> None:
        """Test that model fields persist in session state."""
        from nlsq.gui.state import initialize_state

        state = initialize_state()
        state.model_type = "builtin"
        state.model_name = "lorentzian"
        state.auto_p0 = False

        # Verify persistence
        assert state.model_type == "builtin"
        assert state.model_name == "lorentzian"
        assert state.auto_p0 is False

    def test_state_persists_fit_result(self) -> None:
        """Test that fit result persists in session state."""
        from nlsq.gui.state import initialize_state

        state = initialize_state()
        mock_result = {
            "popt": np.array([1.0, 2.0, 3.0]),
            "pcov": np.eye(3),
            "success": True,
        }
        state.fit_result = mock_result

        # Verify persistence
        assert state.fit_result is not None
        assert state.fit_result["success"] is True
        assert len(state.fit_result["popt"]) == 3


class TestWorkflowEndToEnd:
    """Tests for complete workflow from data to export."""

    def test_workflow_progression(self) -> None:
        """Test that workflow progresses through all stages correctly."""
        from nlsq.gui.app import can_access_page, get_page_status
        from nlsq.gui.state import SessionState

        # Create fresh state with cleared model to simulate fresh app start
        state = SessionState()
        state.model_type = ""

        # Stage 1: Initial state - only data page accessible
        assert can_access_page("data", state) is True
        assert can_access_page("model", state) is False
        status = get_page_status(state)
        assert status["data"]["loaded"] is False

        # Stage 2: Load data - model page becomes accessible
        state.xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state.ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert can_access_page("model", state) is True
        assert can_access_page("fitting", state) is False
        status = get_page_status(state)
        assert status["data"]["loaded"] is True
        assert status["data"]["point_count"] == 5

        # Stage 3: Select model - fitting page becomes accessible
        state.model_type = "builtin"
        state.model_name = "linear"
        assert can_access_page("fitting", state) is True
        assert can_access_page("results", state) is False
        status = get_page_status(state)
        assert status["model"]["selected"] is True
        assert status["model"]["name"] == "linear"

        # Stage 4: Complete fit - results and export pages become accessible
        state.fit_result = {
            "popt": np.array([2.0, 0.0]),
            "pcov": np.eye(2),
            "success": True,
        }
        assert can_access_page("results", state) is True
        assert can_access_page("export", state) is True
        status = get_page_status(state)
        assert status["fit"]["complete"] is True

    def test_workflow_step_messages(self) -> None:
        """Test that appropriate messages are returned for each workflow step."""
        from nlsq.gui.app import get_workflow_step_message
        from nlsq.gui.state import SessionState

        # Create fresh state with cleared model to simulate fresh app start
        state = SessionState()
        state.model_type = ""

        # No data loaded
        msg = get_workflow_step_message(state)
        assert "Load data" in msg or "data" in msg.lower()

        # Data loaded, no model
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        msg = get_workflow_step_message(state)
        assert "model" in msg.lower() or "Select" in msg

        # Model selected, no fit
        state.model_type = "builtin"
        state.model_name = "gaussian"
        msg = get_workflow_step_message(state)
        assert "fit" in msg.lower() or "Run" in msg

        # Fit complete
        state.fit_result = {"popt": [1.0], "success": True}
        msg = get_workflow_step_message(state)
        assert "complete" in msg.lower() or "Results" in msg or "Export" in msg


class TestSidebarStatusPanel:
    """Tests for sidebar status panel display."""

    def test_render_sidebar_status_with_no_data(self) -> None:
        """Test sidebar status when no data is loaded."""
        from nlsq.gui.app import get_page_status
        from nlsq.gui.state import SessionState

        state = SessionState()
        status = get_page_status(state)

        assert status["data"]["loaded"] is False
        assert "point_count" not in status["data"] or status["data"]["point_count"] == 0

    def test_render_sidebar_status_with_data(self) -> None:
        """Test sidebar status when data is loaded."""
        from nlsq.gui.app import get_page_status
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state.ydata = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        state.data_file_name = "test_data.csv"

        status = get_page_status(state)

        assert status["data"]["loaded"] is True
        assert status["data"]["point_count"] == 5
        assert status["data"]["file_name"] == "test_data.csv"

    def test_render_sidebar_status_fit_running(self) -> None:
        """Test sidebar status when fit is running."""
        from nlsq.gui.app import get_page_status
        from nlsq.gui.state import SessionState

        state = SessionState()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        state.model_type = "builtin"
        state.model_name = "gaussian"
        state.fit_running = True

        status = get_page_status(state)

        assert status["fit"]["running"] is True
        assert status["fit"]["complete"] is False


class TestHelpAndAbout:
    """Tests for help and about section."""

    def test_get_version_info(self) -> None:
        """Test that version info can be retrieved."""
        from nlsq.gui.app import get_version_info

        version_info = get_version_info()

        assert "version" in version_info
        assert "github_url" in version_info
        assert isinstance(version_info["version"], str)

    def test_get_help_tips(self) -> None:
        """Test that help tips are available."""
        from nlsq.gui.app import get_help_tips

        tips = get_help_tips()

        assert isinstance(tips, list)
        assert len(tips) > 0
        # Should have at least workflow tips
        assert any("data" in tip.lower() or "load" in tip.lower() for tip in tips)
