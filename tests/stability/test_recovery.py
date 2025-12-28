"""
Comprehensive tests for OptimizationRecovery error paths.

Target: Cover recovery mechanisms and error handling for Sprint 1 safety net.
Goal: 10-12 tests covering failure types, recovery strategies, and error paths.
"""

import numpy as np
import pytest

from nlsq.stability.recovery import OptimizationRecovery


class TestRecoveryBasics:
    """Test basic recovery functionality."""

    def setup_method(self):
        """Setup recovery instance."""
        self.recovery = OptimizationRecovery(max_retries=3)

    def test_successful_recovery_convergence_failure(self):
        """Test successful recovery from convergence failure."""
        state = {"x": np.array([1.0, 2.0]), "iteration": 10, "cost": 1.5}

        def mock_optimization(**kwargs):
            # Mock successful optimization after recovery
            return {"success": True, "x": kwargs.get("x", state["x"]), "cost": 0.1}

        success, result = self.recovery.recover_from_failure(
            "convergence", state, mock_optimization
        )

        assert success
        assert result["success"]
        assert len(self.recovery.recovery_history) == 1

    def test_recovery_failure_all_retries_exhausted(self):
        """Test recovery failure when all retries exhausted."""
        state = {"x": np.array([1.0, 2.0]), "iteration": 10, "cost": 1.5}

        def mock_optimization(**kwargs):
            # Mock always failing optimization
            return {"success": False, "x": kwargs.get("x", state["x"]), "cost": np.inf}

        success, result = self.recovery.recover_from_failure(
            "numerical", state, mock_optimization
        )

        assert not success
        assert "error" in result
        assert "Recovery failed" in result["error"]

    def test_recovery_history_tracking(self):
        """Test recovery history is properly tracked."""
        state = {"x": np.array([1.0, 2.0]), "iteration": 5, "cost": 2.0}

        def mock_optimization(**kwargs):
            return {"success": True, "x": kwargs.get("x", state["x"]), "cost": 0.5}

        # First recovery attempt
        self.recovery.recover_from_failure("convergence", state, mock_optimization)

        assert len(self.recovery.recovery_history) == 1
        assert self.recovery.recovery_history[0]["failure_type"] == "convergence"
        assert self.recovery.recovery_history[0]["iteration"] == 5

        # Second recovery attempt
        state2 = {"x": np.array([3.0, 4.0]), "iteration": 8, "cost": 1.0}
        self.recovery.recover_from_failure("numerical", state2, mock_optimization)

        assert len(self.recovery.recovery_history) == 2
        assert self.recovery.recovery_history[1]["failure_type"] == "numerical"


class TestRecoveryStrategies:
    """Test individual recovery strategies."""

    def setup_method(self):
        """Setup recovery instance."""
        self.recovery = OptimizationRecovery(max_retries=2)

    def test_perturb_parameters_strategy(self):
        """Test _perturb_parameters recovery strategy."""
        state = {"x": np.array([1.0, 2.0]), "iteration": 5, "cost": 1.0}

        modified_state = self.recovery._perturb_parameters(
            "convergence", state, retry=0
        )

        assert "x" in modified_state
        assert isinstance(modified_state, dict)
        # Strategy should return valid state (parameters may or may not be perturbed)

    def test_switch_algorithm_strategy(self):
        """Test _switch_algorithm recovery strategy."""
        state = {"method": "trf", "x": np.array([1.0, 2.0])}

        # This strategy might switch methods or adjust settings
        # Just verify it returns a valid state
        modified_state = self.recovery._switch_algorithm("numerical", state, retry=0)

        assert isinstance(modified_state, dict)
        assert "x" in modified_state

    def test_adjust_regularization_strategy(self):
        """Test _adjust_regularization recovery strategy."""
        state = {"x": np.array([1.0, 2.0]), "cost": 1.0}

        modified_state = self.recovery._adjust_regularization(
            "numerical", state, retry=0
        )

        assert isinstance(modified_state, dict)
        # Should have regularization-related modifications

    def test_reformulate_problem_strategy(self):
        """Test _reformulate_problem recovery strategy."""
        state = {"x": np.array([1.0, 2.0]), "bounds": (-np.inf, np.inf)}

        modified_state = self.recovery._reformulate_problem(
            "convergence", state, retry=0
        )

        assert isinstance(modified_state, dict)

    def test_multi_start_strategy(self):
        """Test _multi_start recovery strategy."""
        state = {"x": np.array([1.0, 2.0]), "iteration": 10}

        modified_state = self.recovery._multi_start("convergence", state, retry=1)

        assert isinstance(modified_state, dict)
        assert "x" in modified_state


class TestRecoveryErrorPaths:
    """Test error handling in recovery."""

    def setup_method(self):
        """Setup recovery instance."""
        self.recovery = OptimizationRecovery(max_retries=2)

    @pytest.mark.filterwarnings("ignore:Recovery strategy.*failed:UserWarning")
    def test_recovery_with_exception_in_strategy(self):
        """Test recovery handles exceptions in strategies gracefully."""
        state = {"x": np.array([1.0, 2.0]), "iteration": 5}

        def failing_optimization(**kwargs):
            # This should trigger exception handling in recovery
            raise ValueError("Mock optimization failure")

        # Should catch the exception and continue with next strategy
        success, result = self.recovery.recover_from_failure(
            "numerical", state, failing_optimization
        )

        # All strategies should fail due to exception
        assert not success
        assert "error" in result

    def test_check_recovery_success_with_valid_result(self):
        """Test _check_recovery_success with valid result."""
        result = {"success": True, "cost": 0.5, "x": np.array([1.0, 2.0])}

        assert self.recovery._check_recovery_success(result)

    def test_check_recovery_success_with_failed_result(self):
        """Test _check_recovery_success with failed result."""
        result = {"success": False, "cost": np.inf}

        assert not self.recovery._check_recovery_success(result)


# Total: 11 comprehensive tests covering recovery error paths
