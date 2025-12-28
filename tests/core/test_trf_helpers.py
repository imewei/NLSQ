"""Tests for TRF helper methods extracted during refactoring.

This module tests the individual helper methods that were extracted from
trf_no_bounds to reduce complexity from 31 to <15.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from nlsq.core.trf import TrustRegionReflective


class TestInitializeTRFState:
    """Test the _initialize_trf_state helper method."""

    def test_basic_initialization(self):
        """Test basic state initialization without loss function."""
        trf = TrustRegionReflective()

        # Create simple test data
        n_params = 3
        n_residuals = 10
        x0 = np.array([1.0, 2.0, 3.0])
        f = jnp.ones(n_residuals)
        J = jnp.ones((n_residuals, n_params))
        x_scale = np.ones(n_params)
        data_mask = jnp.ones(n_residuals)

        # Initialize state
        state = trf._initialize_trf_state(
            x0=x0,
            f=f,
            J=J,
            loss_function=None,
            x_scale=x_scale,
            f_scale=1.0,
            data_mask=data_mask,
        )

        # Verify state structure
        assert isinstance(state, dict)
        assert "x" in state
        assert "f" in state
        assert "J" in state
        assert "cost" in state
        assert "g" in state
        assert "scale" in state
        assert "scale_inv" in state
        assert "Delta" in state
        assert "nfev" in state
        assert "njev" in state
        assert "m" in state
        assert "n" in state

        # Verify dimensions
        assert state["m"] == n_residuals
        assert state["n"] == n_params
        assert state["nfev"] == 1
        assert state["njev"] == 1

        # Verify x is a copy
        assert state["x"] is not x0
        assert_array_almost_equal(state["x"], x0)

        # Verify Delta is positive
        assert state["Delta"] > 0

    def test_initialization_with_jac_scaling(self):
        """Test state initialization with Jacobian-based scaling."""
        trf = TrustRegionReflective()

        x0 = np.array([1.0, 2.0])
        f = jnp.array([1.0, 2.0, 3.0])
        J = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        data_mask = jnp.ones(3)

        state = trf._initialize_trf_state(
            x0=x0,
            f=f,
            J=J,
            loss_function=None,
            x_scale="jac",  # Use Jacobian scaling
            f_scale=1.0,
            data_mask=data_mask,
        )

        # Verify jac_scale flag is set
        assert state["jac_scale"] is True
        assert "scale" in state
        assert "scale_inv" in state

    def test_initialization_with_zero_initial_guess(self):
        """Test that Delta is set to 1.0 when initial guess is zero."""
        trf = TrustRegionReflective()

        x0 = np.zeros(2)  # Zero initial guess
        f = jnp.ones(5)
        J = jnp.ones((5, 2))
        x_scale = np.ones(2)
        data_mask = jnp.ones(5)

        state = trf._initialize_trf_state(
            x0=x0,
            f=f,
            J=J,
            loss_function=None,
            x_scale=x_scale,
            f_scale=1.0,
            data_mask=data_mask,
        )

        # When x0 is zero, Delta should default to 1.0
        assert state["Delta"] == 1.0

    def test_gradient_computation(self):
        """Test that gradient is computed correctly."""
        trf = TrustRegionReflective()

        x0 = np.array([1.0, 1.0])
        f = jnp.array([1.0, 2.0])
        J = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # Identity-like
        x_scale = np.ones(2)
        data_mask = jnp.ones(2)

        state = trf._initialize_trf_state(
            x0=x0,
            f=f,
            J=J,
            loss_function=None,
            x_scale=x_scale,
            f_scale=1.0,
            data_mask=data_mask,
        )

        # Gradient should be J^T @ f
        expected_g = J.T @ f
        assert_array_almost_equal(state["g"], expected_g)

    def test_cost_function_computation(self):
        """Test that cost function is computed correctly."""
        trf = TrustRegionReflective()

        x0 = np.array([1.0])
        f = jnp.array([2.0, 3.0])
        J = jnp.ones((2, 1))
        x_scale = np.ones(1)
        data_mask = jnp.ones(2)

        state = trf._initialize_trf_state(
            x0=x0,
            f=f,
            J=J,
            loss_function=None,
            x_scale=x_scale,
            f_scale=1.0,
            data_mask=data_mask,
        )

        # Cost should be 0.5 * ||f||^2 = 0.5 * (4 + 9) = 6.5
        expected_cost = 0.5 * (2.0**2 + 3.0**2)
        assert abs(float(state["cost"]) - expected_cost) < 1e-10


class TestCheckConvergenceCriteria:
    """Test the _check_convergence_criteria helper method."""

    def test_convergence_satisfied(self):
        """Test convergence when gradient norm is below tolerance."""
        trf = TrustRegionReflective()

        # Small gradient (converged)
        g = jnp.array([1e-8, 1e-9])
        gtol = 1e-6

        status = trf._check_convergence_criteria(g, gtol)

        assert status == 1  # Convergence status

    def test_convergence_not_satisfied(self):
        """Test no convergence when gradient norm exceeds tolerance."""
        trf = TrustRegionReflective()

        # Large gradient (not converged)
        g = jnp.array([0.1, 0.2])
        gtol = 1e-6

        status = trf._check_convergence_criteria(g, gtol)

        assert status is None  # No convergence

    def test_convergence_boundary_case(self):
        """Test convergence at exact boundary."""
        trf = TrustRegionReflective()

        gtol = 1e-6
        # Gradient norm exactly at tolerance (should NOT converge, needs to be <)
        g = jnp.array([gtol, 0.0])

        status = trf._check_convergence_criteria(g, gtol)

        assert status is None  # Boundary case: need g_norm < gtol (strict)

    def test_convergence_just_below_tolerance(self):
        """Test convergence just below tolerance."""
        trf = TrustRegionReflective()

        gtol = 1e-6
        # Gradient norm just below tolerance
        g = jnp.array([gtol * 0.99, 0.0])

        status = trf._check_convergence_criteria(g, gtol)

        assert status == 1  # Should converge


class TestSolveTrustRegionSubproblem:
    """Test the _solve_trust_region_subproblem helper method."""

    # Note: CG solver test skipped due to JAX tracing complexity
    # The CG path is tested via integration tests in test_trf_simple.py

    def test_subproblem_exact_solver(self):
        """Test subproblem solving with exact (SVD) solver."""
        trf = TrustRegionReflective()

        # Simple problem
        J = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        f = jnp.array([1.0, 2.0])
        g = J.T @ f
        scale = np.ones(2)
        Delta = 1.0
        alpha = 0.01

        result = trf._solve_trust_region_subproblem(
            J, f, g, scale, Delta, alpha, solver="exact"
        )

        # Verify result structure
        assert "J_h" in result
        assert "g_h" in result
        assert "d" in result
        assert "d_jnp" in result
        assert result["step_h"] is None  # Exact solver computes step later
        assert result["s"] is not None  # Should have SVD components
        assert result["V"] is not None
        assert result["uf"] is not None

    def test_subproblem_scaling(self):
        """Test that scaling is applied correctly."""
        trf = TrustRegionReflective()

        J = jnp.ones((3, 2))
        f = jnp.ones(3)
        g = J.T @ f
        scale = np.array([2.0, 3.0])  # Non-uniform scaling
        Delta = 1.0
        alpha = 0.01

        result = trf._solve_trust_region_subproblem(
            J,
            f,
            g,
            scale,
            Delta,
            alpha,
            solver="exact",  # Use exact solver
        )

        # Verify scaling factors are stored
        assert_array_almost_equal(result["d"], scale)
        assert_array_almost_equal(result["d_jnp"], scale)


class TestEvaluateStepAcceptance:
    """Test the _evaluate_step_acceptance helper method."""

    def test_successful_step_acceptance(self):
        """Test that a good step is accepted."""
        trf = TrustRegionReflective()

        # Simple linear problem: y = a*x + b
        def model(x, xdata, ydata, data_mask, transform):
            a, b = x
            return ydata - (a * xdata + b)

        def jac_func(x, xdata, ydata, data_mask, transform):
            # Jacobian: df/da = -xdata, df/db = -1
            return jnp.stack([-xdata, -jnp.ones_like(xdata)], axis=1)

        # Data
        xdata = np.linspace(0, 5, 10)
        ydata = 2.0 * xdata + 1.0  # True: a=2, b=1
        data_mask = jnp.ones(10)

        # Current state (far from optimum)
        x = np.array([0.5, 0.5])
        f = model(x, xdata, ydata, data_mask, None)
        J = jac_func(x, xdata, ydata, data_mask, None)
        cost = 0.5 * jnp.sum(f**2)

        # Subproblem solution (mock - step toward optimum)
        d = np.ones(2)
        d_jnp = jnp.ones(2)
        J_h = J * d_jnp
        g_h = J.T @ f
        step_h = jnp.array([0.5, 0.2])  # Small step toward solution

        result = trf._evaluate_step_acceptance(
            fun=model,
            jac=jac_func,
            x=x,
            f=f,
            J=J,
            J_h=J_h,
            g_h_jnp=g_h,
            cost=float(cost),
            d=d,
            d_jnp=d_jnp,
            Delta=1.0,
            alpha=0.01,
            step_h=step_h,
            s=None,
            V=None,
            uf=None,
            xdata=xdata,
            ydata=ydata,
            data_mask=data_mask,
            transform=None,
            loss_function=None,
            f_scale=1.0,
            scale_inv=np.ones(2),
            jac_scale=False,
            solver="cg",
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=100,
            nfev=1,
        )

        # Step should be accepted (moves toward optimum)
        assert result["accepted"] == True  # noqa: E712
        assert result["actual_reduction"] > 0
        assert result["nfev"] == 2  # One function evaluation
        assert "x_new" in result
        assert "f_new" in result
        assert "J_new" in result
        assert "g_new" in result

    def test_step_acceptance_result_structure(self):
        """Test that result dictionary contains expected keys."""
        trf = TrustRegionReflective()

        # Simple linear problem
        def model(x, xdata, ydata, data_mask, transform):
            a, b = x
            return ydata - (a * xdata + b)

        def jac_func(x, xdata, ydata, data_mask, transform):
            return jnp.stack([-xdata, -jnp.ones_like(xdata)], axis=1)

        xdata = np.linspace(0, 5, 10)
        ydata = 2.0 * xdata + 1.0
        data_mask = jnp.ones(10)

        x = np.array([0.5, 0.5])
        f = model(x, xdata, ydata, data_mask, None)
        J = jac_func(x, xdata, ydata, data_mask, None)
        cost = 0.5 * jnp.sum(f**2)

        d = np.ones(2)
        d_jnp = jnp.ones(2)
        J_h = J * d_jnp
        g_h = J.T @ f
        step_h = jnp.array([0.1, 0.1])

        result = trf._evaluate_step_acceptance(
            fun=model,
            jac=jac_func,
            x=x,
            f=f,
            J=J,
            J_h=J_h,
            g_h_jnp=g_h,
            cost=float(cost),
            d=d,
            d_jnp=d_jnp,
            Delta=1.0,
            alpha=0.01,
            step_h=step_h,
            s=None,
            V=None,
            uf=None,
            xdata=xdata,
            ydata=ydata,
            data_mask=data_mask,
            transform=None,
            loss_function=None,
            f_scale=1.0,
            scale_inv=np.ones(2),
            jac_scale=False,
            solver="cg",
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=100,
            nfev=1,
        )

        # Verify result structure
        assert "accepted" in result
        assert "actual_reduction" in result
        assert "step_norm" in result
        assert "Delta" in result
        assert "alpha" in result
        assert "termination_status" in result
        assert "nfev" in result
        assert "njev" in result

    def test_max_nfev_termination(self):
        """Test that evaluation stops when max_nfev is reached."""
        trf = TrustRegionReflective()

        # Function that always returns same residual
        def model(x, xdata, ydata, data_mask, transform):
            return jnp.ones(5)

        def jac_func(x, xdata, ydata, data_mask, transform):
            return jnp.ones((5, 2))

        xdata = np.linspace(0, 1, 5)
        ydata = np.ones(5)
        data_mask = jnp.ones(5)
        x = np.array([1.0, 1.0])
        f = model(x, xdata, ydata, data_mask, None)
        J = jac_func(x, xdata, ydata, data_mask, None)
        cost = 0.5 * jnp.sum(f**2)

        d = np.ones(2)
        d_jnp = jnp.ones(2)
        J_h = J * d_jnp
        g_h = J.T @ f
        step_h = jnp.array([0.1, 0.1])

        # Set max_nfev to current nfev (no more evaluations allowed)
        result = trf._evaluate_step_acceptance(
            fun=model,
            jac=jac_func,
            x=x,
            f=f,
            J=J,
            J_h=J_h,
            g_h_jnp=g_h,
            cost=float(cost),
            d=d,
            d_jnp=d_jnp,
            Delta=1.0,
            alpha=0.01,
            step_h=step_h,
            s=None,
            V=None,
            uf=None,
            xdata=xdata,
            ydata=ydata,
            data_mask=data_mask,
            transform=None,
            loss_function=None,
            f_scale=1.0,
            scale_inv=np.ones(2),
            jac_scale=False,
            solver="cg",
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=5,  # Already at limit
            nfev=5,
        )

        # Should exit immediately
        assert result["accepted"] == False  # noqa: E712
        assert result["nfev"] == 5  # No additional evaluations


class TestHelperMethodsIntegration:
    """Integration tests to ensure helpers work together."""

    def test_initialize_produces_valid_state(self):
        """Test that initialized state can be used by other methods."""
        trf = TrustRegionReflective()

        # Simple exponential decay problem
        x0 = np.array([1.0, 0.1])
        t = np.linspace(0, 10, 20)
        y_true = 2.5 * np.exp(-0.5 * t)
        y_data = y_true + 0.1 * np.random.RandomState(42).randn(len(t))

        # Compute initial residuals and Jacobian (mock)
        f = jnp.array(y_data - x0[0] * np.exp(-x0[1] * t))
        J = jnp.ones((len(t), 2))  # Simplified Jacobian
        x_scale = np.ones(2)
        data_mask = jnp.ones(len(t))

        state = trf._initialize_trf_state(
            x0=x0,
            f=f,
            J=J,
            loss_function=None,
            x_scale=x_scale,
            f_scale=1.0,
            data_mask=data_mask,
        )

        # Verify state is ready for optimization
        assert state["m"] == len(t)
        assert state["n"] == 2
        assert state["Delta"] > 0
        assert len(state["g"]) == 2
        assert state["cost"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
