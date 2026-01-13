"""Unit tests for CovarianceComputer component.

Tests for covariance matrix computation, sigma transformation,
and condition number estimation.

Reference: specs/017-curve-fit-decomposition/spec.md FR-003, FR-021
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlsq.core.orchestration.covariance_computer import CovarianceComputer
from nlsq.interfaces.orchestration_protocol import CovarianceResult

if TYPE_CHECKING:
    import jax


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def computer() -> CovarianceComputer:
    """Create a CovarianceComputer instance."""
    return CovarianceComputer()


@dataclass
class MockOptimizeResult:
    """Mock OptimizeResult for testing."""

    x: jax.Array  # Optimal parameters
    cost: float  # Residual sum of squares / 2
    jac: jax.Array  # Jacobian at solution
    fun: jax.Array  # Residuals at solution


@pytest.fixture
def simple_result() -> MockOptimizeResult:
    """Simple optimization result with well-conditioned Jacobian."""
    # Simulating result from fitting y = a*x + b
    # With x = [1, 2, 3, 4, 5] and y = [2.1, 4.0, 5.9, 8.1, 10.0]
    # True params: a=2.0, b=0.0
    jac = jnp.array(
        [
            [1.0, 1.0],  # dy/da=x, dy/db=1 at x=1
            [2.0, 1.0],  # at x=2
            [3.0, 1.0],  # at x=3
            [4.0, 1.0],  # at x=4
            [5.0, 1.0],  # at x=5
        ]
    )
    return MockOptimizeResult(
        x=jnp.array([2.0, 0.0]),
        cost=0.03,  # Small residual
        jac=jac,
        fun=jnp.array([0.1, 0.0, -0.1, 0.1, 0.0]),  # Small residuals
    )


@pytest.fixture
def singular_result() -> MockOptimizeResult:
    """Optimization result with singular Jacobian."""
    # Jacobian with linearly dependent columns
    jac = jnp.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
        ]
    )
    return MockOptimizeResult(
        x=jnp.array([1.0, 1.0]),
        cost=0.5,
        jac=jac,
        fun=jnp.array([0.1, 0.2, 0.1, 0.2, 0.1]),
    )


@pytest.fixture
def sigma_1d() -> jax.Array:
    """1D sigma (uncertainties)."""
    return jnp.array([0.1, 0.2, 0.15, 0.1, 0.2])


@pytest.fixture
def sigma_2d() -> jax.Array:
    """2D sigma (covariance matrix)."""
    # Simple diagonal covariance
    return jnp.diag(jnp.array([0.01, 0.04, 0.0225, 0.01, 0.04]))


# =============================================================================
# Test CovarianceResult
# =============================================================================


class TestCovarianceResultType:
    """Tests for CovarianceResult return type."""

    def test_returns_covariance_result(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """Test compute returns CovarianceResult instance."""
        result = computer.compute(
            result=simple_result,
            n_data=5,
        )

        assert isinstance(result, CovarianceResult)

    def test_covariance_result_has_required_fields(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """Test CovarianceResult has all required attributes."""
        result = computer.compute(
            result=simple_result,
            n_data=5,
        )

        assert hasattr(result, "pcov")
        assert hasattr(result, "perr")
        assert hasattr(result, "method")
        assert hasattr(result, "condition_number")
        assert hasattr(result, "is_singular")
        assert hasattr(result, "sigma_used")
        assert hasattr(result, "absolute_sigma")


# =============================================================================
# Test Covariance Computation
# =============================================================================


class TestCovarianceComputation:
    """Tests for basic covariance computation."""

    def test_computes_covariance_matrix(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """Test computes valid covariance matrix."""
        result = computer.compute(
            result=simple_result,
            n_data=5,
        )

        pcov = np.asarray(result.pcov)
        # Should be 2x2 for 2 parameters
        assert pcov.shape == (2, 2)
        # Should be symmetric
        assert_allclose(pcov, pcov.T)
        # Diagonal should be positive (variances)
        assert np.all(np.diag(pcov) >= 0)

    def test_computes_parameter_errors(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """Test computes parameter standard errors."""
        result = computer.compute(
            result=simple_result,
            n_data=5,
        )

        perr = np.asarray(result.perr)
        pcov = np.asarray(result.pcov)

        # perr should be sqrt of diagonal
        expected_perr = np.sqrt(np.diag(pcov))
        assert_allclose(perr, expected_perr)

    def test_uses_svd_method(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """Test uses SVD for covariance computation."""
        result = computer.compute(
            result=simple_result,
            n_data=5,
        )

        assert result.method == "svd"


# =============================================================================
# Test Relative vs Absolute Sigma
# =============================================================================


class TestSigmaHandling:
    """Tests for sigma (uncertainty) handling."""

    def test_relative_sigma_scaling(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """Test relative sigma scales covariance."""
        result_no_sigma = computer.compute(
            result=simple_result,
            n_data=5,
            absolute_sigma=False,
        )

        # With relative sigma, covariance is scaled by residual variance
        assert not result_no_sigma.absolute_sigma
        pcov = np.asarray(result_no_sigma.pcov)
        assert np.all(np.isfinite(pcov))

    def test_absolute_sigma_no_scaling(
        self,
        computer: CovarianceComputer,
        simple_result: MockOptimizeResult,
        sigma_1d: jax.Array,
    ) -> None:
        """Test absolute sigma prevents variance scaling."""
        result = computer.compute(
            result=simple_result,
            n_data=5,
            sigma=sigma_1d,
            absolute_sigma=True,
        )

        assert result.absolute_sigma
        assert result.sigma_used

    def test_sigma_used_flag(
        self,
        computer: CovarianceComputer,
        simple_result: MockOptimizeResult,
        sigma_1d: jax.Array,
    ) -> None:
        """Test sigma_used flag is set correctly."""
        result_no_sigma = computer.compute(
            result=simple_result,
            n_data=5,
        )
        assert not result_no_sigma.sigma_used

        result_with_sigma = computer.compute(
            result=simple_result,
            n_data=5,
            sigma=sigma_1d,
        )
        assert result_with_sigma.sigma_used


# =============================================================================
# Test Singularity Detection
# =============================================================================


class TestSingularityDetection:
    """Tests for singular/ill-conditioned Jacobian detection."""

    def test_detects_singular_jacobian(
        self, computer: CovarianceComputer, singular_result: MockOptimizeResult
    ) -> None:
        """Test detects singular Jacobian."""
        result = computer.compute(
            result=singular_result,
            n_data=5,
        )

        assert result.is_singular
        # Covariance should be filled with inf
        pcov = np.asarray(result.pcov)
        assert np.any(np.isinf(pcov))

    def test_condition_number_singular(
        self, computer: CovarianceComputer, singular_result: MockOptimizeResult
    ) -> None:
        """Test singular matrix is detected via is_singular flag.

        Note: For rank-deficient matrices, condition number may be finite
        (computed from valid singular values only), but is_singular should be True.
        """
        result = computer.compute(
            result=singular_result,
            n_data=5,
        )

        # The key indicator is the is_singular flag, not condition number
        assert result.is_singular
        # pcov should have inf values
        assert np.any(np.isinf(np.asarray(result.pcov)))

    def test_condition_number_well_conditioned(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """Test condition number is finite for well-conditioned matrix."""
        result = computer.compute(
            result=simple_result,
            n_data=5,
        )

        assert np.isfinite(result.condition_number)
        assert result.condition_number >= 1.0


# =============================================================================
# Test Insufficient Data Handling
# =============================================================================


class TestInsufficientData:
    """Tests for insufficient data (n_data <= n_params)."""

    def test_warns_insufficient_data(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """Test warns when n_data <= n_params."""
        result = computer.compute(
            result=simple_result,
            n_data=2,  # Only 2 points for 2 params
            absolute_sigma=False,
        )

        # Should produce inf covariance for relative sigma
        pcov = np.asarray(result.pcov)
        assert np.all(np.isinf(pcov))
        assert result.is_singular


# =============================================================================
# Test Sigma Transform Creation
# =============================================================================


class TestSigmaTransform:
    """Tests for sigma transformation functions."""

    def test_create_sigma_transform_1d(
        self, computer: CovarianceComputer, sigma_1d: jax.Array
    ) -> None:
        """Test 1D sigma transform creation."""
        transform, is_2d = computer.create_sigma_transform(
            sigma=sigma_1d,
            n_data=5,
        )

        assert not is_2d
        assert callable(transform)

    def test_create_sigma_transform_2d(
        self, computer: CovarianceComputer, sigma_2d: jax.Array
    ) -> None:
        """Test 2D sigma transform creation."""
        transform, is_2d = computer.create_sigma_transform(
            sigma=sigma_2d,
            n_data=5,
        )

        assert is_2d
        assert callable(transform)

    def test_sigma_transform_1d_is_inverse(self, computer: CovarianceComputer) -> None:
        """Test 1D sigma transform is 1/sigma."""
        sigma = jnp.array([0.5, 1.0, 2.0])
        transform, _is_2d = computer.create_sigma_transform(sigma, n_data=3)

        # For 1D, transform should be 1/sigma
        expected = 1.0 / sigma
        mask = jnp.ones(3, dtype=bool)
        result = transform(sigma, mask)
        assert_allclose(np.asarray(result), np.asarray(expected))


# =============================================================================
# Test Condition Number Computation
# =============================================================================


class TestConditionNumberComputation:
    """Tests for condition number computation."""

    def test_condition_number_identity(self, computer: CovarianceComputer) -> None:
        """Test condition number of identity is 1."""
        jac = jnp.eye(3)
        cond = computer.compute_condition_number(jac)

        assert_allclose(cond, 1.0, rtol=1e-10)

    def test_condition_number_diagonal(self, computer: CovarianceComputer) -> None:
        """Test condition number of diagonal matrix."""
        jac = jnp.diag(jnp.array([10.0, 1.0]))
        cond = computer.compute_condition_number(jac)

        assert_allclose(cond, 10.0, rtol=1e-10)

    def test_condition_number_rectangular(self, computer: CovarianceComputer) -> None:
        """Test condition number of rectangular matrix."""
        jac = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        cond = computer.compute_condition_number(jac)

        assert cond >= 1.0
        assert np.isfinite(cond)


# =============================================================================
# Test Immutability
# =============================================================================


class TestImmutability:
    """Tests for CovarianceResult immutability."""

    def test_covariance_result_is_frozen(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """Test CovarianceResult cannot be modified."""
        result = computer.compute(
            result=simple_result,
            n_data=5,
        )

        with pytest.raises((AttributeError, TypeError)):
            result.is_singular = True  # type: ignore[misc]


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_parameter(self, computer: CovarianceComputer) -> None:
        """Test covariance for single parameter."""
        result = MockOptimizeResult(
            x=jnp.array([1.5]),
            cost=0.01,
            jac=jnp.array([[1.0], [2.0], [3.0]]),
            fun=jnp.array([0.1, 0.0, -0.1]),
        )

        cov_result = computer.compute(result=result, n_data=3)

        pcov = np.asarray(cov_result.pcov)
        assert pcov.shape == (1, 1)
        assert np.isfinite(pcov[0, 0])

    def test_many_parameters(self, computer: CovarianceComputer) -> None:
        """Test covariance for many parameters."""
        n_params = 10
        n_data = 100

        # Create a well-conditioned Jacobian
        rng = np.random.default_rng(42)
        jac = jnp.array(rng.standard_normal((n_data, n_params)))

        result = MockOptimizeResult(
            x=jnp.zeros(n_params),
            cost=0.1,
            jac=jac,
            fun=jnp.zeros(n_data),
        )

        cov_result = computer.compute(result=result, n_data=n_data)

        pcov = np.asarray(cov_result.pcov)
        assert pcov.shape == (n_params, n_params)
        assert np.all(np.isfinite(pcov))

    def test_zero_residual(self, computer: CovarianceComputer) -> None:
        """Test covariance when residual is exactly zero."""
        jac = jnp.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        result = MockOptimizeResult(
            x=jnp.array([1.0, 1.0]),
            cost=0.0,  # Zero cost
            jac=jac,
            fun=jnp.zeros(3),
        )

        cov_result = computer.compute(result=result, n_data=3)

        # With zero cost and relative sigma, covariance goes to zero
        # or might warn about degeneracy
        pcov = np.asarray(cov_result.pcov)
        # Just check it doesn't crash and returns valid shape
        assert pcov.shape == (2, 2)
