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

    def test_none_jac_raises(self, computer: CovarianceComputer) -> None:
        """A missing Jacobian must raise, not crash deep inside JAX."""
        bad_result = MockOptimizeResult(
            x=jnp.array([1.0, 1.0]),
            cost=0.1,
            jac=None,
            fun=jnp.array([0.1]),
        )

        with pytest.raises(ValueError, match="2-D Jacobian"):
            computer.compute(result=bad_result, n_data=5)

    def test_1d_jac_raises(self, computer: CovarianceComputer) -> None:
        """A 1-D 'Jacobian' must raise, not silently misbehave."""
        bad_result = MockOptimizeResult(
            x=jnp.array([1.0]),
            cost=0.1,
            jac=jnp.array([1.0, 2.0, 3.0]),
            fun=jnp.array([0.1]),
        )

        with pytest.raises(ValueError, match="2-D Jacobian"):
            computer.compute(result=bad_result, n_data=3)


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

        # For 1D, transform should be 1/sigma (data_mask not needed — filtering
        # happens upstream in DataPreprocessor before sigma reaches this function)
        expected = 1.0 / sigma
        result = transform(sigma)
        assert_allclose(np.asarray(result), np.asarray(expected))

    def test_create_sigma_transform_rejects_length_mismatch(
        self, computer: CovarianceComputer, sigma_1d: jax.Array
    ) -> None:
        """sigma length must match n_data, not silently be accepted."""
        with pytest.raises(ValueError, match="must match n_data"):
            computer.create_sigma_transform(sigma=sigma_1d, n_data=999)

    def test_create_sigma_transform_rejects_non_positive_1d(
        self, computer: CovarianceComputer
    ) -> None:
        """A zero/negative 1D sigma must raise instead of producing inf weights."""
        sigma = jnp.array([0.1, 0.0, 0.2])

        with pytest.raises(ValueError, match="positive"):
            computer.create_sigma_transform(sigma=sigma, n_data=3)

    def test_setup_sigma_transform_rejects_non_positive_1d(
        self, computer: CovarianceComputer
    ) -> None:
        """Legacy setup_sigma_transform must also reject non-positive sigma."""
        sigma = np.array([0.1, -1.0, 0.2])
        ydata = np.zeros(3)
        mask = np.ones(3, dtype=bool)

        with pytest.raises(ValueError, match="positive"):
            computer.setup_sigma_transform(sigma, ydata, mask, len_diff=0, m=3)

    def test_create_sigma_transform_rejects_non_positive_definite_2d(
        self, computer: CovarianceComputer
    ) -> None:
        """A finite, correctly-shaped but non-PD 2D sigma must be rejected
        up front — not returned as a transform that silently produces NaN
        when the caller later runs Cholesky on it."""
        sigma = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric, eigenvalues -1 and 3

        with pytest.raises(ValueError, match="positive definite"):
            computer.create_sigma_transform(sigma=sigma, n_data=2)

    def test_create_sigma_transform_rejects_asymmetric_2d(
        self, computer: CovarianceComputer
    ) -> None:
        """A non-symmetric 2D sigma must be rejected."""
        sigma = jnp.array([[1.0, 5.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="symmetric"):
            computer.create_sigma_transform(sigma=sigma, n_data=2)

    def test_create_sigma_transform_rejects_2d_shape_mismatch(
        self, computer: CovarianceComputer
    ) -> None:
        """A 2D sigma whose shape doesn't match (n_data, n_data) must raise."""
        sigma = jnp.eye(3)

        with pytest.raises(ValueError, match="must be"):
            computer.create_sigma_transform(sigma=sigma, n_data=5)

    def test_create_sigma_transform_rejects_2d_non_finite(
        self, computer: CovarianceComputer
    ) -> None:
        """A 2D sigma containing NaN/Inf must raise."""
        sigma = jnp.array([[1.0, 0.0], [0.0, jnp.nan]])

        with pytest.raises(ValueError, match="non-finite"):
            computer.create_sigma_transform(sigma=sigma, n_data=2)

    def test_create_sigma_transform_rejects_1d_non_finite(
        self, computer: CovarianceComputer
    ) -> None:
        """A 1D sigma containing NaN/Inf must raise."""
        sigma = jnp.array([0.1, jnp.nan, 0.2])

        with pytest.raises(ValueError, match="non-finite"):
            computer.create_sigma_transform(sigma=sigma, n_data=3)

    def test_setup_sigma_transform_non_pd_2d_reports_min_eigenvalue(
        self, computer: CovarianceComputer
    ) -> None:
        """A non-PD 2D sigma routed through the legacy padded path must
        surface the precise min-eigenvalue message, not a generic one
        swallowed by a self-catching except block."""
        sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1, 3
        ydata = np.zeros(2)
        mask = np.ones(2, dtype=bool)

        with pytest.raises(ValueError, match="Minimum eigenvalue"):
            computer.setup_sigma_transform(sigma, ydata, mask, len_diff=0, m=2)

    def test_setup_sigma_transform_non_pd_2d_with_padding_raises(
        self, computer: CovarianceComputer
    ) -> None:
        """The padded (len_diff > 0) 2D path must also reject a non-PD sigma."""
        sigma = np.array([[1.0, 2.0], [2.0, 1.0]])
        ydata = np.zeros(3)  # padded: 2 real points + 1 pad row
        mask = np.array([True, True, False])

        with pytest.raises(ValueError, match="positive definite"):
            computer.setup_sigma_transform(sigma, ydata, mask, len_diff=1, m=2)


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

    def test_condition_number_n_data_matches_compute_threshold(
        self, computer: CovarianceComputer, simple_result: MockOptimizeResult
    ) -> None:
        """With n_data given, the threshold must match compute()'s condition_number."""
        computed = computer.compute(result=simple_result, n_data=5)
        cond = computer.compute_condition_number(simple_result.jac, n_data=5)

        assert_allclose(cond, computed.condition_number, rtol=1e-10)

    def test_condition_number_n_data_changes_threshold_for_padded_jacobian(
        self, computer: CovarianceComputer
    ) -> None:
        """n_data must actually change which singular values survive the
        threshold for a row-padded Jacobian — not just be accepted and
        ignored. Without n_data, the threshold scales with the padded
        jac.shape[0]; with the true (smaller) n_data, it doesn't."""
        # s0=1.0, s1=1e-13. Row-padded to 10_000 rows (streaming/chunking
        # style) but only 5 rows are real data.
        n_data = 5
        jac = jnp.zeros((10_000, 2))
        jac = jac.at[0, 0].set(1.0)
        jac = jac.at[1, 1].set(1e-13)

        # eps*5*1.0 ≈ 1.11e-15 < 1e-13: s1 survives -> both singular values used
        cond_with_n_data = computer.compute_condition_number(jac, n_data=n_data)
        # eps*10_000*1.0 ≈ 2.22e-12 > 1e-13: s1 filtered out -> only s0 survives
        cond_without_n_data = computer.compute_condition_number(jac)

        assert cond_with_n_data > 1e10
        assert_allclose(cond_without_n_data, 1.0, rtol=1e-6)

    def test_condition_number_none_jac_raises(
        self, computer: CovarianceComputer
    ) -> None:
        """A missing Jacobian must raise, not crash deep inside JAX."""
        with pytest.raises(ValueError, match="2-D"):
            computer.compute_condition_number(None)

    def test_condition_number_1d_jac_raises(self, computer: CovarianceComputer) -> None:
        """A 1-D 'Jacobian' must raise instead of an unguarded IndexError."""
        with pytest.raises(ValueError, match="2-D"):
            computer.compute_condition_number(jnp.array([1.0, 2.0, 3.0]))


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
