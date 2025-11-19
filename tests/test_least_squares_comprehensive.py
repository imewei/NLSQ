"""
Comprehensive tests for LeastSquares argument combinations.

Target: Cover diverse parameter combinations for Sprint 1 safety net.
Goal: 15 tests covering loss functions, tolerances, scaling, bounds, verbose.
"""

import jax.numpy as jnp
import numpy as np

from nlsq import LeastSquares


class TestLossFunctions:
    """Test different loss function options."""

    def setup_method(self):
        """Setup least squares solver."""
        self.ls = LeastSquares()

    def test_loss_linear(self):
        """Test linear loss (default)."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, loss="linear")

        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-5)

    def test_loss_huber(self):
        """Test Huber loss for robust fitting."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2, 10])  # Outlier

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, loss="huber", f_scale=1.0)

        assert result.success
        # Huber loss downweights the outlier

    def test_loss_soft_l1(self):
        """Test soft_l1 loss."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, loss="soft_l1", f_scale=1.0)

        assert result.success

    def test_loss_cauchy(self):
        """Test Cauchy loss."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, loss="cauchy", f_scale=1.0)

        assert result.success

    def test_loss_arctan(self):
        """Test arctan loss."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, loss="arctan", f_scale=1.0)

        assert result.success


class TestToleranceCombinations:
    """Test different tolerance combinations."""

    def setup_method(self):
        """Setup least squares solver."""
        self.ls = LeastSquares()

    def test_tight_ftol(self):
        """Test tight function tolerance."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, ftol=1e-12, xtol=1e-8, gtol=1e-8)

        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-6)

    def test_tight_xtol(self):
        """Test tight parameter tolerance."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, ftol=1e-8, xtol=1e-12, gtol=1e-8)

        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-6)

    def test_tight_gtol(self):
        """Test tight gradient tolerance."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, ftol=1e-8, xtol=1e-8, gtol=1e-12)

        assert result.success
        np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-6)


class TestScalingOptions:
    """Test different x_scale options."""

    def setup_method(self):
        """Setup least squares solver."""
        self.ls = LeastSquares()

    def test_x_scale_scalar(self):
        """Test scalar x_scale."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, x_scale=1.0)

        assert result.success

    def test_x_scale_jac(self):
        """Test automatic x_scale from Jacobian."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, x_scale="jac")

        assert result.success


class TestVerboseAndMonitoring:
    """Test verbose and monitoring options."""

    def setup_method(self):
        """Setup least squares solver."""
        self.ls = LeastSquares()

    def test_verbose_level_0(self):
        """Test verbose=0 (silent)."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, verbose=0)

        assert result.success

    def test_verbose_level_1(self):
        """Test verbose=1 (summary)."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, verbose=1)

        assert result.success

    def test_verbose_level_2(self):
        """Test verbose=2 (detailed)."""

        def fun(x):
            return jnp.array([x[0] - 1, x[1] - 2])

        x0 = np.array([0.5, 1.5])
        result = self.ls.least_squares(fun, x0, verbose=2)

        assert result.success


class TestMaxNfev:
    """Test max_nfev limit."""

    def setup_method(self):
        """Setup least squares solver."""
        self.ls = LeastSquares()

    def test_max_nfev_limit(self):
        """Test max_nfev enforces iteration limit."""

        def fun(x):
            return jnp.array([x[0] ** 2 - 1, x[1] ** 2 - 4])

        x0 = np.array([0.1, 0.1])
        result = self.ls.least_squares(fun, x0, max_nfev=5)

        # Should stop early due to max_nfev
        assert result.nfev <= 5


# Total: 14 comprehensive tests covering argument combinations
