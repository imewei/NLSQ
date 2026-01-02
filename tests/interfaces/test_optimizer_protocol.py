"""Protocol contract tests for OptimizerProtocol.

This module tests that optimizer implementations conform to the
OptimizerProtocol, LeastSquaresOptimizerProtocol, and CurveFitProtocol
defined in nlsq.interfaces.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from nlsq.interfaces.optimizer_protocol import (
    CurveFitProtocol,
    LeastSquaresOptimizerProtocol,
    OptimizerProtocol,
)


class TestOptimizerProtocolDefinition:
    """Test that OptimizerProtocol is correctly defined."""

    def test_protocol_is_runtime_checkable(self):
        """OptimizerProtocol should be runtime_checkable."""

        # The @runtime_checkable decorator allows isinstance() checks
        class MockOptimizer:
            def optimize(self, fun, x0, args=(), **kwargs):
                return None

        assert isinstance(MockOptimizer(), OptimizerProtocol)

    def test_protocol_requires_optimize_method(self):
        """Classes missing optimize() should not satisfy protocol."""

        class IncompleteOptimizer:
            def other_method(self):
                pass

        assert not isinstance(IncompleteOptimizer(), OptimizerProtocol)

    def test_minimal_implementation(self):
        """Minimal implementation should satisfy protocol."""

        class MinimalOptimizer:
            def optimize(
                self,
                fun: Callable[..., np.ndarray],
                x0: np.ndarray,
                args: tuple[Any, ...] = (),
                **kwargs: Any,
            ) -> Any:
                return {"x": x0}

        optimizer = MinimalOptimizer()
        assert isinstance(optimizer, OptimizerProtocol)

        # Verify the method works
        result = optimizer.optimize(lambda x: x, np.array([1.0]))
        assert "x" in result


class TestLeastSquaresOptimizerProtocol:
    """Test LeastSquaresOptimizerProtocol conformance."""

    def test_protocol_is_runtime_checkable(self):
        """LeastSquaresOptimizerProtocol should be runtime_checkable."""

        class MockLSOptimizer:
            def least_squares(self, fun, x0, jac=None, bounds=None, args=(), **kwargs):
                return {"x": x0}

        assert isinstance(MockLSOptimizer(), LeastSquaresOptimizerProtocol)

    def test_protocol_requires_least_squares_method(self):
        """Classes missing least_squares() should not satisfy protocol."""

        class IncompleteOptimizer:
            def optimize(self, fun, x0, args=(), **kwargs):
                pass

        assert not isinstance(IncompleteOptimizer(), LeastSquaresOptimizerProtocol)

    def test_full_implementation(self):
        """Full implementation with all parameters should satisfy protocol."""

        class FullLSOptimizer:
            def least_squares(
                self,
                fun: Callable[..., np.ndarray],
                x0: np.ndarray,
                jac: Callable[..., np.ndarray] | str | None = None,
                bounds: tuple[np.ndarray, np.ndarray] | None = None,
                args: tuple[Any, ...] = (),
                **kwargs: Any,
            ) -> Any:
                return {"x": x0, "success": True}

        optimizer = FullLSOptimizer()
        assert isinstance(optimizer, LeastSquaresOptimizerProtocol)

        # Test with bounds
        lower = np.array([-1.0])
        upper = np.array([1.0])
        result = optimizer.least_squares(
            lambda x: x,
            np.array([0.0]),
            bounds=(lower, upper),
        )
        assert result["success"]


class TestCurveFitProtocol:
    """Test CurveFitProtocol conformance."""

    def test_protocol_is_runtime_checkable(self):
        """CurveFitProtocol should be runtime_checkable."""

        class MockCurveFitter:
            def curve_fit(
                self, f, xdata, ydata, p0=None, sigma=None, bounds=None, **kwargs
            ):
                return np.array([1.0]), np.array([[1.0]])

        assert isinstance(MockCurveFitter(), CurveFitProtocol)

    def test_protocol_requires_curve_fit_method(self):
        """Classes missing curve_fit() should not satisfy protocol."""

        class IncompleteFitter:
            def fit(self, f, x, y):
                pass

        assert not isinstance(IncompleteFitter(), CurveFitProtocol)

    def test_curve_fit_adapter_conforms(self):
        """CurveFitAdapter should conform to CurveFitProtocol."""
        from nlsq.core.adapters import CurveFitAdapter

        adapter = CurveFitAdapter()
        assert isinstance(adapter, CurveFitProtocol)

    def test_minimal_curve_fitter(self):
        """Minimal curve fitter implementation."""

        class MinimalFitter:
            def curve_fit(
                self,
                f: Callable[..., np.ndarray],
                xdata: np.ndarray,
                ydata: np.ndarray,
                p0: np.ndarray | None = None,
                sigma: np.ndarray | None = None,
                bounds: tuple[np.ndarray, np.ndarray] | None = None,
                **kwargs: Any,
            ) -> tuple[np.ndarray, np.ndarray]:
                # Simple constant fit for testing
                return np.array([np.mean(ydata)]), np.array([[1.0]])

        fitter = MinimalFitter()
        assert isinstance(fitter, CurveFitProtocol)

        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0, 6.0])
        popt, _pcov = fitter.curve_fit(lambda x, a: a * x, xdata, ydata)
        assert popt.shape == (1,)


class TestConcreteImplementations:
    """Test that concrete NLSQ implementations satisfy protocols."""

    def test_curve_fit_class_has_curve_fit(self):
        """CurveFit class should have a curve_fit method."""
        from nlsq.core.minpack import CurveFit

        fitter = CurveFit()
        assert hasattr(fitter, "curve_fit")
        assert callable(fitter.curve_fit)

    def test_least_squares_class_exists(self):
        """LeastSquares class should exist."""
        from nlsq.core.least_squares import LeastSquares

        ls = LeastSquares()
        assert hasattr(ls, "least_squares")


class TestProtocolDuckTyping:
    """Test duck typing behavior of protocols."""

    def test_unrelated_class_with_matching_signature(self):
        """Any class with matching method should satisfy protocol."""

        class ThirdPartyOptimizer:
            """Simulates a third-party optimizer."""

            def optimize(self, fun, x0, args=(), **kwargs):
                # Some other optimizer implementation
                return {"x": x0, "fun": fun(x0)}

        optimizer = ThirdPartyOptimizer()
        assert isinstance(optimizer, OptimizerProtocol)

    def test_lambda_function_not_protocol(self):
        """Lambda functions should not satisfy class protocols."""
        optimize_func = lambda fun, x0, args=(): {"x": x0}
        assert not isinstance(optimize_func, OptimizerProtocol)
