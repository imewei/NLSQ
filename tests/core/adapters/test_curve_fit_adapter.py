"""Tests for CurveFitAdapter protocol conformance."""

from unittest.mock import patch

import numpy as np
import pytest


class TestCurveFitAdapterProtocol:
    """Tests for CurveFitAdapter protocol conformance.

    Note: Protocol conformance assertion was moved here from module-level
    in nlsq/core/adapters/curve_fit_adapter.py to avoid import-time overhead.
    """

    def test_protocol_conformance(self):
        """Verify CurveFitAdapter implements CurveFitProtocol."""
        from nlsq.core.adapters.curve_fit_adapter import CurveFitAdapter
        from nlsq.interfaces.optimizer_protocol import CurveFitProtocol

        adapter = CurveFitAdapter()
        assert isinstance(adapter, CurveFitProtocol), (
            "CurveFitAdapter must implement CurveFitProtocol"
        )

    def test_adapter_is_callable(self):
        """Verify adapter instance is callable."""
        from nlsq.core.adapters.curve_fit_adapter import CurveFitAdapter

        adapter = CurveFitAdapter()
        assert adapter is not None
        # Adapter should have curve_fit method
        assert hasattr(adapter, "curve_fit")
        assert callable(adapter.curve_fit)


class TestCurveFitAdapterBugFixRegressions:
    """Regression tests for the three-brain-review bug fixes to
    CurveFitAdapter: bounds=None crashing, with_global_optimization()
    silently never activating, and sigma being dropped in the global path.
    """

    def test_curve_fit_no_bounds_does_not_crash(self):
        """curve_fit()'s own default bounds is (-inf, inf), never None;
        the adapter used to pass bounds=None straight through, which
        crashes in prepare_bounds()'s `for b in bounds` on the default
        no-bounds call. This is a separate bounds=None guard from the one
        in nlsq/core/factories.py -- exercising ConfiguredOptimizer or
        configure_curve_fit does not cover this call site."""
        from nlsq.core.adapters.curve_fit_adapter import CurveFitAdapter

        def model(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 20)
        y = 2.5 * x + 1.0

        adapter = CurveFitAdapter()
        popt, pcov = adapter.curve_fit(model, x, y, p0=np.array([1.0, 0.0]))
        assert popt is not None
        assert pcov is not None

    def test_with_global_optimization_actually_routes_through_it(self):
        """with_global_optimization() used to only store `_global_config`
        on the adapter; curve_fit() never read it, so global optimization
        silently never activated -- every call fell through to the
        standard local curve_fit() path regardless of the stored config.
        Spy on MultiStartOrchestrator.fit to verify the adapter actually
        routes through it now."""
        from nlsq.core.adapters.curve_fit_adapter import CurveFitAdapter
        from nlsq.global_optimization.config import GlobalOptimizationConfig
        from nlsq.global_optimization.multi_start import MultiStartOrchestrator

        def model(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 20)
        y = 2.5 * x + 1.0

        adapter = CurveFitAdapter.with_global_optimization(
            GlobalOptimizationConfig(n_starts=3)
        )

        with patch.object(
            MultiStartOrchestrator, "fit", return_value={"popt": np.array([1.0, 0.0])}
        ) as mock_fit:
            adapter.curve_fit(model, x, y, p0=np.array([1.0, 0.0]))

        mock_fit.assert_called_once()

    def test_without_global_optimization_does_not_route_through_it(self):
        """A plain CurveFitAdapter() (no with_global_optimization) must
        never touch MultiStartOrchestrator -- the standard local path."""
        from nlsq.core.adapters.curve_fit_adapter import CurveFitAdapter
        from nlsq.global_optimization.multi_start import MultiStartOrchestrator

        def model(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 20)
        y = 2.5 * x + 1.0

        adapter = CurveFitAdapter()

        with patch.object(MultiStartOrchestrator, "fit") as mock_fit:
            adapter.curve_fit(model, x, y, p0=np.array([1.0, 0.0]))

        mock_fit.assert_not_called()

    def test_global_optimization_sigma_is_forwarded_not_dropped(self):
        """The global-optimization branch used to silently drop a
        caller-supplied sigma: it's a named parameter of curve_fit()'s own
        signature (so it never lands in **kwargs), and the old code never
        forwarded it into the MultiStartOrchestrator.fit() call. Verify
        sigma reaches that call now."""
        from nlsq.core.adapters.curve_fit_adapter import CurveFitAdapter
        from nlsq.global_optimization.config import GlobalOptimizationConfig
        from nlsq.global_optimization.multi_start import MultiStartOrchestrator

        def model(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 20)
        y = 2.5 * x + 1.0
        sigma = np.ones_like(y)

        adapter = CurveFitAdapter.with_global_optimization(
            GlobalOptimizationConfig(n_starts=3)
        )

        with patch.object(
            MultiStartOrchestrator, "fit", return_value={"popt": np.array([1.0, 0.0])}
        ) as mock_fit:
            adapter.curve_fit(model, x, y, p0=np.array([1.0, 0.0]), sigma=sigma)

        mock_fit.assert_called_once()
        np.testing.assert_array_equal(mock_fit.call_args.kwargs["sigma"], sigma)
