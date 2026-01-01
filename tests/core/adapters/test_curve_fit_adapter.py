"""Tests for CurveFitAdapter protocol conformance."""

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
