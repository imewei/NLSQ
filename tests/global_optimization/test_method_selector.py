"""Tests for MethodSelector class.

Tests cover:
- Method selection based on scale ratio
- evosax availability checking
- Fallback to multi-start when CMA-ES unavailable
- Auto mode selection logic
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class TestMethodSelectorBasic:
    """Tests for basic MethodSelector functionality."""

    def test_import(self) -> None:
        """Test that MethodSelector can be imported."""
        from nlsq.global_optimization.method_selector import MethodSelector

        assert MethodSelector is not None

    def test_instantiation(self) -> None:
        """Test basic instantiation."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()
        assert selector is not None

    def test_instantiation_with_scale_threshold(self) -> None:
        """Test instantiation with custom scale threshold."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector(scale_threshold=100.0)
        assert selector.scale_threshold == 100.0


class TestMethodSelectorScaleRatio:
    """Tests for scale ratio computation."""

    def test_compute_scale_ratio_uniform_bounds(self) -> None:
        """Test scale ratio for uniform bounds (ratio = 1)."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()
        lower = np.array([0, 0, 0])
        upper = np.array([1, 1, 1])

        ratio = selector.compute_scale_ratio(lower, upper)
        assert ratio == 1.0

    def test_compute_scale_ratio_multi_scale(self) -> None:
        """Test scale ratio for multi-scale bounds."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()
        # D0 ~ 1e4, gamma0 ~ 1e-3, n ~ 1
        # Ranges: 1e4, 1e-1, 2.5
        lower = np.array([1e2, 1e-4, 0.5])
        upper = np.array([1e6, 1e-1, 3.0])

        ratio = selector.compute_scale_ratio(lower, upper)
        # Max range / min range = 1e4 / 1e-3 = 1e7
        # Wait, let's compute: ranges = [1e6-1e2, 1e-1-1e-4, 3-0.5] = [~1e6, ~0.1, 2.5]
        # ratio = 1e6 / 0.1 = 1e7
        assert ratio > 1000  # Should indicate multi-scale

    def test_compute_scale_ratio_handles_zero_range(self) -> None:
        """Test scale ratio handles zero-width bounds gracefully."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()
        # One parameter is fixed (zero range)
        lower = np.array([0, 1, 0])
        upper = np.array([1, 1, 1])  # Middle param is fixed

        ratio = selector.compute_scale_ratio(lower, upper)
        # Should not raise, should handle zero range
        assert ratio >= 1.0


class TestMethodSelectorSelect:
    """Tests for method selection logic."""

    def test_select_explicit_cmaes_with_evosax(self) -> None:
        """Test that explicit 'cmaes' method is honored when evosax available."""
        from nlsq.global_optimization.cmaes_config import is_evosax_available
        from nlsq.global_optimization.method_selector import MethodSelector

        if not is_evosax_available():
            pytest.skip("evosax not installed")

        selector = MethodSelector()
        lower = np.array([0, 0])
        upper = np.array([1, 1])

        method = selector.select(
            requested_method="cmaes", lower_bounds=lower, upper_bounds=upper
        )
        assert method == "cmaes"

    def test_select_explicit_cmaes_without_evosax_falls_back(self) -> None:
        """Test that 'cmaes' falls back to multi-start when evosax unavailable."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()
        lower = np.array([0, 0])
        upper = np.array([1, 1])

        # Mock evosax as unavailable
        with patch(
            "nlsq.global_optimization.method_selector.is_evosax_available",
            return_value=False,
        ):
            method = selector.select(
                requested_method="cmaes", lower_bounds=lower, upper_bounds=upper
            )
            assert method == "multi-start"

    def test_select_explicit_multi_start(self) -> None:
        """Test that explicit 'multi-start' method is honored."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()
        lower = np.array([0, 0])
        upper = np.array([1, 1])

        method = selector.select(
            requested_method="multi-start", lower_bounds=lower, upper_bounds=upper
        )
        assert method == "multi-start"

    def test_select_auto_low_scale_prefers_multi_start(self) -> None:
        """Test that 'auto' prefers multi-start for low scale ratio."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector(scale_threshold=1000.0)
        # Uniform scale bounds
        lower = np.array([0, 0, 0])
        upper = np.array([1, 1, 1])

        method = selector.select(
            requested_method="auto", lower_bounds=lower, upper_bounds=upper
        )
        # Scale ratio = 1, below threshold of 1000, should use multi-start
        assert method == "multi-start"

    def test_select_auto_high_scale_prefers_cmaes(self) -> None:
        """Test that 'auto' prefers CMA-ES for high scale ratio."""
        from nlsq.global_optimization.cmaes_config import is_evosax_available
        from nlsq.global_optimization.method_selector import MethodSelector

        if not is_evosax_available():
            pytest.skip("evosax not installed")

        selector = MethodSelector(scale_threshold=1000.0)
        # Multi-scale bounds (>1000x range difference)
        lower = np.array([1e2, 1e-5, 0.5])
        upper = np.array([1e6, 1e-1, 3.0])

        method = selector.select(
            requested_method="auto", lower_bounds=lower, upper_bounds=upper
        )
        # Scale ratio > 1000, should prefer CMA-ES
        assert method == "cmaes"

    def test_select_auto_high_scale_falls_back_without_evosax(self) -> None:
        """Test that 'auto' falls back to multi-start even for high scale without evosax."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector(scale_threshold=1000.0)
        # Multi-scale bounds
        lower = np.array([1e2, 1e-5, 0.5])
        upper = np.array([1e6, 1e-1, 3.0])

        with patch(
            "nlsq.global_optimization.method_selector.is_evosax_available",
            return_value=False,
        ):
            method = selector.select(
                requested_method="auto", lower_bounds=lower, upper_bounds=upper
            )
            # Even though high scale, evosax unavailable, so multi-start
            assert method == "multi-start"

    def test_select_none_method_defaults_to_auto(self) -> None:
        """Test that None method is treated as 'auto'."""
        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector(scale_threshold=1000.0)
        lower = np.array([0, 0])
        upper = np.array([1, 1])

        method = selector.select(
            requested_method=None, lower_bounds=lower, upper_bounds=upper
        )
        # Should behave like 'auto' - low scale ratio means multi-start
        assert method == "multi-start"


class TestMethodSelectorLogging:
    """Tests for fallback logging."""

    def test_logs_info_on_cmaes_fallback(self, caplog) -> None:
        """Test that INFO log is emitted when falling back from CMA-ES."""
        import logging

        from nlsq.global_optimization.method_selector import MethodSelector

        selector = MethodSelector()
        lower = np.array([0, 0])
        upper = np.array([1, 1])

        with (
            caplog.at_level(
                logging.INFO, logger="nlsq.global_optimization.method_selector"
            ),
            patch(
                "nlsq.global_optimization.method_selector.is_evosax_available",
                return_value=False,
            ),
        ):
            selector.select(
                requested_method="cmaes", lower_bounds=lower, upper_bounds=upper
            )

        # Should log about fallback - look for "falling" (from "Falling back")
        assert any(
            "falling" in record.message.lower()
            or "multi-start" in record.message.lower()
            for record in caplog.records
        )
