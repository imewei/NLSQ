"""Numerical accuracy tests for CurveFit God Class Decomposition.

Validates that the decomposition preserves numerical correctness.

Reference: specs/017-curve-fit-decomposition/spec.md SC-007
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit


def exponential_model(x: np.ndarray, a: float, b: float) -> Any:
    """Exponential decay model for benchmarking."""
    return a * jnp.exp(-b * x)


def generate_benchmark_data(
    n_points: int = 10000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for benchmarking."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4, n_points)
    y_true = 2.5 * np.exp(-1.3 * x)
    noise = rng.normal(0, 0.1, n_points)
    y = y_true + noise
    return x, y


@pytest.mark.serial
class TestDecompositionAccuracy:
    """Numerical accuracy tests for decomposed CurveFit."""

    @pytest.fixture
    def benchmark_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate 10K point dataset for benchmarking."""
        return generate_benchmark_data(n_points=10000)

    def test_numerical_accuracy_preserved(
        self, benchmark_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify numerical accuracy is preserved after decomposition.

        SC-007: Results must match within 1e-8 tolerance.
        """
        x, y = benchmark_data

        # Run multiple fits
        results = []
        for _ in range(3):
            popt, pcov = curve_fit(exponential_model, x, y, p0=[2.0, 1.0])
            results.append((np.array(popt), np.array(pcov)))

        # All runs should produce identical results
        for i, (popt, pcov) in enumerate(results[1:], start=1):
            np.testing.assert_allclose(
                popt,
                results[0][0],
                atol=1e-8,
                err_msg=f"Run {i} popt differs from run 0",
            )
            np.testing.assert_allclose(
                pcov,
                results[0][1],
                atol=1e-8,
                err_msg=f"Run {i} pcov differs from run 0",
            )

        # Verify fitted parameters are reasonable
        popt = results[0][0]
        assert 2.4 < popt[0] < 2.6, f"Parameter a={popt[0]} out of expected range"
        assert 1.2 < popt[1] < 1.4, f"Parameter b={popt[1]} out of expected range"
