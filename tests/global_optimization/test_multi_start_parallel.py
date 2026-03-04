"""Tests for parallel multi-start optimization.

Tests for _fit_single_start() worker function, _select_worker_count()
adaptive hardware detection, and parallel ThreadPoolExecutor integration
in evaluate_starting_points().
"""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.unit
def test_fit_single_start_returns_tuple() -> None:
    """Worker function returns (params, loss, result) tuple."""
    from nlsq.global_optimization.multi_start import _fit_single_start

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    rng = np.random.default_rng(42)
    x = np.linspace(0, 5, 50)
    y_true = 3.0 * np.exp(-0.5 * x)
    y = y_true + rng.normal(0, 0.05, 50)

    params, loss, result = _fit_single_start(
        model, x, y, p0=np.array([2.0, 0.3]), bounds=([0, 0], [10, 5]), kwargs={}
    )

    assert params.shape == (2,)
    assert loss < float("inf")
    assert result is not None


@pytest.mark.unit
def test_fit_single_start_exception_returns_inf() -> None:
    """Worker function returns inf loss on failure."""
    from nlsq.global_optimization.multi_start import _fit_single_start

    def bad_model(x, a):
        raise RuntimeError("always fails")

    x = np.linspace(0, 1, 10)
    y = np.ones(10)
    p0 = np.array([1.0])

    params, loss, result = _fit_single_start(
        bad_model, x, y, p0=p0, bounds=(-np.inf, np.inf), kwargs={}
    )

    assert loss == float("inf")
    assert result is None
    assert np.array_equal(params, p0)


@pytest.mark.unit
class TestSelectWorkerCount:
    def test_single_gpu_caps_at_4(self) -> None:
        from nlsq.global_optimization.multi_start import _select_worker_count

        with patch("nlsq.global_optimization.multi_start.jax.devices") as mock_devices:
            mock_devices.return_value = ["gpu0"]
            assert _select_worker_count(10) == 4
            assert _select_worker_count(2) == 2

    def test_multi_gpu_matches_device_count(self) -> None:
        from nlsq.global_optimization.multi_start import _select_worker_count

        with patch("nlsq.global_optimization.multi_start.jax.devices") as mock_devices:
            mock_devices.return_value = ["gpu0", "gpu1", "gpu2", "gpu3"]
            assert _select_worker_count(10) == 4
            assert _select_worker_count(2) == 2

    def test_cpu_only_uses_cores(self) -> None:
        from nlsq.global_optimization.multi_start import _select_worker_count

        with (
            patch("nlsq.global_optimization.multi_start.jax.devices") as mock_devices,
            patch("os.cpu_count", return_value=8),
        ):
            mock_devices.return_value = []
            assert _select_worker_count(20) == 8
            assert _select_worker_count(4) == 4

    def test_single_start_returns_1(self) -> None:
        from nlsq.global_optimization.multi_start import _select_worker_count

        with patch("nlsq.global_optimization.multi_start.jax.devices") as mock_devices:
            mock_devices.return_value = ["gpu0"]
            assert _select_worker_count(1) == 1


@pytest.mark.unit
def test_parallel_evaluate_reports_diagnostics() -> None:
    """Parallel evaluation reports parallel diagnostics."""
    from nlsq.global_optimization import (
        GlobalOptimizationConfig,
        MultiStartOrchestrator,
    )

    rng = np.random.default_rng(42)

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    x = np.linspace(0, 5, 100)
    y = 3.0 * np.exp(-0.5 * x) + rng.normal(0, 0.05, 100)

    config = GlobalOptimizationConfig(n_starts=4, sampler="lhs")
    orch = MultiStartOrchestrator(config=config)
    result = orch.fit(model, x, y, bounds=([0, 0], [10, 5]))

    diag = result.get("multistart_diagnostics", {})
    if hasattr(result, "multistart_diagnostics"):
        diag = result.multistart_diagnostics

    # Should have parallel diagnostics keys
    assert "parallel" in diag
    assert "n_workers" in diag
    assert "wall_time_sec" in diag
    assert diag["n_workers"] >= 1
    assert diag["wall_time_sec"] > 0
