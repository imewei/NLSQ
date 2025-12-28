"""Tests for memory detection system (Task Group 2).

This module tests the CPU and GPU memory detection capabilities for
the NLSQ workflow system, including:

- CPU memory detection via psutil (mock psutil.virtual_memory)
- GPU memory detection via JAX device API (mock jax.devices)
- Fallback to 16GB when CPU detection fails
- Fallback when GPU detection fails (containerized environments)
- Combined memory estimation (CPU + GPU)
- Memory cleanup function (gc.collect + jax.clear_caches)
"""

import gc
from unittest.mock import MagicMock, patch

import jax
import pytest

from nlsq.core.workflow import MemoryTier
from nlsq.streaming.large_dataset import (
    GPUMemoryEstimator,
    MemoryEstimator,
    cleanup_memory,
    get_memory_tier,
)


class TestCPUMemoryDetection:
    """Tests for CPU memory detection via psutil."""

    def test_cpu_memory_detection_via_psutil(self):
        """Test CPU memory detection via psutil.virtual_memory."""
        # Create a mock memory info object
        mock_memory = MagicMock()
        mock_memory.available = 32 * 1024**3  # 32 GB
        mock_memory.total = 64 * 1024**3  # 64 GB

        with patch("psutil.virtual_memory", return_value=mock_memory):
            available_gb = MemoryEstimator.get_available_memory_gb()

            # Should return approximately 32 GB
            assert 31.9 <= available_gb <= 32.1, f"Expected ~32 GB, got {available_gb}"

    def test_fallback_to_16gb_when_detection_fails(self):
        """Test fallback to 16GB when CPU detection fails."""
        # Mock psutil.virtual_memory to raise an exception
        with patch("psutil.virtual_memory", side_effect=Exception("Detection failed")):
            available_gb = MemoryEstimator.get_available_memory_gb()

            # Should fall back to 16 GB (updated from 4 GB per requirements)
            assert available_gb == 16.0, (
                f"Expected 16.0 GB fallback, got {available_gb}"
            )

    def test_cpu_memory_detection_returns_positive(self):
        """Test that CPU memory detection always returns a positive value."""
        available_gb = MemoryEstimator.get_available_memory_gb()

        # Should always return a positive value
        assert available_gb > 0, "Available memory should be positive"


class TestGPUMemoryDetection:
    """Tests for GPU memory detection via JAX device API."""

    def test_gpu_memory_detection_via_jax_device_api(self):
        """Test GPU memory detection via jax.devices()[0].memory_stats()."""
        # Create a mock GPU device with memory_stats
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_device.memory_stats.return_value = {
            "bytes_limit": 16 * 1024**3,  # 16 GB total
            "bytes_in_use": 4 * 1024**3,  # 4 GB in use
        }

        mock_devices = [mock_device]

        with patch.object(jax, "devices", return_value=mock_devices):
            estimator = GPUMemoryEstimator()
            available_gb = estimator.get_available_gpu_memory_gb()

            # Should return approximately 12 GB (16 - 4)
            assert 11.9 <= available_gb <= 12.1, f"Expected ~12 GB, got {available_gb}"

    def test_fallback_when_gpu_detection_fails(self):
        """Test fallback when GPU detection fails (containerized environments)."""
        # Create a mock device that raises an exception on memory_stats
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_device.memory_stats.side_effect = Exception("No memory stats available")

        mock_devices = [mock_device]

        with patch.object(jax, "devices", return_value=mock_devices):
            estimator = GPUMemoryEstimator()
            available_gb = estimator.get_available_gpu_memory_gb()

            # Should return 0 when GPU detection fails
            assert available_gb == 0.0, f"Expected 0 GB on failure, got {available_gb}"

    def test_cpu_only_environment_returns_zero(self):
        """Test that CPU-only environments return 0 for GPU memory."""
        # Create a mock CPU-only device
        mock_device = MagicMock()
        mock_device.platform = "cpu"
        mock_device.device_kind = "cpu"

        mock_devices = [mock_device]

        with patch.object(jax, "devices", return_value=mock_devices):
            estimator = GPUMemoryEstimator()
            available_gb = estimator.get_available_gpu_memory_gb()

            # Should return 0 for CPU-only environment
            assert available_gb == 0.0, (
                f"Expected 0 GB for CPU-only, got {available_gb}"
            )

    def test_multiple_gpus_aggregate_memory(self):
        """Test that multiple GPUs aggregate available memory."""
        # Create mock GPU devices
        mock_gpu1 = MagicMock()
        mock_gpu1.platform = "gpu"
        mock_gpu1.memory_stats.return_value = {
            "bytes_limit": 16 * 1024**3,
            "bytes_in_use": 4 * 1024**3,
        }

        mock_gpu2 = MagicMock()
        mock_gpu2.platform = "gpu"
        mock_gpu2.memory_stats.return_value = {
            "bytes_limit": 16 * 1024**3,
            "bytes_in_use": 2 * 1024**3,
        }

        mock_devices = [mock_gpu1, mock_gpu2]

        with patch.object(jax, "devices", return_value=mock_devices):
            estimator = GPUMemoryEstimator()
            available_gb = estimator.get_available_gpu_memory_gb()

            # Should aggregate: (16-4) + (16-2) = 12 + 14 = 26 GB
            assert 25.9 <= available_gb <= 26.1, f"Expected ~26 GB, got {available_gb}"


class TestCombinedMemoryEstimation:
    """Tests for combined CPU + GPU memory estimation."""

    def test_combined_memory_estimation_cpu_and_gpu(self):
        """Test combined memory estimation (CPU + GPU)."""
        # Mock CPU memory
        mock_cpu_memory = MagicMock()
        mock_cpu_memory.available = 64 * 1024**3  # 64 GB

        # Mock GPU device
        mock_gpu = MagicMock()
        mock_gpu.platform = "gpu"
        mock_gpu.memory_stats.return_value = {
            "bytes_limit": 40 * 1024**3,  # 40 GB total
            "bytes_in_use": 8 * 1024**3,  # 8 GB in use
        }

        with (
            patch("psutil.virtual_memory", return_value=mock_cpu_memory),
            patch.object(jax, "devices", return_value=[mock_gpu]),
        ):
            total_gb = MemoryEstimator.get_total_available_memory_gb()

            # CPU (64) + GPU available (40-8=32) = 96 GB
            assert 95.9 <= total_gb <= 96.1, f"Expected ~96 GB, got {total_gb}"

    def test_combined_memory_cpu_only(self):
        """Test combined memory when no GPU available."""
        # Mock CPU memory
        mock_cpu_memory = MagicMock()
        mock_cpu_memory.available = 32 * 1024**3  # 32 GB

        # Mock CPU-only device
        mock_cpu_device = MagicMock()
        mock_cpu_device.platform = "cpu"

        with (
            patch("psutil.virtual_memory", return_value=mock_cpu_memory),
            patch.object(jax, "devices", return_value=[mock_cpu_device]),
        ):
            total_gb = MemoryEstimator.get_total_available_memory_gb()

            # CPU only: 32 GB (GPU = 0)
            assert 31.9 <= total_gb <= 32.1, f"Expected ~32 GB, got {total_gb}"

    def test_memory_estimation_no_caching(self):
        """Test that memory estimation re-evaluates on each call (no caching)."""
        call_count = 0

        def mock_virtual_memory():
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.available = (32 + call_count) * 1024**3  # Increase each call
            return mock

        with patch("psutil.virtual_memory", side_effect=mock_virtual_memory):
            # Call multiple times
            result1 = MemoryEstimator.get_available_memory_gb()
            result2 = MemoryEstimator.get_available_memory_gb()
            result3 = MemoryEstimator.get_available_memory_gb()

            # Each call should have triggered virtual_memory
            assert call_count == 3, f"Expected 3 calls, got {call_count}"

            # Results should be different (not cached)
            assert result1 != result2, "Memory values should not be cached"
            assert result2 != result3, "Memory values should not be cached"


class TestMemoryCleanup:
    """Tests for memory cleanup function."""

    def test_memory_cleanup_calls_gc_collect(self):
        """Test that memory cleanup calls gc.collect()."""
        with patch.object(gc, "collect") as mock_gc:
            cleanup_memory()

            # gc.collect should have been called
            mock_gc.assert_called_once()

    def test_memory_cleanup_calls_jax_clear_caches(self):
        """Test that memory cleanup calls jax.clear_caches()."""
        with (
            patch.object(gc, "collect"),
            patch.object(jax, "clear_caches") as mock_clear,
        ):
            cleanup_memory()

            # jax.clear_caches should have been called
            mock_clear.assert_called_once()

    def test_memory_cleanup_graceful_on_error(self):
        """Test that memory cleanup handles errors gracefully."""
        with (
            patch.object(gc, "collect"),
            patch.object(jax, "clear_caches", side_effect=Exception("Cache error")),
        ):
            # Should not raise exception
            cleanup_memory()  # No exception = test passes


class TestMemoryTierClassification:
    """Tests for MemoryTier classification function."""

    def test_memory_tier_low(self):
        """Test LOW memory tier classification (<16GB)."""
        tier = get_memory_tier(8.0)
        assert tier == MemoryTier.LOW

        tier = get_memory_tier(15.9)
        assert tier == MemoryTier.LOW

    def test_memory_tier_medium(self):
        """Test MEDIUM memory tier classification (16-64GB)."""
        tier = get_memory_tier(16.0)
        assert tier == MemoryTier.MEDIUM

        tier = get_memory_tier(32.0)
        assert tier == MemoryTier.MEDIUM

        tier = get_memory_tier(63.9)
        assert tier == MemoryTier.MEDIUM

    def test_memory_tier_high(self):
        """Test HIGH memory tier classification (64-128GB)."""
        tier = get_memory_tier(64.0)
        assert tier == MemoryTier.HIGH

        tier = get_memory_tier(100.0)
        assert tier == MemoryTier.HIGH

        tier = get_memory_tier(127.9)
        assert tier == MemoryTier.HIGH

    def test_memory_tier_very_high(self):
        """Test VERY_HIGH memory tier classification (>128GB)."""
        tier = get_memory_tier(128.0)
        assert tier == MemoryTier.VERY_HIGH

        tier = get_memory_tier(256.0)
        assert tier == MemoryTier.VERY_HIGH

        tier = get_memory_tier(1024.0)
        assert tier == MemoryTier.VERY_HIGH

    def test_memory_tier_thresholds(self):
        """Test MemoryTier uses correct thresholds: 16GB, 64GB, 128GB."""
        # Verify the thresholds are as specified in requirements
        assert MemoryTier.LOW.max_memory_gb == 16.0
        assert MemoryTier.MEDIUM.max_memory_gb == 64.0
        assert MemoryTier.HIGH.max_memory_gb == 128.0
        assert MemoryTier.VERY_HIGH.max_memory_gb == float("inf")
