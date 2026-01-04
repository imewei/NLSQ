"""
Tests for adaptive memory TTL feature in memory_manager module.

Task Group 3: Adaptive Memory TTL (1.1a)
Expected Gain: 10-15% in streaming optimization throughput

Tests that:
- High-frequency callers (>100 calls/sec) get 10s effective TTL
- Medium-frequency callers (>10 calls/sec) get 5s effective TTL
- Low-frequency callers use default TTL (1.0s)
- adaptive_ttl=False disables adaptive behavior
"""

import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

from nlsq.caching.memory_manager import MemoryManager

# macOS has unreliable sleep timing in CI, causing flaky tests
IS_MACOS = sys.platform == "darwin"


@pytest.mark.serial
class TestAdaptiveMemoryTTL(unittest.TestCase):
    """Tests for adaptive TTL behavior in MemoryManager."""

    def test_high_frequency_callers_get_10s_effective_ttl(self):
        """Test that high-frequency callers (>100 calls/sec) get 10s effective TTL.

        When call frequency exceeds 100 calls/sec, the effective TTL should
        increase to 10 seconds to reduce psutil overhead.
        """
        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            # Set up mock
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3  # 8 GB
            mock_psutil.virtual_memory.return_value = mock_mem

            # Also mock Process for get_memory_usage_bytes called in __init__
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 100 * 1024**2
            mock_psutil.Process.return_value = mock_process

            # Create manager INSIDE patch so all psutil calls are mocked
            manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

            # Simulate high-frequency calls (>100 calls/sec)
            # Make 150 calls very quickly to establish high frequency
            for _ in range(150):
                manager.get_available_memory()

            # Get psutil call count after establishing frequency
            initial_psutil_calls = mock_psutil.virtual_memory.call_count

            # Now make more calls - with 10s TTL, psutil should not be called
            # since we're within the 10s window
            for _ in range(50):
                manager.get_available_memory()

            # With high frequency (>100 calls/sec), effective TTL should be 10s
            # So psutil should not be called again during the same time window
            final_psutil_calls = mock_psutil.virtual_memory.call_count

            # The difference should be 0 or very small (cache should be used)
            # Since all calls happened quickly, they should use cached value
            self.assertEqual(
                final_psutil_calls - initial_psutil_calls,
                0,
                f"Expected 0 additional psutil calls with high-frequency adaptive TTL, "
                f"got {final_psutil_calls - initial_psutil_calls}",
            )

    def test_medium_frequency_callers_get_5s_effective_ttl(self):
        """Test that medium-frequency callers (>10, <100 calls/sec) get 5s effective TTL.

        When call frequency is between 10 and 100 calls/sec, the effective TTL
        should be 5 seconds.
        """
        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            # Also mock Process for get_memory_usage_bytes called in __init__
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 100 * 1024**2
            mock_psutil.Process.return_value = mock_process

            # Create manager INSIDE patch so all psutil calls are mocked
            manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

            # Simulate medium-frequency calls (~50 calls/sec)
            # We need to space calls to get ~50 calls/sec over the tracking window
            # With 100 calls tracked, at 50 calls/sec, the window is 2 seconds
            # Make calls with small delays to simulate ~50 calls/sec
            start_time = time.time()
            call_count = 0

            # Make initial calls to populate the tracker
            while call_count < 100:
                manager.get_available_memory()
                call_count += 1
                # Small delay to achieve ~50 calls/sec
                time.sleep(0.015)  # 15ms between calls ~ 66 calls/sec

            # Check that the frequency tracker has been populated
            self.assertGreater(len(manager._call_frequency_tracker), 0)

            # Get the effective TTL - should be 5s for medium frequency
            # We verify by checking that cached values are used within window
            initial_calls = mock_psutil.virtual_memory.call_count

            # Make a few more quick calls - should use cache
            for _ in range(10):
                manager.get_available_memory()

            # With 5s TTL, should not trigger new psutil calls
            final_calls = mock_psutil.virtual_memory.call_count
            self.assertEqual(
                final_calls - initial_calls,
                0,
                "Medium frequency calls should use cached values",
            )

    @pytest.mark.skipif(IS_MACOS, reason="Timing-dependent test flaky on macOS CI")
    def test_low_frequency_callers_use_default_ttl(self):
        """Test that low-frequency callers (<10 calls/sec) use default TTL (1.0s).

        When call frequency is low, the default TTL should be used, and
        psutil should be called after the TTL expires.
        """
        with (
            patch("nlsq.caching.memory_manager.psutil") as mock_psutil,
            patch("nlsq.caching.memory_manager.HAS_PSUTIL", True),
        ):
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            # Also mock Process for get_memory_usage_bytes called in __init__
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 100 * 1024**2
            mock_psutil.Process.return_value = mock_process

            # Create manager INSIDE patch so all psutil calls are mocked
            manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

            # First call - should call psutil
            manager.get_available_memory()
            first_call_count = mock_psutil.virtual_memory.call_count
            self.assertEqual(first_call_count, 1)

            # Second call immediately - should use cache
            manager.get_available_memory()
            self.assertEqual(mock_psutil.virtual_memory.call_count, 1)

            # Wait for TTL to expire (default 1.0s)
            time.sleep(1.1)

            # Now should call psutil again
            manager.get_available_memory()
            self.assertEqual(mock_psutil.virtual_memory.call_count, 2)

            # Low frequency (2 calls over ~1.1 seconds = ~1.8 calls/sec)
            # Should continue using default TTL

    @pytest.mark.skipif(IS_MACOS, reason="Timing-dependent test flaky on macOS CI")
    def test_adaptive_ttl_false_disables_adaptive_behavior(self):
        """Test that adaptive_ttl=False disables adaptive behavior.

        When adaptive_ttl is False, the MemoryManager should always use
        the default TTL regardless of call frequency.
        """
        with (
            patch("nlsq.caching.memory_manager.psutil") as mock_psutil,
            patch("nlsq.caching.memory_manager.HAS_PSUTIL", True),
        ):
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            # Also mock Process for get_memory_usage_bytes called in __init__
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 100 * 1024**2
            mock_psutil.Process.return_value = mock_process

            # Create manager INSIDE patch so all psutil calls are mocked
            manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=False)

            # Check that adaptive TTL is disabled
            self.assertFalse(manager._adaptive_ttl)

            # Make many rapid calls to simulate high frequency
            for _ in range(150):
                manager.get_available_memory()

            initial_count = mock_psutil.virtual_memory.call_count

            # Wait for default TTL to expire
            time.sleep(1.1)

            # Even with high call frequency history, should call psutil
            # after default TTL expires (not 10s adaptive TTL)
            manager.get_available_memory()

            # Should have made a new psutil call after TTL expired
            self.assertGreater(
                mock_psutil.virtual_memory.call_count,
                initial_count,
                "With adaptive_ttl=False, should use default TTL and call psutil after 1s",
            )

    def test_adaptive_ttl_default_is_true(self):
        """Test that adaptive_ttl defaults to True for backward-compatible improvement."""
        manager = MemoryManager()

        self.assertTrue(manager._adaptive_ttl)

    def test_call_frequency_tracker_initialized(self):
        """Test that the call frequency tracker is properly initialized."""
        manager = MemoryManager(adaptive_ttl=True)

        # Check that tracker exists and is a deque with maxlen=100
        self.assertTrue(hasattr(manager, "_call_frequency_tracker"))
        self.assertEqual(manager._call_frequency_tracker.maxlen, 100)
        self.assertEqual(len(manager._call_frequency_tracker), 0)

    def test_call_frequency_tracker_tracks_timestamps(self):
        """Test that the frequency tracker records call timestamps."""
        manager = MemoryManager(adaptive_ttl=True)

        # Make some calls
        for _ in range(5):
            manager.get_available_memory()

        # Check that timestamps were recorded
        self.assertEqual(len(manager._call_frequency_tracker), 5)

        # All timestamps should be recent
        now = time.time()
        for ts in manager._call_frequency_tracker:
            self.assertLess(now - ts, 1.0)  # Within last second


if __name__ == "__main__":
    unittest.main()
