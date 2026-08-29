"""Tests for Phase 3 optimizations (Task Group 9).

Tests cover:
- Telemetry circular buffer with maxlen=1000
"""

import unittest
from collections import deque


class TestTelemetryCircularBuffer(unittest.TestCase):
    """Tests for Telemetry Circular Buffer (1.3a) in memory_manager.py."""

    def setUp(self):
        """Set up test fixtures."""
        from nlsq.caching.memory_manager import MemoryManager

        self.manager = MemoryManager(enable_adaptive_safety=True)

    def tearDown(self):
        """Clean up after tests."""
        self.manager.clear_pool()

    def test_safety_telemetry_is_deque_with_maxlen(self):
        """Test that _safety_telemetry is a deque with maxlen=1000."""
        self.assertIsInstance(self.manager._safety_telemetry, deque)
        self.assertEqual(self.manager._safety_telemetry.maxlen, 1000)

    def test_telemetry_buffer_bounds_at_1000(self):
        """Test that telemetry buffer never exceeds 1000 entries."""
        # Add more than 1000 telemetry records
        for i in range(1500):
            self.manager._record_safety_telemetry(
                bytes_predicted=1000 * (i + 1),
                bytes_actual=900 * (i + 1),
            )

        # Should be bounded at 1000
        self.assertEqual(len(self.manager._safety_telemetry), 1000)

    def test_telemetry_maintains_recent_records(self):
        """Test that circular buffer maintains most recent 1000 records."""
        # Add 1500 records
        for i in range(1500):
            self.manager._record_safety_telemetry(
                bytes_predicted=i,
                bytes_actual=i,
            )

        # Check that we have the most recent records (500-1499)
        self.assertEqual(len(self.manager._safety_telemetry), 1000)

        # First record should be from index 500
        self.assertEqual(self.manager._safety_telemetry[0]["bytes_predicted"], 500)

        # Last record should be from index 1499
        self.assertEqual(self.manager._safety_telemetry[-1]["bytes_predicted"], 1499)

    def test_telemetry_works_in_long_runs(self):
        """Test that telemetry does not grow unbounded in multi-day simulation."""
        # Simulate many optimization runs
        for i in range(10000):
            self.manager._record_safety_telemetry(
                bytes_predicted=1000,
                bytes_actual=900,
            )

        # Should never exceed maxlen
        self.assertLessEqual(len(self.manager._safety_telemetry), 1000)


if __name__ == "__main__":
    unittest.main()
