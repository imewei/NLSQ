from nlsq.caching.memory_manager import MemoryManager


def test_peak_never_below_current():
    s = MemoryManager().get_memory_stats()
    assert s["peak_usage_gb"] >= s["current_usage_gb"] > 0
