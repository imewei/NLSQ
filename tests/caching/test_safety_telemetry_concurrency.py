from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from nlsq.caching.memory_manager import MemoryManager

pytestmark = pytest.mark.serial


@pytest.mark.parametrize(
    "method", ["get_safety_telemetry", "_update_adaptive_safety_factor"]
)
def test_telemetry_reader_tolerates_concurrent_record(method):
    entered = Event()
    resume = Event()

    class BlockingEntry(dict):
        def __getitem__(self, key):
            entered.set()
            assert resume.wait(5)
            return super().__getitem__(key)

    manager = MemoryManager(enable_adaptive_safety=True)
    manager._warmup_runs = 1
    entry = {"safety_factor_needed": 1.0, "current_safety_factor": 1.2}
    manager._safety_telemetry.append(BlockingEntry(entry))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(getattr(manager, method))
        try:
            assert entered.wait(5)
            manager._safety_telemetry.append(entry)
        finally:
            resume.set()
        future.result(timeout=5)
