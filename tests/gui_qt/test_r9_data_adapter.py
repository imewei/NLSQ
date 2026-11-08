"""Round-9 regressions: GUI sigma validation, semicolon CSV, stats race."""

import threading

import numpy as np

from nlsq.caching.memory_manager import MemoryManager
from nlsq.gui_qt.adapters.data_adapter import (
    detect_delimiter,
    parse_table_text,
    validate_data,
)


def test_validate_data_rejects_non_positive_sigma():
    x = np.arange(5.0)
    for bad in (0.0, -1.0):
        s = np.ones(5)
        s[2] = bad
        res = validate_data(x, x, s)
        assert not res.is_valid and "sigma" in res.message
    assert validate_data(x, x, np.ones(5)).is_valid


def test_semicolon_csv_loads():
    assert detect_delimiter("x;y\n1;2\n3;4") == ";"
    rows, header = parse_table_text("x;y\n1;2\n3;4")
    assert header == ["x", "y"] and rows == [[1.0, 2.0], [3.0, 4.0]]


def test_get_memory_stats_survives_concurrent_allocations():
    mm = MemoryManager()
    stop = threading.Event()

    def spam():
        while not stop.is_set():
            mm.allocate_array((8,), np.float64)

    t = threading.Thread(target=spam)
    t.start()
    try:
        for _ in range(300):
            mm.get_memory_stats()
    finally:
        stop.set()
        t.join()
