"""Outstanding allocations must retain independent storage."""

import numpy as np

from nlsq.caching.memory_manager import MemoryManager


def test_nested_temporary_allocations_do_not_overwrite_outer_array():
    manager = MemoryManager()
    with manager.temporary_allocation((4,)) as outer:
        outer[:] = 7
        with manager.temporary_allocation((4,)) as inner:
            inner[:] = 3
            np.testing.assert_array_equal(outer, np.full(4, 7))
            assert not np.shares_memory(outer, inner)
    with manager.temporary_allocation((4,)) as reused:
        assert reused is outer
        np.testing.assert_array_equal(reused, np.zeros(4))


def test_pooled_array_is_not_handed_out_twice():
    manager = MemoryManager()
    a = manager.allocate_array((4,))
    manager.free_array(a)
    b = manager.allocate_array((4,))  # pool hit: b is a
    c = manager.allocate_array((4,))  # must be fresh while b is checked out
    assert not np.shares_memory(b, c)
