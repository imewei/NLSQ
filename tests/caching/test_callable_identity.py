"""Callable equality must not merge behaviorally different cache entries."""

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.caching.unified_cache import UnifiedCache


@dataclass(frozen=True)
class Scale:
    factor: float = field(compare=False)

    def __call__(self, x):
        return self.factor * x


def test_equal_callable_instances_keep_distinct_compilations():
    """Equality can exclude callable state that still changes model output."""
    cache = UnifiedCache()
    x = jnp.array([1.0, 2.0])
    first = Scale(2.0)
    second = Scale(3.0)

    first_compiled = cache.get_or_compile(first, (x,), {})
    np.testing.assert_array_equal(first_compiled(x), [2.0, 4.0])
    second_compiled = cache.get_or_compile(second, (x,), {})
    np.testing.assert_array_equal(second_compiled(x), [3.0, 6.0])


@pytest.mark.serial
def test_callable_identity_cleanup_during_cyclic_gc_does_not_deadlock():
    """GC can execute a weakref callback while the registry lock is held."""
    script = textwrap.dedent(
        """
        import gc
        import faulthandler
        import weakref
        from nlsq.caching._closure_serial import closure_serial

        faulthandler.dump_traceback_later(10, exit=True)
        gc.set_threshold(1, 1, 1)

        class C:
            def __init__(self):
                self.cycle = self

            def __call__(self, x):
                return x

        for i in range(10000):
            c = C()
            closure_serial(c)
            del c

        gc.set_threshold(700, 10, 10)
        func = C()
        serial = closure_serial(func)
        assert serial == closure_serial(func)
        ref = weakref.ref(func)
        del func
        gc.collect()
        assert ref() is None
        assert closure_serial(C()) > serial
        faulthandler.cancel_dump_traceback_later()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "NLSQ_SKIP_GPU_CHECK": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
