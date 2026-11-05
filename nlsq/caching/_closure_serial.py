"""Shared per-object identity helper for closure-aware cache-key hashing.

Used by unified_cache.py to distinguish two closures that share source text
but capture different values (e.g. two closures built by the same factory) -
without this, a source-text-only hash lets the second closure silently reuse
the first's cached compiled function / result.
"""

import itertools
import threading
import weakref
from collections.abc import Callable

_closure_serial_counter = itertools.count()
_closure_serial_registry: dict[int, tuple[weakref.ReferenceType, int]] = {}
# Cyclic GC can run a weakref callback while closure_serial holds this lock.
_closure_serial_lock = threading.RLock()


def closure_serial(func: Callable) -> int:
    """Return a stable, per-object, never-reused serial number for ``func``.

    Plain ``id(func)`` is only unique while the object is alive: once a
    closure is garbage-collected, CPython is free to reuse its address for
    an unrelated object created immediately after. A hash keyed on id(func)
    computed for that new object would then collide with a still-cached
    entry meant for the old, GC'd object. The identity-based weakref registry
    assigns each live function object a serial the first time it's seen and
    never reassigns it to a different object, so a genuinely new object
    always misses the lookup and gets a fresh, higher serial even if its
    id() happens to match a dead one's.
    """
    key = id(func)

    def discard(ref: weakref.ReferenceType) -> None:
        with _closure_serial_lock:
            entry = _closure_serial_registry.get(key)
            if entry is not None and entry[0] is ref:
                del _closure_serial_registry[key]

    with _closure_serial_lock:
        entry = _closure_serial_registry.get(key)
        if entry is not None and entry[0]() is func:
            return entry[1]
        # WeakKeyDictionary uses referent equality, which can merge distinct
        # callable objects whose comparison excludes behavior-affecting state.
        ref = weakref.ref(func, discard)
        serial = next(_closure_serial_counter)
        _closure_serial_registry[key] = (ref, serial)
        return serial
