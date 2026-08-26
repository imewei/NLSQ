"""Shared per-object identity helper for closure-aware cache-key hashing.

Used by compilation_cache.py, smart_cache.py, core.py, and unified_cache.py
to distinguish two closures that share source text but capture different
values (e.g. two closures built by the same factory) - without this, a
source-text-only hash lets the second closure silently reuse the first's
cached compiled function / result.
"""

import itertools
import threading
import weakref
from collections.abc import Callable

_closure_serial_counter = itertools.count()
_closure_serial_registry: "weakref.WeakKeyDictionary[Callable, int]" = (
    weakref.WeakKeyDictionary()
)
_closure_serial_lock = threading.Lock()


def closure_serial(func: Callable) -> int:
    """Return a stable, per-object, never-reused serial number for ``func``.

    Plain ``id(func)`` is only unique while the object is alive: once a
    closure is garbage-collected, CPython is free to reuse its address for
    an unrelated object created immediately after. A hash keyed on id(func)
    computed for that new object would then collide with a still-cached
    entry meant for the old, GC'd object. The WeakKeyDictionary registry
    assigns each live function object a serial the first time it's seen and
    never reassigns it to a different object, so a genuinely new object
    always misses the lookup and gets a fresh, higher serial even if its
    id() happens to match a dead one's.
    """
    with _closure_serial_lock:
        serial = _closure_serial_registry.get(func)
        if serial is None:
            serial = next(_closure_serial_counter)
            _closure_serial_registry[func] = serial
        return serial
