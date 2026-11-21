"""
Pytest configuration and shared fixtures for NLSQ test suite.

Provides common test functions, data generators, and fixtures for
scientific computing tests.
"""

import importlib
import os
import tempfile

# CRITICAL: Set JAX environment variables BEFORE importing JAX.
# JAX initializes its backend on first import. If these are set in a fixture
# (after import), they have no effect — JAX will use GPU by default, causing:
#   - 4 xdist workers each preallocating 75% of GPU VRAM (48GB from 16GB GPU)
#   - XLA OOM crashes killing the system
# See: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("NLSQ_SKIP_GPU_CHECK", "1")
# NLSQ requires float64 for numerical accuracy. On CPU backend, JAX defaults
# to float32 unless explicitly configured. On GPU, float64 is available by default.
# This must be set before JAX import to ensure consistent behavior.
os.environ.setdefault("JAX_ENABLE_X64", "true")
# Collection imports can initialize NLSQ/JAX before session fixtures run.
# Isolate both workers and concurrent pytest sessions before those imports.
_worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
_compilation_cache = tempfile.TemporaryDirectory(
    prefix=f"nlsq_jax_test_cache_{_worker_id}_"
)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _compilation_cache.name

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _release_heap_to_os() -> None:
    """Release freed heap memory back to the OS via malloc_trim (Linux only).

    glibc malloc doesn't automatically return freed memory to the OS, causing
    RSS to stay high even after Python objects are freed. malloc_trim(0) forces
    the release of free heap pages, actually reducing process RSS.
    """
    if sys.platform == "linux":
        import ctypes

        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except OSError:
            pass  # Not glibc (e.g., musl) — skip silently


# Dynamically skip gui_qt tests when pytest-qt is not installed
collect_ignore_glob = []
if importlib.util.find_spec("pytestqt") is None:
    collect_ignore_glob.append("gui_qt/*")

# ============================================================================
# JAX Configuration for Optimal Test Performance (Phase 1 Optimization)
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def configure_jax_for_tests():
    """
    Configure JAX compilation cache for test performance.

    Note: JAX platform and memory settings are configured via os.environ
    at module level (above), BEFORE JAX is imported. This is critical because
    JAX initializes its backend on first import.

    The cache environment is also set before collection imports above.
    """
    # Apply the setting explicitly too, in case a plugin imported JAX first.
    jax.config.update("jax_compilation_cache_dir", _compilation_cache.name)
    try:
        yield
    finally:
        _compilation_cache.cleanup()


# ============================================================================
# JAX Memory Cleanup Fixture (OOM Prevention)
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_jax_memory():
    """
    Clean up Python references to large JAX arrays after each test.

    This runs gc.collect() to release Python objects holding JAX arrays.
    The XLA runtime then reclaims the device memory.

    Additionally, every CACHE_CLEAR_INTERVAL tests, we call jax.clear_caches()
    to prevent unbounded growth of the JIT compilation cache. Without this,
    each xdist worker grows to 10-16GB RSS across 900+ tests, exhausting
    the 62GB system RAM and crashing the machine.

    After clearing, we call libc malloc_trim(0) on Linux to return freed heap
    memory to the OS (glibc malloc doesn't do this automatically, causing RSS
    to stay high even after Python objects are freed).
    """
    yield

    import gc

    gc.collect()

    # Periodically clear JIT caches to prevent OOM from unbounded cache growth.
    # With 20 xdist workers on a 62GB system, each worker's budget is ~3GB.
    # JIT cache grows ~50-80MB per unique function signature compiled. At interval
    # 100, workers reach 5-9GB RSS before clearing. Interval 30 keeps peak RSS
    # under 3GB/worker, staying within the per-worker memory budget.
    _CACHE_CLEAR_INTERVAL = 30
    cleanup_jax_memory._call_count = getattr(cleanup_jax_memory, "_call_count", 0) + 1
    if cleanup_jax_memory._call_count % _CACHE_CLEAR_INTERVAL == 0:
        jax.clear_caches()
        gc.collect()
        # On Linux, glibc malloc doesn't return freed memory to the OS by default.
        # malloc_trim(0) forces it to release free heap pages back to the kernel,
        # actually reducing RSS instead of just freeing Python objects.
        _release_heap_to_os()


@pytest.fixture(autouse=True)
def _suppress_expected_optimize_warnings():
    """Silence the benign ``OptimizeWarning`` raised on intentional non-convergence.

    Many tests deliberately drive a fit to its evaluation limit (tiny ``max_nfev``,
    heavy outliers, deliberately hard problems) to exercise failure-handling and
    robustness paths. NLSQ correctly emits a single ``OptimizeWarning`` ("Optimal
    parameters not found ...") from ``nlsq/core/minpack.py`` in those cases — it is
    expected, not a defect.

    A ``[tool.pytest.ini_options] filterwarnings`` entry suppresses this reliably
    when a directory runs alone, but under the full ``-n 2`` xdist suite it leaks
    into the warnings summary non-deterministically (pytest's per-location dedup
    even mis-attributes it to an unrelated test). Applying the filter as an autouse
    fixture pins it to every test's own execution on the main thread — the same
    mechanism as a per-test ``filterwarnings`` marker, but one that also covers
    ``unittest.TestCase`` tests (where markers do not apply) and worker threads
    spawned during a test (``warnings.filters`` is process-global). Tests that
    assert this warning use ``pytest.warns`` / ``assertWarns`` / ``recwarn``, which
    install their own ``simplefilter("always")`` and are therefore unaffected.
    """
    import warnings

    from nlsq.result import OptimizeWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        yield


# ============================================================================
# pytest-xdist Auto Worker Count Cap (OOM Prevention)
# ============================================================================


def pytest_xdist_auto_num_workers(config):
    """Cap xdist worker count based on available system memory.

    Each JAX-based xdist worker can grow to 3-5GB RSS due to JIT compilation
    caches. With ``-n auto``, pytest-xdist defaults to one worker per CPU core
    (e.g. 20 on this machine), which exceeds 62GB RAM. This hook caps workers
    to ``available_memory // budget_per_worker``, ensuring the test suite stays
    within memory limits even when ``-n auto`` is passed explicitly.

    The hook is only called when ``-n auto`` (or ``-n logical``) is used.
    Fixed worker counts like ``-n 2`` bypass this hook entirely.
    """
    budget_per_worker_gb = 4  # Conservative: peak RSS ~3GB with cache clearing
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (AttributeError, ValueError, OSError):
        # AttributeError: os.sysconf unavailable on Windows
        # ValueError/OSError: invalid sysconf key on some platforms
        mem_gb = 16  # Safe fallback
    # Reserve 4GB for OS + other processes
    usable_gb = max(mem_gb - 4, budget_per_worker_gb)
    max_workers = max(1, int(usable_gb // budget_per_worker_gb))
    return max_workers


# ============================================================================
# pytest-xdist Serial Test Grouping (OOM Prevention)
# ============================================================================


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """
    Group tests marked with @pytest.mark.serial to run on the same worker.

    This ensures memory-intensive tests don't run in parallel with each other,
    preventing OOM crashes. Tests with the 'serial' marker are assigned to the
    same xdist group, so they execute sequentially on a single worker.

    Note: This hook runs during test collection, before any tests execute.
    """
    for item in items:
        # Check if test has the serial marker
        if item.get_closest_marker("serial"):
            # Assign all serial tests to the same xdist group
            # This ensures they run on the same worker, one at a time
            item.add_marker(pytest.mark.xdist_group("serial_memory_tests"))
        elif not item.get_closest_marker("xdist_group"):
            # loadgroup otherwise schedules every unmarked test independently.
            # Retain loadscope's module/class isolation for ordinary tests.
            scope = item.nodeid.rsplit("::", 1)[0]
            item.add_marker(pytest.mark.xdist_group(scope))


# ============================================================================
# Session-Scoped Compiled Model Fixtures (Phase 2 Optimization)
# ============================================================================


@pytest.fixture(scope="session")
def compiled_models():
    """
    Pre-compiled model functions cached at session level.

    Phase 2 Optimization: Eliminates repeated JAX JIT compilations across tests.
    Each model function is compiled ONCE per test session and reused across all tests.

    Expected Impact: 8-12% reduction (21-40 seconds from 268s baseline)

    Returns
    -------
    dict
        Dictionary of pre-compiled model functions accessible by name.

    Examples
    --------
    >>> def test_linear_fit(compiled_models):
    ...     from nlsq import curve_fit
    ...     popt, pcov = curve_fit(compiled_models['linear'], x, y, p0=[1, 1])
    """

    # Define all common model functions
    def linear(x, a, b):
        return a * x + b

    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    def gaussian(x, amp, mu, sigma):
        return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def sinusoidal(x, a, b, c, d):
        return a * jnp.sin(b * x + c) + d

    def polynomial(x, *coeffs):
        result = jnp.zeros_like(x)
        for i, c in enumerate(coeffs):
            result = result + c * x**i
        return result

    # Create CurveFit instances (triggers JIT compilation on first use)
    models = {
        "linear": linear,
        "quadratic": quadratic,
        "exponential": exponential,
        "gaussian": gaussian,
        "sinusoidal": sinusoidal,
        "polynomial": polynomial,
    }

    # Warm up compilation cache with dummy data
    import contextlib

    x_dummy = jnp.linspace(0, 10, 100)
    for name, func in models.items():
        if name == "polynomial":
            # Polynomial needs specific coefficients
            with contextlib.suppress(Exception):
                func(x_dummy, 1.0, 1.0, 1.0)
        elif name == "sinusoidal":
            func(x_dummy, 1.0, 1.0, 1.0, 1.0)
        elif name in {"gaussian", "exponential", "quadratic"}:
            func(x_dummy, 1.0, 1.0, 1.0)
        else:  # linear
            func(x_dummy, 1.0, 1.0)

    return models


# ============================================================================
# Test Function Fixtures (Module Scope for Better Cache Reuse)
# ============================================================================


@pytest.fixture(scope="module")  # Changed from function to module scope
def linear_func():
    """Linear function: f(x, a, b) = a*x + b"""

    def f(x, a, b):
        return a * x + b

    return f


@pytest.fixture(scope="module")  # Changed from function to module scope
def exponential_func():
    """Exponential decay: f(x, a, b, c) = a*exp(-b*x) + c"""

    def f(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    return f


# ============================================================================
# Data Generator Fixtures (Module Scope for Better Performance)
# ============================================================================


@pytest.fixture(scope="module")  # Changed from function to module scope
def linear_data():
    """Generate clean linear data (cached at module level)"""

    def generate(n_points=100, a=2.0, b=1.0, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 10, n_points)
        y = a * x + b
        return x, y, np.array([a, b])

    return generate


@pytest.fixture(scope="module")  # Changed from function to module scope
def noisy_linear_data():
    """Generate noisy linear data (cached at module level)"""

    def generate(n_points=100, a=2.0, b=1.0, noise_level=0.1, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 10, n_points)
        y = a * x + b + np.random.normal(0, noise_level, n_points)
        return x, y, np.array([a, b])

    return generate


@pytest.fixture(scope="module")  # Changed from function to module scope
def exponential_data():
    """Generate exponential decay data (cached at module level)"""

    def generate(n_points=100, a=2.5, b=1.3, c=0.1, noise_level=0.01, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 5, n_points)
        y = a * np.exp(-b * x) + c
        if noise_level > 0:
            y = y + np.random.normal(0, noise_level, n_points)
        return x, y, np.array([a, b, c])

    return generate


# ============================================================================
# Random Seed Fixtures
# ============================================================================


@pytest.fixture
def random_seed():
    """Provide consistent random seed"""
    return 42


@pytest.fixture(autouse=True)
def reset_random_seed(random_seed):
    """Automatically reset random seed before each test"""
    np.random.seed(random_seed)


@pytest.fixture(autouse=True)
def restore_nlsq_logging_state():
    """Undo logger-level, global-level and NLSQ_DEBUG/NLSQ_VERBOSE changes a test leaves.

    ``enable_debug_mode()`` / ``enable_verbose_mode()`` lower every ``nlsq.*``
    logger and set ``_global_level``, which ``get_logger()`` bakes into each logger
    it first creates (and then caches for the worker's lifetime). Tests restored
    only the root ``nlsq`` logger, so a leaked DEBUG level failed unrelated
    log-capture tests that landed on the same xdist worker.
    """
    import logging

    from nlsq.utils import logging as nlsq_logging

    def _nlsq_loggers():
        return {
            name: lg
            for name, lg in logging.root.manager.loggerDict.items()
            if name.startswith("nlsq") and isinstance(lg, logging.Logger)
        }

    levels = {name: lg.level for name, lg in _nlsq_loggers().items()}
    env = {k: os.environ.get(k) for k in ("NLSQ_DEBUG", "NLSQ_VERBOSE")}
    saved_global = nlsq_logging._global_level
    yield
    nlsq_logging._global_level = saved_global
    # A logger first created inside the test inherits the test's leaked level;
    # give it the level a clean first creation would have produced.
    default = (
        saved_global
        if saved_global is not None
        else logging.INFO
        if env["NLSQ_VERBOSE"] == "1"
        else logging.WARNING
    )
    for name, lg in _nlsq_loggers().items():
        lg.setLevel(levels.get(name, default))
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ============================================================================
# Loss Function Fixtures
# ============================================================================


@pytest.fixture(params=["linear", "huber", "soft_l1", "cauchy", "arctan"])
def loss_type(request):
    """All supported loss function types"""
    return request.param


# ============================================================================
# Bounds Fixtures
# ============================================================================


@pytest.fixture
def no_bounds():
    """No parameter bounds"""
    return (-np.inf, np.inf)


# ============================================================================
# Polling Utilities (Test Flakiness Prevention)
# ============================================================================


import time
from collections.abc import Callable


def wait_for(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    poll_interval: float = 0.01,
    message: str = "Condition not met",
) -> bool:
    """Wait for a condition with exponential backoff.

    This utility replaces hard-coded time.sleep() calls in tests, providing
    more reliable waiting that adapts to the actual time needed rather than
    using fixed delays.

    Parameters
    ----------
    condition : Callable[[], bool]
        A callable that returns True when the condition is met.
    timeout : float, default=5.0
        Maximum time to wait in seconds.
    poll_interval : float, default=0.01
        Initial poll interval in seconds. Grows with exponential backoff.
    message : str, default="Condition not met"
        Error message if timeout is reached.

    Returns
    -------
    bool
        True if condition was met within timeout.

    Raises
    ------
    TimeoutError
        If condition is not met within timeout.

    Examples
    --------
    >>> # Wait for cache entry to expire
    >>> cache.set("key", "value", ttl=0.1)
    >>> wait_for(lambda: cache.get("key") is None, timeout=2.0)

    >>> # Wait for file to be created
    >>> wait_for(lambda: path.exists(), timeout=10.0, message="File not created")

    Notes
    -----
    Uses exponential backoff to reduce CPU usage while maintaining responsiveness.
    The poll interval starts at `poll_interval` and grows by 1.5x each iteration,
    capped at 0.1 seconds.
    """
    start = time.perf_counter()
    interval = poll_interval

    while time.perf_counter() - start < timeout:
        if condition():
            return True
        time.sleep(interval)
        interval = min(interval * 1.5, 0.1)  # Exponential backoff, max 100ms

    raise TimeoutError(f"{message} within {timeout}s")
