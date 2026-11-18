"""Central configuration management for NLSQ package."""

from __future__ import annotations

import contextvars
import copy
import dataclasses
import json
import logging
import math
import numbers
import os
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

# Module logger: calling logging.info()/debug() on the root logger would run
# logging.basicConfig() and attach a stderr handler to the application's root.
_logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    """True when env var ``name`` is 1/true/yes/on (case-insensitive), as NLSQ_DEBUG."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_positive_count(value) -> bool:
    """True for a finite integral non-bool real ``>= 1`` (``3``, ``np.int64(3)``, ``3.0``).

    Mirrors ``nlsq.core._option_checks.is_integral_count``, which this module
    cannot import (nlsq.core imports nlsq.config at import time).
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return False
    # compare the value itself, not a float rounding (huge ints, Fraction)
    if not isinstance(value, numbers.Integral) and not (
        math.isfinite(value) and value == int(value)  # type: ignore[call-overload]
    ):
        return False
    return value >= 1  # type: ignore[operator]


@dataclass(slots=True)
class MemoryConfig:
    """Configuration for memory management and GPU settings.

    Attributes
    ----------
    memory_limit_gb : float | None
        Maximum memory limit in GB (0.1-1024). ``None`` (default) means
        unset: strategy selection uses the detected system memory.
    gpu_memory_fraction : float | None
        Fraction of GPU memory to use (0.0-1.0, None for automatic)
    chunk_size_mb : int | None
        Default chunk size in MB for data processing. Stored and validated only:
        nothing reads it yet (as ``NLSQ_CHUNK_SIZE_MB``, it warns).
    out_of_memory_strategy : str
        Strategy when out of memory: 'fallback', 'reduce', 'error'. Stored and
        validated only: nothing reads it yet (as ``NLSQ_OOM_STRATEGY``, it warns).
    safety_factor : float | None
        Fraction of the memory limit that strategy selection may use
        (0.1-1.0). ``None`` (default) leaves the workflow default of 0.75.
    auto_chunk_threshold_gb : float
        Automatically enable chunking above this memory threshold. Stored only:
        nothing reads it yet.
    progress_reporting : bool
        Enable progress reporting for large operations. Stored only: nothing
        reads it yet (``NLSQ_DISABLE_PROGRESS_REPORTING`` warns).
    min_chunk_size : int
        Minimum chunk size in data points
    max_chunk_size : int
        Maximum chunk size in data points
    """

    memory_limit_gb: float | None = None
    gpu_memory_fraction: float | None = None
    chunk_size_mb: int | None = None
    out_of_memory_strategy: str = "fallback"
    safety_factor: float | None = None
    auto_chunk_threshold_gb: float = 4.0
    progress_reporting: bool = True
    min_chunk_size: int = 1000
    max_chunk_size: int = 1_000_000

    def __post_init__(self):
        """Validate configuration values."""
        for name in ("memory_limit_gb", "gpu_memory_fraction", "safety_factor"):
            if isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a number, not bool")
        if self.chunk_size_mb is not None:
            if not _is_positive_count(self.chunk_size_mb):
                raise ValueError(
                    f"chunk_size_mb must be None or a positive integer, got {self.chunk_size_mb!r}",
                )
            self.chunk_size_mb = int(self.chunk_size_mb)
        threshold = self.auto_chunk_threshold_gb
        # +inf is the documented "chunking disabled" value (configure_for_large_datasets)
        if isinstance(threshold, (bool, np.bool_)) or not threshold > 0:
            raise ValueError(
                f"auto_chunk_threshold_gb must be positive, got {threshold!r}",
            )
        for name in ("min_chunk_size", "max_chunk_size"):
            value = getattr(self, name)
            if not _is_positive_count(value):
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            setattr(self, name, int(value))

        if self.memory_limit_gb is not None and not 0.1 <= self.memory_limit_gb <= 1024:
            raise ValueError(
                f"memory_limit_gb must be between 0.1 and 1024, got {self.memory_limit_gb}",
            )

        if self.gpu_memory_fraction is not None:
            if not 0.0 < self.gpu_memory_fraction <= 1.0:
                raise ValueError(
                    f"gpu_memory_fraction must be between 0.0 and 1.0, got {self.gpu_memory_fraction}",
                )

        if self.safety_factor is not None and not 0.1 <= self.safety_factor <= 1.0:
            raise ValueError(
                f"safety_factor must be between 0.1 and 1.0, got {self.safety_factor}",
            )

        if self.out_of_memory_strategy not in ["fallback", "reduce", "error"]:
            raise ValueError(
                f"out_of_memory_strategy must be 'fallback', 'reduce', or 'error', got {self.out_of_memory_strategy}",
            )

        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) cannot be larger than max_chunk_size ({self.max_chunk_size})",
            )


@dataclass(slots=True)
class LargeDatasetConfig:
    """Configuration for large dataset processing.

    Attributes
    ----------
    enable_automatic_solver_selection : bool
        Automatically select optimal solver based on dataset size
    solver_selection_thresholds : Dict[str, int]
        Thresholds for automatic solver selection. Both fields are stored and
        validated only: nothing reads them yet (``NLSQ_DISABLE_AUTO_SOLVER_SELECTION``
        warns).

    Notes
    -----
    As of v0.2.0, all subsampling parameters have been removed. Use streaming
    optimization instead for unlimited datasets. See MIGRATION_V0.2.0.md for
    migration instructions.
    """

    enable_automatic_solver_selection: bool = True
    solver_selection_thresholds: dict[str, int] = field(
        default_factory=lambda: {
            "direct": 100_000,  # Use direct solver below this size
            "iterative": 10_000_000,  # Use iterative solver below this size
            "chunked": 100_000_000,  # Use chunked processing above this size
        },
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.solver_selection_thresholds:
            return  # Empty dict means "apply no threshold overrides" — valid.
        required_keys = {"direct", "iterative", "chunked"}
        missing = required_keys - set(self.solver_selection_thresholds)
        if missing:
            raise ValueError(
                f"solver_selection_thresholds missing required keys: {missing}",
            )
        thresholds = self.solver_selection_thresholds
        if not (thresholds["direct"] < thresholds["iterative"] < thresholds["chunked"]):
            raise ValueError(
                "solver_selection_thresholds must be monotonically increasing: "
                f"direct={thresholds['direct']} < iterative={thresholds['iterative']} "
                f"< chunked={thresholds['chunked']}",
            )
        for key, val in thresholds.items():
            if val <= 0:
                raise ValueError(
                    f"solver_selection_thresholds['{key}'] must be positive, "
                    f"got {val!r}",
                )


# MemoryConfig fields that are validated and stored but that nothing consumes yet.
_UNCONSUMED_ENV = {
    "chunk_size_mb": "NLSQ_CHUNK_SIZE_MB",
    "out_of_memory_strategy": "NLSQ_OOM_STRATEGY",
    "progress_reporting": "NLSQ_DISABLE_PROGRESS_REPORTING",
}


def _warn_unconsumed_env(overrides: dict) -> None:
    for key, env_name in _UNCONSUMED_ENV.items():
        if key in overrides:
            warnings.warn(
                f"{env_name} is set but no NLSQ code reads MemoryConfig.{key} "
                "yet; it has no effect.",
                UserWarning,
                stacklevel=4,
            )


# Context-local overrides installed by memory_context / large_dataset_context.
_memory_override: contextvars.ContextVar[MemoryConfig | None] = contextvars.ContextVar(
    "nlsq_memory_override", default=None
)
_large_dataset_override: contextvars.ContextVar[LargeDatasetConfig | None] = (
    contextvars.ContextVar("nlsq_large_dataset_override", default=None)
)


class JAXConfig:
    """Singleton configuration manager for JAX and memory settings.

    This class ensures that JAX configuration is set once and consistently
    across all NLSQ modules, avoiding duplicate configuration calls. It also
    manages memory settings and large dataset configuration.
    """

    _instance: JAXConfig | None = None
    _lock: threading.RLock = threading.RLock()
    # Dedicated lock for precision_context's enter/yield/exit span, kept
    # separate from `_lock` (singleton construction, memory config, etc.)
    # so a long-running precision_context body on one thread doesn't block
    # unrelated JAXConfig operations on another thread.
    _precision_lock: threading.RLock = threading.RLock()
    _x64_enabled: bool = False
    _initialized: bool = False
    _memory_config: MemoryConfig | None = None
    _large_dataset_config: LargeDatasetConfig | None = None
    _gpu_memory_configured: bool = False

    def __new__(cls) -> JAXConfig:
        """Ensure singleton pattern (thread-safe)."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize JAX and memory configuration if not already done."""
        with self._lock:
            if not self._initialized:
                self._initialize_jax()
                self._initialize_memory_config()
                self._initialize_large_dataset_config()
                self._initialized = True

    def _initialize_jax(self):
        """Initialize JAX with default NLSQ settings."""
        # Import here to avoid circular imports
        from jax import config

        # Force CPU backend if requested (useful for testing)
        if _env_flag("NLSQ_FORCE_CPU") or os.getenv("JAX_PLATFORM_NAME") == "cpu":
            config.update("jax_platform_name", "cpu")

        # Enable 64-bit precision by default for NLSQ
        if not self._x64_enabled and not _env_flag("NLSQ_DISABLE_X64"):
            config.update("jax_enable_x64", True)
            self._x64_enabled = True

        # Configure persistent compilation cache (eliminates 2-10s cold start)
        self._configure_persistent_cache(config)

        # Configure GPU memory if specified
        self._configure_gpu_memory()

    def _configure_persistent_cache(self, config):
        """Configure JAX persistent compilation cache.

        This enables caching of compiled functions across Python sessions,
        eliminating cold-start overhead of 2-10 seconds.
        """
        # Skip if explicitly disabled
        if _env_flag("NLSQ_DISABLE_PERSISTENT_CACHE"):
            return

        # Parse before touching jax.config so a bad value cannot leave the
        # cache half-configured.
        min_secs: float | None = None
        raw_min_secs = os.getenv("NLSQ_CACHE_MIN_COMPILE_TIME_SECS")
        if raw_min_secs is not None:
            try:
                min_secs = float(raw_min_secs)
                if not math.isfinite(min_secs) or min_secs < 0:
                    raise ValueError(raw_min_secs)
            except ValueError:
                min_secs = None
                _logger.warning(
                    f"Invalid NLSQ_CACHE_MIN_COMPILE_TIME_SECS={raw_min_secs!r}; ignored"
                )

        # Keep a cache dir the user already configured (env var or jax.config).
        # (JAX copies JAX_COMPILATION_CACHE_DIR into jax.config at import.)
        # An explicit NLSQ_CACHE_MIN_COMPILE_TIME_SECS still applies.
        if config.jax_compilation_cache_dir:
            if min_secs is not None:
                config.update("jax_persistent_cache_min_compile_time_secs", min_secs)
            return

        # Honor JAX_COMPILATION_CACHE_DIR set after jax was imported (jax.config
        # did not pick it up), so NLSQ doesn't redirect it to its own default.
        cache_dir = os.getenv("JAX_COMPILATION_CACHE_DIR")
        if cache_dir:
            nlsq_cache_dir = os.getenv("NLSQ_JAX_CACHE_DIR")
            if nlsq_cache_dir:
                _logger.info(
                    f"Using JAX_COMPILATION_CACHE_DIR={cache_dir} "
                    f"(NLSQ_JAX_CACHE_DIR={nlsq_cache_dir} ignored)",
                )
        else:
            # Empty NLSQ_JAX_CACHE_DIR means unset (os.makedirs("") would fail).
            cache_dir = os.getenv("NLSQ_JAX_CACHE_DIR") or os.path.expanduser(
                "~/.cache/nlsq/jax_cache"
            )

        try:
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)

            # Enable persistent compilation cache
            config.update("jax_compilation_cache_dir", cache_dir)

            # Only cache compilations that take at least 1 second
            config.update(
                "jax_persistent_cache_min_compile_time_secs",
                1.0 if min_secs is None else min_secs,
            )

            _logger.debug(f"JAX persistent cache enabled at {cache_dir}")
        except Exception as e:
            # Non-fatal: log warning and continue without persistent cache
            _logger.warning(
                f"Failed to enable JAX persistent compilation cache: {e}. "
                "Cold-start may be slower.",
            )

    def _configure_gpu_memory(self):
        """Configure GPU memory settings via XLA environment variables.

        Sets XLA_PYTHON_CLIENT_PREALLOCATE and XLA_PYTHON_CLIENT_MEM_FRACTION
        to prevent XLA from consuming all GPU memory. These must be set before
        JAX initializes the GPU backend, so we set them as early as possible.

        Without these defaults, XLA preallocates 75% of GPU memory at startup
        and never releases it, which combined with unbounded compilation cache
        growth can cause OOM for long-running sessions.
        """
        if self._gpu_memory_configured:
            return

        # Default: disable preallocation to allow memory to grow/shrink
        # This prevents XLA from grabbing 75% of GPU memory at startup.
        # Users can override via XLA_PYTHON_CLIENT_PREALLOCATE=true.
        if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            _logger.debug("Set XLA_PYTHON_CLIENT_PREALLOCATE=false (default for NLSQ)")

        # Check for NLSQ-specific GPU memory fraction setting
        gpu_memory_fraction = os.getenv("NLSQ_GPU_MEMORY_FRACTION")
        if gpu_memory_fraction:
            try:
                fraction = float(gpu_memory_fraction)
                if 0.0 < fraction <= 1.0:
                    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fraction)
                    _logger.info(f"Set XLA_PYTHON_CLIENT_MEM_FRACTION={fraction}")
                else:
                    warnings.warn(
                        f"Invalid NLSQ_GPU_MEMORY_FRACTION: {gpu_memory_fraction}. Must be between 0.0 and 1.0.",
                        stacklevel=2,
                    )
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_GPU_MEMORY_FRACTION: {gpu_memory_fraction}. Must be a number.",
                    stacklevel=2,
                )

        # Configure memory preallocation if explicitly requested. JAX has no
        # documented config key for this — only the XLA_PYTHON_CLIENT_PREALLOCATE
        # env var actually works — so set it directly, overriding any prior
        # XLA_PYTHON_CLIENT_PREALLOCATE=true.
        if _env_flag("NLSQ_DISABLE_GPU_PREALLOCATION"):
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            _logger.info("Disabled GPU memory preallocation via XLA env var")

        self._gpu_memory_configured = True

    def _initialize_memory_config(self):
        """Initialize memory configuration from environment variables."""
        if self._memory_config is not None:
            return

        # Collect env overrides, then construct MemoryConfig in one shot
        # (safe if MemoryConfig ever becomes frozen=True)
        overrides: dict = {}

        limit = os.getenv("NLSQ_MEMORY_LIMIT_GB")
        if limit:
            try:
                overrides["memory_limit_gb"] = float(limit)
            except ValueError:
                warnings.warn(f"Invalid NLSQ_MEMORY_LIMIT_GB: {limit}", stacklevel=2)

        fraction = os.getenv("NLSQ_GPU_MEMORY_FRACTION")
        if fraction:
            try:
                overrides["gpu_memory_fraction"] = float(fraction)
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_GPU_MEMORY_FRACTION: {fraction}",
                    stacklevel=2,
                )

        chunk_size = os.getenv("NLSQ_CHUNK_SIZE_MB")
        if chunk_size:
            try:
                overrides["chunk_size_mb"] = int(chunk_size)
            except ValueError:
                warnings.warn(f"Invalid NLSQ_CHUNK_SIZE_MB: {chunk_size}", stacklevel=2)

        strategy = os.getenv("NLSQ_OOM_STRATEGY")
        if strategy:
            if strategy in ["fallback", "reduce", "error"]:
                overrides["out_of_memory_strategy"] = strategy
            else:
                warnings.warn(
                    f"Invalid NLSQ_OOM_STRATEGY: {strategy}. Must be 'fallback', 'reduce', or 'error'.",
                    stacklevel=2,
                )

        safety_factor = os.getenv("NLSQ_SAFETY_FACTOR")
        if safety_factor:
            try:
                overrides["safety_factor"] = float(safety_factor)
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_SAFETY_FACTOR: {safety_factor}",
                    stacklevel=2,
                )

        if _env_flag("NLSQ_DISABLE_PROGRESS_REPORTING"):
            overrides["progress_reporting"] = False

        # An out-of-range value must not make every later fit raise from this
        # lazy initializer: warn and fall back to the default for that field.
        env_names = {
            "memory_limit_gb": "NLSQ_MEMORY_LIMIT_GB",
            "gpu_memory_fraction": "NLSQ_GPU_MEMORY_FRACTION",
            "chunk_size_mb": "NLSQ_CHUNK_SIZE_MB",
            "out_of_memory_strategy": "NLSQ_OOM_STRATEGY",
            "safety_factor": "NLSQ_SAFETY_FACTOR",
        }
        for key in list(overrides):
            try:
                MemoryConfig(**{key: overrides[key]})
            except ValueError as exc:
                value = overrides.pop(key)
                warnings.warn(
                    f"Ignoring {env_names.get(key, key)}={value}: {exc}",
                    stacklevel=2,
                )

        _warn_unconsumed_env(overrides)
        self._memory_config = MemoryConfig(**overrides)

    def _initialize_large_dataset_config(self):
        """Initialize large dataset configuration from environment variables."""
        if self._large_dataset_config is not None:
            return

        # Load defaults
        large_dataset_config = LargeDatasetConfig()

        # Override from environment variables
        if _env_flag("NLSQ_DISABLE_AUTO_SOLVER_SELECTION"):
            large_dataset_config.enable_automatic_solver_selection = False
            warnings.warn(
                "NLSQ_DISABLE_AUTO_SOLVER_SELECTION is set but no NLSQ code reads "
                "LargeDatasetConfig.enable_automatic_solver_selection yet; it has "
                "no effect.",
                UserWarning,
                stacklevel=2,
            )

        self._large_dataset_config = large_dataset_config

    @classmethod
    def enable_x64(cls, enable: bool = True):
        """Enable or disable 64-bit precision.

        Parameters
        ----------
        enable : bool, optional
            If True, enable 64-bit precision. If False, use 32-bit.
            Default is True.
        """
        from jax import config

        instance = cls()

        with cls._lock:
            # Compare against the live JAX flag: anything else (user code,
            # another library, JAX_ENABLE_X64) may have changed it since we
            # last cached it, which would turn this call into a silent no-op.
            if bool(config.jax_enable_x64) != enable:
                config.update("jax_enable_x64", enable)
            instance._x64_enabled = enable

    @classmethod
    def is_x64_enabled(cls) -> bool:
        """Check if 64-bit precision is enabled.

        Returns
        -------
        bool
            True if 64-bit precision is enabled, False otherwise.
        """
        from jax import config

        instance = cls()
        instance._x64_enabled = bool(config.jax_enable_x64)
        return instance._x64_enabled

    @classmethod
    @contextmanager
    def precision_context(cls, use_x64: bool):
        """Context manager for temporarily changing precision.

        Parameters
        ----------
        use_x64 : bool
            If True, use 64-bit precision within context.
            If False, use 32-bit precision.

        Examples
        --------
        >>> with JAXConfig.precision_context(use_x64=False):
        ...     # Code here runs with 32-bit precision
        ...     result = some_computation()
        >>> # Back to previous precision setting
        """
        instance = cls()

        # Hold _precision_lock for the whole enter/yield/exit span, not just
        # each enable_x64() call individually. jax_enable_x64 is one
        # process-wide flag; without this, two overlapping precision_context
        # calls on different threads can each snapshot the other's transient
        # state and restore the wrong value on exit. This uses a dedicated
        # lock (not the general `_lock`) so a long-running body here doesn't
        # block unrelated JAXConfig operations (singleton construction,
        # memory config) on other threads.
        with cls._precision_lock:
            original_state = cls.is_x64_enabled()
            try:
                cls.enable_x64(use_x64)
                yield
            finally:
                cls.enable_x64(original_state)

    # Memory configuration methods
    #
    # memory_context / large_dataset_context install a context-local override
    # (contextvars) that the getters consult before the process-wide value, so
    # contexts never block each other or a setter on another thread. The override
    # is seen by the creating thread/asyncio task and by anything that runs in a
    # copy of its context; a plain ``threading.Thread`` (or a pool that does not
    # use ``contextvars.copy_context().run``) starts with an empty context and
    # reads the process-wide value. A setter called inside a context updates that
    # context's value (discarded on exit); outside any context it updates the
    # process-wide value under a short-lived lock.
    @classmethod
    def get_memory_config(cls) -> MemoryConfig:
        """Get the current memory configuration.

        Returns
        -------
        MemoryConfig
            Current memory configuration (the innermost ``memory_context``
            override of the calling context, else the process-wide value)
        """
        override = _memory_override.get()
        if override is not None:
            return copy.copy(override)
        instance = cls()
        if instance._memory_config is None:
            instance._initialize_memory_config()
        # Explicit validation after initialization (not assert - can be optimized away)
        if instance._memory_config is None:
            raise RuntimeError("Memory config initialization failed")
        return copy.copy(instance._memory_config)

    @classmethod
    def set_memory_config(cls, config: MemoryConfig):
        """Set the memory configuration.

        Inside a ``memory_context`` this changes only that context's value;
        otherwise the process-wide value.

        Parameters
        ----------
        config : MemoryConfig
            New memory configuration
        """
        if _memory_override.get() is not None:
            _memory_override.set(copy.deepcopy(config))
        else:
            instance = cls()
            with cls._lock:
                instance._memory_config = copy.deepcopy(config)

        # Apply GPU memory settings immediately if possible
        try:
            if config.gpu_memory_fraction is not None:
                # Note: JAX memory fraction handling varies by version and backend
                # This is stored in our config for use by downstream components
                _logger.info(
                    f"Updated GPU memory fraction to {config.gpu_memory_fraction} (stored for downstream use)",
                )
        except ImportError:
            pass  # JAX not available

    @classmethod
    def get_large_dataset_config(cls) -> LargeDatasetConfig:
        """Get the current large dataset configuration.

        Returns
        -------
        LargeDatasetConfig
            Current large dataset configuration (the innermost
            ``large_dataset_context`` override of the calling context, else the
            process-wide value)
        """
        override = _large_dataset_override.get()
        if override is not None:
            return copy.deepcopy(override)
        instance = cls()
        if instance._large_dataset_config is None:
            instance._initialize_large_dataset_config()
        # Explicit validation after initialization (not assert - can be optimized away)
        if instance._large_dataset_config is None:
            raise RuntimeError("Large dataset config initialization failed")
        return copy.deepcopy(instance._large_dataset_config)

    @classmethod
    def set_large_dataset_config(cls, config: LargeDatasetConfig):
        """Set the large dataset configuration.

        Inside a ``large_dataset_context`` this changes only that context's
        value; otherwise the process-wide value.

        Parameters
        ----------
        config : LargeDatasetConfig
            New large dataset configuration
        """
        if _large_dataset_override.get() is not None:
            _large_dataset_override.set(copy.deepcopy(config))
        else:
            instance = cls()
            with cls._lock:
                instance._large_dataset_config = copy.deepcopy(config)

    @classmethod
    @contextmanager
    def memory_context(cls, memory_config: MemoryConfig):
        """Context manager for temporarily changing memory configuration.

        The change is local to the calling thread/asyncio task (see the note
        above): other threads keep the process-wide value, and a helper
        ``threading.Thread`` started inside the block does not see it. Contexts
        nest, never block, and are undone on exit, including on exceptions.

        Parameters
        ----------
        memory_config : MemoryConfig
            Temporary memory configuration

        Examples
        --------
        >>> from nlsq.config import JAXConfig, MemoryConfig
        >>> temp_config = MemoryConfig(memory_limit_gb=16.0)
        >>> with JAXConfig.memory_context(temp_config):
        ...     # Code here runs with increased memory limit
        ...     result = fit_large_dataset(func, x, y)
        >>> # Back to previous memory settings
        """
        cls()  # make sure the process-wide config is initialised
        token = _memory_override.set(copy.deepcopy(memory_config))
        try:
            yield
        finally:
            _memory_override.reset(token)

    @classmethod
    @contextmanager
    def large_dataset_context(cls, large_dataset_config: LargeDatasetConfig):
        """Context manager for temporarily changing large dataset configuration.

        Context-local like :meth:`memory_context`.

        Parameters
        ----------
        large_dataset_config : LargeDatasetConfig
            Temporary large dataset configuration
        """
        cls()
        token = _large_dataset_override.set(copy.deepcopy(large_dataset_config))
        try:
            yield
        finally:
            _large_dataset_override.reset(token)


# Initialize configuration on module import
_config = JAXConfig()


# Convenience functions
def enable_x64(enable: bool = True):
    """Enable or disable 64-bit precision.

    Parameters
    ----------
    enable : bool, optional
        If True, enable 64-bit precision. If False, use 32-bit.
        Default is True.
    """
    JAXConfig.enable_x64(enable)


def is_x64_enabled() -> bool:
    """Check if 64-bit precision is enabled.

    Returns
    -------
    bool
        True if 64-bit precision is enabled, False otherwise.
    """
    return JAXConfig.is_x64_enabled()


def precision_context(use_x64: bool):
    """Context manager for temporarily changing precision.

    Parameters
    ----------
    use_x64 : bool
        If True, use 64-bit precision within context.
        If False, use 32-bit precision.

    Examples
    --------
    >>> from nlsq.config import precision_context
    >>> with precision_context(use_x64=False):
    ...     # Code here runs with 32-bit precision
    ...     result = some_computation()
    >>> # Back to previous precision setting
    """
    return JAXConfig.precision_context(use_x64)


# Memory management convenience functions
def get_memory_config() -> MemoryConfig:
    """Get the current memory configuration.

    Returns
    -------
    MemoryConfig
        Current memory configuration
    """
    return JAXConfig.get_memory_config()


def set_memory_limits(
    memory_limit_gb: float | None,
    gpu_memory_fraction: float | None = None,
    safety_factor: float | None = None,
):
    """Set memory limits for NLSQ operations.

    Parameters
    ----------
    memory_limit_gb : float | None
        Maximum memory to use in GB (0.1-1024). ``None`` clears the configured
        limit, so strategy selection uses the detected system memory.
    gpu_memory_fraction : float, optional
        Fraction of GPU memory to use (0.0-1.0)
    safety_factor : float, optional
        Fraction of the limit that strategy selection may use. ``None``
        (default) keeps the current setting.

    Notes
    -----
    Inside a ``memory_context`` the change is context-local (thread / asyncio-task
    local, via ``contextvars``) and is discarded when the context exits. Plain
    threads started inside the context do not inherit it; use
    ``contextvars.copy_context().run`` to propagate it.

    Examples
    --------
    >>> from nlsq.config import set_memory_limits
    >>> # Set 16GB memory limit with 80% GPU memory usage
    >>> set_memory_limits(16.0, gpu_memory_fraction=0.8)
    """
    current_config = get_memory_config()
    new_config = MemoryConfig(
        memory_limit_gb=memory_limit_gb,
        gpu_memory_fraction=(
            gpu_memory_fraction
            if gpu_memory_fraction is not None
            else current_config.gpu_memory_fraction
        ),
        safety_factor=(
            safety_factor if safety_factor is not None else current_config.safety_factor
        ),
        chunk_size_mb=current_config.chunk_size_mb,
        out_of_memory_strategy=current_config.out_of_memory_strategy,
        auto_chunk_threshold_gb=current_config.auto_chunk_threshold_gb,
        progress_reporting=current_config.progress_reporting,
        min_chunk_size=current_config.min_chunk_size,
        max_chunk_size=current_config.max_chunk_size,
    )
    JAXConfig.set_memory_config(new_config)


def configure_for_large_datasets(
    memory_limit_gb: float | None = None,
    enable_chunking: bool = True,
    progress_reporting: bool = True,
):
    """Configure NLSQ for optimal large dataset performance.

    This function sets up memory management, chunking, streaming, and other
    settings for handling large datasets efficiently.

    Parameters
    ----------
    memory_limit_gb : float, optional
        Maximum memory to use in GB. ``None`` (default) leaves the limit
        unset, so strategy selection uses the detected system memory.
    enable_chunking : bool, optional
        Enable automatic data chunking (default: True). Recorded only: nothing
        reads ``auto_chunk_threshold_gb`` yet, so False has no effect (warns).
    progress_reporting : bool, optional
        Enable progress reporting for long operations (default: True). Recorded
        only: nothing reads it yet, so False has no effect (warns).

    Notes
    -----
    All large datasets use streaming optimization for 100% data utilization.
    Memory fields this call does not take (``safety_factor``,
    ``gpu_memory_fraction``, ``chunk_size_mb``, ...) keep their current values.
    Inside a ``memory_context`` the change is context-local (thread / asyncio-task
    local) and is discarded when the context exits; plain threads started there do
    not inherit it (use ``contextvars.copy_context().run``).

    Examples
    --------
    >>> from nlsq.config import configure_for_large_datasets
    >>> # Configure for large datasets with 16GB memory limit
    >>> configure_for_large_datasets(
    ...     memory_limit_gb=16.0,
    ...     progress_reporting=True
    ... )
    """
    if not enable_chunking or not progress_reporting:
        warnings.warn(
            "enable_chunking=False and progress_reporting=False are recorded in the "
            "configuration but no NLSQ code reads them yet, so they have no effect.",
            UserWarning,
            stacklevel=2,
        )
    # Keep the fields this call does not take (safety_factor, gpu_memory_fraction, ...)
    memory_config = dataclasses.replace(
        get_memory_config(),
        memory_limit_gb=memory_limit_gb,
        auto_chunk_threshold_gb=(
            memory_limit_gb if memory_limit_gb is not None else 8.0
        )
        * 0.5
        if enable_chunking
        else float("inf"),
        progress_reporting=progress_reporting,
    )
    JAXConfig.set_memory_config(memory_config)

    # Configure large dataset settings
    large_dataset_config = LargeDatasetConfig(
        enable_automatic_solver_selection=True,
    )
    JAXConfig.set_large_dataset_config(large_dataset_config)

    _logger.info("Configured NLSQ for large datasets:")
    _logger.info(
        f"  Memory limit: {memory_limit_gb if memory_limit_gb is not None else 'auto'} GB",
    )
    _logger.info("  Streaming: enabled (always available)")
    _logger.info(f"  Chunking: {'enabled' if enable_chunking else 'disabled'}")
    _logger.info(
        f"  Progress reporting: {'enabled' if progress_reporting else 'disabled'}",
    )


def get_large_dataset_config() -> LargeDatasetConfig:
    """Get the current large dataset configuration.

    Returns
    -------
    LargeDatasetConfig
        Current large dataset configuration
    """
    return JAXConfig.get_large_dataset_config()


def memory_context(memory_config: MemoryConfig):
    """Context manager for temporarily changing memory configuration.

    Parameters
    ----------
    memory_config : MemoryConfig
        Temporary memory configuration

    Notes
    -----
    The change is context-local (thread / asyncio-task local, via ``contextvars``):
    other threads are unaffected and never block. A plain ``threading.Thread``
    started inside the block does not inherit it and reads the process-wide
    value; run pool work through ``contextvars.copy_context().run`` to propagate
    it.

    Examples
    --------
    >>> from nlsq.config import memory_context, MemoryConfig
    >>> temp_config = MemoryConfig(memory_limit_gb=16.0)
    >>> with memory_context(temp_config):
    ...     # Code here runs with increased memory limit
    ...     result = fit_large_dataset(func, x, y)
    >>> # Back to previous memory settings
    """
    return JAXConfig.memory_context(memory_config)


def large_dataset_context(large_dataset_config: LargeDatasetConfig):
    """Context manager for temporarily changing large dataset configuration.

    Parameters
    ----------
    large_dataset_config : LargeDatasetConfig
        Temporary large dataset configuration

    Notes
    -----
    The change is context-local (thread / asyncio-task local, via ``contextvars``):
    other threads are unaffected and never block. A plain ``threading.Thread``
    started inside the block does not inherit it and reads the process-wide
    value; run pool work through ``contextvars.copy_context().run`` to propagate
    it.

    Examples
    --------
    >>> from nlsq.config import large_dataset_context, LargeDatasetConfig
    >>> temp_config = LargeDatasetConfig(enable_automatic_solver_selection=True)
    >>> with large_dataset_context(temp_config):
    ...     # Code here uses automatic solver selection
    ...     result = fit_large_dataset(func, x, y)
    """
    return JAXConfig.large_dataset_context(large_dataset_config)


# Jacobian mode configuration functions
def get_jacobian_mode() -> tuple[str, str]:
    """Get Jacobian mode from configuration sources.

    Configuration precedence (highest to lowest):
    1. Environment variable (NLSQ_JACOBIAN_MODE)
    2. Config file (~/.nlsq/config.json)
    3. Auto-default

    Returns
    -------
    mode : str
        Jacobian mode ('auto', 'fwd', or 'rev')
    source : str
        Source of the configuration ('environment variable', 'config file', 'auto-default')

    Examples
    --------
    >>> from nlsq.config import get_jacobian_mode
    >>> mode, source = get_jacobian_mode()
    >>> print(f"Using {mode} mode from {source}")
    Using auto mode from auto-default

    Notes
    -----
    Valid jacobian_mode values:
    - 'auto': Automatically select based on problem dimensions
    - 'fwd': Force forward-mode automatic differentiation (jacfwd)
    - 'rev': Force reverse-mode automatic differentiation (jacrev)
    """
    # Check environment variable (highest priority)
    env_mode = os.environ.get("NLSQ_JACOBIAN_MODE")
    if env_mode:
        if env_mode in ("auto", "fwd", "rev"):
            return env_mode, "environment variable"
        warnings.warn(
            f"Invalid NLSQ_JACOBIAN_MODE: {env_mode}. Must be 'auto', 'fwd', or 'rev'. Using auto-default.",
            stacklevel=2,
        )

    # Check config file
    config_path = os.path.expanduser("~/.nlsq/config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
                if not isinstance(config, dict):
                    raise ValueError("Expected a JSON object at the config file root")
                if "jacobian_mode" in config:
                    mode = config["jacobian_mode"]
                    if mode in ("auto", "fwd", "rev"):
                        return mode, "config file"
                    warnings.warn(
                        f"Invalid jacobian_mode in config file: {mode}. Must be 'auto', 'fwd', or 'rev'. Using auto-default.",
                        stacklevel=2,
                    )
        except (OSError, ValueError) as e:
            warnings.warn(
                f"Failed to read Jacobian mode from config file: {e}. Using auto-default.",
                stacklevel=2,
            )

    # Default to auto
    return "auto", "auto-default"


def set_jacobian_mode(mode: str):
    """Set Jacobian mode via environment variable.

    This sets the NLSQ_JACOBIAN_MODE environment variable for the current process.
    To persist the setting, use a config file at ~/.nlsq/config.json.

    Parameters
    ----------
    mode : str
        Jacobian mode ('auto', 'fwd', or 'rev')

    Raises
    ------
    ValueError
        If mode is not one of 'auto', 'fwd', 'rev'

    Examples
    --------
    >>> from nlsq.config import set_jacobian_mode
    >>> set_jacobian_mode('rev')  # Force reverse-mode AD for all fits
    """
    if mode not in ("auto", "fwd", "rev"):
        raise ValueError(
            f"Invalid jacobian_mode: {mode}. Must be 'auto', 'fwd', or 'rev'.",
        )

    os.environ["NLSQ_JACOBIAN_MODE"] = mode
    _logger.info(f"Set Jacobian mode to '{mode}' via environment variable")
