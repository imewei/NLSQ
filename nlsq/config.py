"""Central configuration management for NLSQ package."""

import json
import logging
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nlsq.precision.mixed_precision import MixedPrecisionConfig


@dataclass(slots=True)
class MemoryConfig:
    """Configuration for memory management and GPU settings.

    Attributes
    ----------
    memory_limit_gb : float
        Maximum memory limit in GB (default: 8.0)
    gpu_memory_fraction : Optional[float]
        Fraction of GPU memory to use (0.0-1.0, None for automatic)
    enable_mixed_precision_fallback : bool
        Enable automatic mixed precision fallback for large datasets
    chunk_size_mb : Optional[int]
        Default chunk size in MB for data processing
    out_of_memory_strategy : str
        Strategy when out of memory: 'fallback', 'reduce', 'error'
    safety_factor : float
        Safety factor for memory calculations (0.0-1.0)
    auto_chunk_threshold_gb : float
        Automatically enable chunking above this memory threshold
    progress_reporting : bool
        Enable progress reporting for large operations
    min_chunk_size : int
        Minimum chunk size in data points
    max_chunk_size : int
        Maximum chunk size in data points
    """

    memory_limit_gb: float = 8.0
    gpu_memory_fraction: float | None = None
    enable_mixed_precision_fallback: bool = True
    chunk_size_mb: int | None = None
    out_of_memory_strategy: str = "fallback"
    safety_factor: float = 0.8
    auto_chunk_threshold_gb: float = 4.0
    progress_reporting: bool = True
    min_chunk_size: int = 1000
    max_chunk_size: int = 1_000_000

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.1 <= self.memory_limit_gb <= 1024:
            raise ValueError(
                f"memory_limit_gb must be between 0.1 and 1024, got {self.memory_limit_gb}"
            )

        if self.gpu_memory_fraction is not None:
            if not 0.0 < self.gpu_memory_fraction <= 1.0:
                raise ValueError(
                    f"gpu_memory_fraction must be between 0.0 and 1.0, got {self.gpu_memory_fraction}"
                )

        if not 0.1 <= self.safety_factor <= 1.0:
            raise ValueError(
                f"safety_factor must be between 0.1 and 1.0, got {self.safety_factor}"
            )

        if self.out_of_memory_strategy not in ["fallback", "reduce", "error"]:
            raise ValueError(
                f"out_of_memory_strategy must be 'fallback', 'reduce', or 'error', got {self.out_of_memory_strategy}"
            )

        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) cannot be larger than max_chunk_size ({self.max_chunk_size})"
            )


@dataclass(slots=True)
class LargeDatasetConfig:
    """Configuration for large dataset processing.

    Attributes
    ----------
    enable_automatic_solver_selection : bool
        Automatically select optimal solver based on dataset size
    solver_selection_thresholds : Dict[str, int]
        Thresholds for automatic solver selection

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
        }
    )


class JAXConfig:
    """Singleton configuration manager for JAX and memory settings.

    This class ensures that JAX configuration is set once and consistently
    across all NLSQ modules, avoiding duplicate configuration calls. It also
    manages memory settings, large dataset configuration, and mixed precision
    configuration.
    """

    _instance: Optional["JAXConfig"] = None
    _x64_enabled: bool = False
    _initialized: bool = False
    _memory_config: MemoryConfig | None = None
    _large_dataset_config: LargeDatasetConfig | None = None
    _mixed_precision_config: "MixedPrecisionConfig | None" = None
    _gpu_memory_configured: bool = False

    def __new__(cls) -> "JAXConfig":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize JAX and memory configuration if not already done."""
        if not self._initialized:
            self._initialize_jax()
            self._initialize_memory_config()
            self._initialize_large_dataset_config()
            self._initialize_mixed_precision_config()
            self._initialized = True

    def _initialize_jax(self):
        """Initialize JAX with default NLSQ settings."""
        # Import here to avoid circular imports
        from jax import config

        # Force CPU backend if requested (useful for testing)
        if (
            os.getenv("NLSQ_FORCE_CPU", "0") == "1"
            or os.getenv("JAX_PLATFORM_NAME") == "cpu"
        ):
            config.update("jax_platform_name", "cpu")

        # Enable 64-bit precision by default for NLSQ
        if not self._x64_enabled and os.getenv("NLSQ_DISABLE_X64") != "1":
            config.update("jax_enable_x64", True)
            self._x64_enabled = True

        # Configure persistent compilation cache (eliminates 2-10s cold start)
        self._configure_persistent_cache(config)

        # Configure GPU memory if specified
        self._configure_gpu_memory(config)

    def _configure_persistent_cache(self, config):
        """Configure JAX persistent compilation cache.

        This enables caching of compiled functions across Python sessions,
        eliminating cold-start overhead of 2-10 seconds.
        """
        # Skip if explicitly disabled
        if os.getenv("NLSQ_DISABLE_PERSISTENT_CACHE") == "1":
            return

        # Set cache directory
        cache_dir = os.getenv(
            "NLSQ_JAX_CACHE_DIR", os.path.expanduser("~/.cache/nlsq/jax_cache")
        )

        try:
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)

            # Enable persistent compilation cache
            config.update("jax_compilation_cache_dir", cache_dir)

            # Only cache compilations that take at least 1 second
            min_compile_time = float(os.getenv("NLSQ_CACHE_MIN_COMPILE_TIME_SECS", "1"))
            config.update(
                "jax_persistent_cache_min_compile_time_secs", min_compile_time
            )

            logging.debug(f"JAX persistent cache enabled at {cache_dir}")
        except Exception as e:
            # Non-fatal: log warning and continue without persistent cache
            logging.warning(
                f"Failed to enable JAX persistent compilation cache: {e}. "
                "Cold-start may be slower."
            )

    def _configure_gpu_memory(self, config):
        """Configure GPU memory settings."""
        if self._gpu_memory_configured:
            return

        # Check environment variables for GPU memory settings
        gpu_memory_fraction = os.getenv("NLSQ_GPU_MEMORY_FRACTION")
        if gpu_memory_fraction:
            try:
                fraction = float(gpu_memory_fraction)
                if 0.0 < fraction <= 1.0:
                    # Note: JAX memory fraction is handled differently and may not have a direct config option
                    # This is stored in our config for use by downstream components
                    logging.info(
                        f"Set GPU memory fraction to {fraction} (stored for downstream use)"
                    )
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

        # Configure memory preallocation
        if os.getenv("NLSQ_DISABLE_GPU_PREALLOCATION") == "1":
            try:
                config.update("jax_preallocate_gpu_memory", False)
                logging.info("Disabled GPU memory preallocation")
            except AttributeError:
                # JAX version may not support this option
                logging.warning(
                    "JAX version does not support jax_preallocate_gpu_memory option"
                )

        self._gpu_memory_configured = True

    def _initialize_memory_config(self):
        """Initialize memory configuration from environment variables."""
        if self._memory_config is not None:
            return

        # Load defaults
        memory_config = MemoryConfig()

        # Override from environment variables
        limit = os.getenv("NLSQ_MEMORY_LIMIT_GB")
        if limit:
            try:
                memory_config.memory_limit_gb = float(limit)
            except ValueError:
                warnings.warn(f"Invalid NLSQ_MEMORY_LIMIT_GB: {limit}", stacklevel=2)

        fraction = os.getenv("NLSQ_GPU_MEMORY_FRACTION")
        if fraction:
            try:
                memory_config.gpu_memory_fraction = float(fraction)
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_GPU_MEMORY_FRACTION: {fraction}", stacklevel=2
                )

        if os.getenv("NLSQ_DISABLE_MIXED_PRECISION_FALLBACK") == "1":
            memory_config.enable_mixed_precision_fallback = False

        chunk_size = os.getenv("NLSQ_CHUNK_SIZE_MB")
        if chunk_size:
            try:
                memory_config.chunk_size_mb = int(chunk_size)
            except ValueError:
                warnings.warn(f"Invalid NLSQ_CHUNK_SIZE_MB: {chunk_size}", stacklevel=2)

        if os.getenv("NLSQ_OOM_STRATEGY"):
            strategy = os.getenv("NLSQ_OOM_STRATEGY")
            if strategy in ["fallback", "reduce", "error"]:
                memory_config.out_of_memory_strategy = strategy
            else:
                warnings.warn(
                    f"Invalid NLSQ_OOM_STRATEGY: {strategy}. Must be 'fallback', 'reduce', or 'error'.",
                    stacklevel=2,
                )

        safety_factor = os.getenv("NLSQ_SAFETY_FACTOR")
        if safety_factor:
            try:
                memory_config.safety_factor = float(safety_factor)
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_SAFETY_FACTOR: {safety_factor}", stacklevel=2
                )

        if os.getenv("NLSQ_DISABLE_PROGRESS_REPORTING") == "1":
            memory_config.progress_reporting = False

        self._memory_config = memory_config

    def _initialize_large_dataset_config(self):
        """Initialize large dataset configuration from environment variables."""
        if self._large_dataset_config is not None:
            return

        # Load defaults
        large_dataset_config = LargeDatasetConfig()

        # Emit deprecation warnings for removed sampling environment variables (v0.2.0)
        deprecated_env_vars = {
            "NLSQ_DISABLE_SAMPLING": "Subsampling was removed in v0.2.0. Use streaming optimization instead.",
            "NLSQ_SAMPLING_THRESHOLD": "Subsampling was removed in v0.2.0. Use streaming optimization instead.",
            "NLSQ_MAX_SAMPLED_SIZE": "Subsampling was removed in v0.2.0. Use streaming optimization instead.",
            "NLSQ_SAMPLING_STRATEGY": "Subsampling was removed in v0.2.0. Use streaming optimization instead.",
        }
        for env_var, message in deprecated_env_vars.items():
            if os.getenv(env_var):
                warnings.warn(
                    f"{env_var} environment variable is no longer used. {message} "
                    "See MIGRATION_V0.2.0.md for details.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # Override from environment variables
        if os.getenv("NLSQ_DISABLE_AUTO_SOLVER_SELECTION") == "1":
            large_dataset_config.enable_automatic_solver_selection = False

        self._large_dataset_config = large_dataset_config

    def _initialize_mixed_precision_config(self):
        """Initialize mixed precision configuration from environment variables."""
        if self._mixed_precision_config is not None:
            return

        # Import here to avoid circular imports
        from nlsq.precision.mixed_precision import MixedPrecisionConfig

        # Load defaults
        mp_config = MixedPrecisionConfig()

        # Override from environment variables
        if os.getenv("NLSQ_MIXED_PRECISION_VERBOSE") == "1":
            mp_config.verbose = True
            logging.info(
                "Mixed precision verbose mode enabled via NLSQ_MIXED_PRECISION_VERBOSE"
            )

        # Gradient explosion threshold
        grad_threshold = os.getenv("NLSQ_GRADIENT_EXPLOSION_THRESHOLD")
        if grad_threshold:
            try:
                mp_config.gradient_explosion_threshold = float(grad_threshold)
                logging.info(
                    f"Custom gradient explosion threshold: {mp_config.gradient_explosion_threshold:.2e}"
                )
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_GRADIENT_EXPLOSION_THRESHOLD: {grad_threshold}",
                    stacklevel=2,
                )

        # Precision limit threshold
        prec_threshold = os.getenv("NLSQ_PRECISION_LIMIT_THRESHOLD")
        if prec_threshold:
            try:
                mp_config.precision_limit_threshold = float(prec_threshold)
                logging.info(
                    f"Custom precision limit threshold: {mp_config.precision_limit_threshold:.2e}"
                )
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_PRECISION_LIMIT_THRESHOLD: {prec_threshold}",
                    stacklevel=2,
                )

        # Stall window
        stall_window = os.getenv("NLSQ_STALL_WINDOW")
        if stall_window:
            try:
                mp_config.stall_window = int(stall_window)
                logging.info(
                    f"Custom stall window: {mp_config.stall_window} iterations"
                )
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_STALL_WINDOW: {stall_window}", stacklevel=2
                )

        # Max degradation iterations
        max_degrad = os.getenv("NLSQ_MAX_DEGRADATION_ITERATIONS")
        if max_degrad:
            try:
                mp_config.max_degradation_iterations = int(max_degrad)
                logging.info(
                    f"Custom max degradation iterations: {mp_config.max_degradation_iterations}"
                )
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_MAX_DEGRADATION_ITERATIONS: {max_degrad}",
                    stacklevel=2,
                )

        # Tolerance relaxation factor
        relax_factor = os.getenv("NLSQ_TOLERANCE_RELAXATION_FACTOR")
        if relax_factor:
            try:
                mp_config.tolerance_relaxation_factor = float(relax_factor)
                logging.info(
                    f"Custom tolerance relaxation factor: {mp_config.tolerance_relaxation_factor:.1f}x"
                )
            except ValueError:
                warnings.warn(
                    f"Invalid NLSQ_TOLERANCE_RELAXATION_FACTOR: {relax_factor}",
                    stacklevel=2,
                )

        # Enable/disable mixed precision
        if os.getenv("NLSQ_DISABLE_MIXED_PRECISION") == "1":
            mp_config.enable_mixed_precision_fallback = False
            logging.info("Mixed precision disabled via NLSQ_DISABLE_MIXED_PRECISION")

        self._mixed_precision_config = mp_config

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

        if enable and not instance._x64_enabled:
            config.update("jax_enable_x64", True)
            instance._x64_enabled = True
        elif not enable and instance._x64_enabled:
            config.update("jax_enable_x64", False)
            instance._x64_enabled = False

    @classmethod
    def is_x64_enabled(cls) -> bool:
        """Check if 64-bit precision is enabled.

        Returns
        -------
        bool
            True if 64-bit precision is enabled, False otherwise.
        """
        instance = cls()
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
        original_state = instance._x64_enabled

        try:
            cls.enable_x64(use_x64)
            yield
        finally:
            cls.enable_x64(original_state)

    # Memory configuration methods
    @classmethod
    def get_memory_config(cls) -> MemoryConfig:
        """Get the current memory configuration.

        Returns
        -------
        MemoryConfig
            Current memory configuration
        """
        instance = cls()
        if instance._memory_config is None:
            instance._initialize_memory_config()
        # Explicit validation after initialization (not assert - can be optimized away)
        if instance._memory_config is None:
            raise RuntimeError("Memory config initialization failed")
        return instance._memory_config

    @classmethod
    def set_memory_config(cls, config: MemoryConfig):
        """Set the memory configuration.

        Parameters
        ----------
        config : MemoryConfig
            New memory configuration
        """
        instance = cls()
        instance._memory_config = config

        # Apply GPU memory settings immediately if possible
        try:
            pass

            if config.gpu_memory_fraction is not None:
                # Note: JAX memory fraction handling varies by version and backend
                # This is stored in our config for use by downstream components
                logging.info(
                    f"Updated GPU memory fraction to {config.gpu_memory_fraction} (stored for downstream use)"
                )
        except ImportError:
            pass  # JAX not available

    @classmethod
    def get_large_dataset_config(cls) -> LargeDatasetConfig:
        """Get the current large dataset configuration.

        Returns
        -------
        LargeDatasetConfig
            Current large dataset configuration
        """
        instance = cls()
        if instance._large_dataset_config is None:
            instance._initialize_large_dataset_config()
        # Explicit validation after initialization (not assert - can be optimized away)
        if instance._large_dataset_config is None:
            raise RuntimeError("Large dataset config initialization failed")
        return instance._large_dataset_config

    @classmethod
    def set_large_dataset_config(cls, config: LargeDatasetConfig):
        """Set the large dataset configuration.

        Parameters
        ----------
        config : LargeDatasetConfig
            New large dataset configuration
        """
        instance = cls()
        instance._large_dataset_config = config

    @classmethod
    def get_mixed_precision_config(cls):
        """Get the current mixed precision configuration.

        Returns
        -------
        MixedPrecisionConfig
            Current mixed precision configuration
        """
        instance = cls()
        if instance._mixed_precision_config is None:
            instance._initialize_mixed_precision_config()
        # Explicit validation after initialization (not assert - can be optimized away)
        if instance._mixed_precision_config is None:
            raise RuntimeError("Mixed precision config initialization failed")
        return instance._mixed_precision_config

    @classmethod
    def set_mixed_precision_config(cls, config):
        """Set the mixed precision configuration.

        Parameters
        ----------
        config : MixedPrecisionConfig
            New mixed precision configuration
        """
        instance = cls()
        instance._mixed_precision_config = config

    @classmethod
    @contextmanager
    def memory_context(cls, memory_config: MemoryConfig):
        """Context manager for temporarily changing memory configuration.

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
        instance = cls()
        original_config = instance._memory_config

        try:
            cls.set_memory_config(memory_config)
            yield
        finally:
            if original_config is not None:
                cls.set_memory_config(original_config)
            else:
                instance._memory_config = None
                instance._initialize_memory_config()

    @classmethod
    @contextmanager
    def large_dataset_context(cls, large_dataset_config: LargeDatasetConfig):
        """Context manager for temporarily changing large dataset configuration.

        Parameters
        ----------
        large_dataset_config : LargeDatasetConfig
            Temporary large dataset configuration
        """
        instance = cls()
        original_config = instance._large_dataset_config

        try:
            cls.set_large_dataset_config(large_dataset_config)
            yield
        finally:
            if original_config is not None:
                cls.set_large_dataset_config(original_config)
            else:
                instance._large_dataset_config = None
                instance._initialize_large_dataset_config()

    @classmethod
    @contextmanager
    def mixed_precision_context(cls, mixed_precision_config):
        """Context manager for temporarily changing mixed precision configuration.

        Parameters
        ----------
        mixed_precision_config : MixedPrecisionConfig
            Temporary mixed precision configuration

        Examples
        --------
        >>> from nlsq.config import JAXConfig
        >>> from nlsq.precision.mixed_precision import MixedPrecisionConfig
        >>> temp_config = MixedPrecisionConfig(verbose=True, max_degradation_iterations=2)
        >>> with JAXConfig.mixed_precision_context(temp_config):
        ...     # Code here runs with custom mixed precision settings
        ...     result = curve_fit(func, x, y)
        >>> # Back to previous mixed precision settings
        """
        instance = cls()
        original_config = instance._mixed_precision_config

        try:
            cls.set_mixed_precision_config(mixed_precision_config)
            yield
        finally:
            if original_config is not None:
                cls.set_mixed_precision_config(original_config)
            else:
                instance._mixed_precision_config = None
                instance._initialize_mixed_precision_config()


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
    memory_limit_gb: float,
    gpu_memory_fraction: float | None = None,
    safety_factor: float = 0.8,
):
    """Set memory limits for NLSQ operations.

    Parameters
    ----------
    memory_limit_gb : float
        Maximum memory to use in GB
    gpu_memory_fraction : float, optional
        Fraction of GPU memory to use (0.0-1.0)
    safety_factor : float, optional
        Safety factor for memory calculations (default: 0.8)

    Examples
    --------
    >>> from nlsq.config import set_memory_limits
    >>> # Set 16GB memory limit with 80% GPU memory usage
    >>> set_memory_limits(16.0, gpu_memory_fraction=0.8)
    """
    current_config = get_memory_config()
    new_config = MemoryConfig(
        memory_limit_gb=memory_limit_gb,
        gpu_memory_fraction=gpu_memory_fraction or current_config.gpu_memory_fraction,
        safety_factor=safety_factor,
        enable_mixed_precision_fallback=current_config.enable_mixed_precision_fallback,
        chunk_size_mb=current_config.chunk_size_mb,
        out_of_memory_strategy=current_config.out_of_memory_strategy,
        auto_chunk_threshold_gb=current_config.auto_chunk_threshold_gb,
        progress_reporting=current_config.progress_reporting,
        min_chunk_size=current_config.min_chunk_size,
        max_chunk_size=current_config.max_chunk_size,
    )
    JAXConfig.set_memory_config(new_config)


def enable_mixed_precision_fallback(enable: bool = True):
    """Enable or disable mixed precision fallback for large datasets.

    When enabled, NLSQ will automatically fall back to mixed precision
    (float32) computation for very large datasets to conserve memory.

    Parameters
    ----------
    enable : bool, optional
        Whether to enable mixed precision fallback (default: True)

    Examples
    --------
    >>> from nlsq.config import enable_mixed_precision_fallback
    >>> # Enable mixed precision fallback
    >>> enable_mixed_precision_fallback(True)
    """
    current_config = get_memory_config()
    new_config = MemoryConfig(
        memory_limit_gb=current_config.memory_limit_gb,
        gpu_memory_fraction=current_config.gpu_memory_fraction,
        enable_mixed_precision_fallback=enable,
        chunk_size_mb=current_config.chunk_size_mb,
        out_of_memory_strategy=current_config.out_of_memory_strategy,
        safety_factor=current_config.safety_factor,
        auto_chunk_threshold_gb=current_config.auto_chunk_threshold_gb,
        progress_reporting=current_config.progress_reporting,
        min_chunk_size=current_config.min_chunk_size,
        max_chunk_size=current_config.max_chunk_size,
    )
    JAXConfig.set_memory_config(new_config)


def configure_for_large_datasets(
    memory_limit_gb: float = 8.0,
    enable_chunking: bool = True,
    progress_reporting: bool = True,
    mixed_precision_fallback: bool = True,
):
    """Configure NLSQ for optimal large dataset performance.

    This function sets up memory management, chunking, streaming, and other
    settings for handling large datasets efficiently.

    Parameters
    ----------
    memory_limit_gb : float, optional
        Maximum memory to use in GB (default: 8.0)
    enable_chunking : bool, optional
        Enable automatic data chunking (default: True)
    progress_reporting : bool, optional
        Enable progress reporting for long operations (default: True)
    mixed_precision_fallback : bool, optional
        Enable mixed precision fallback (default: True)

    Notes
    -----
    All large datasets use streaming optimization for 100% data utilization.

    Examples
    --------
    >>> from nlsq.config import configure_for_large_datasets
    >>> # Configure for large datasets with 16GB memory limit
    >>> configure_for_large_datasets(
    ...     memory_limit_gb=16.0,
    ...     progress_reporting=True
    ... )
    """
    # Configure memory settings
    memory_config = MemoryConfig(
        memory_limit_gb=memory_limit_gb,
        enable_mixed_precision_fallback=mixed_precision_fallback,
        auto_chunk_threshold_gb=memory_limit_gb * 0.5
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

    logging.info("Configured NLSQ for large datasets:")
    logging.info(f"  Memory limit: {memory_limit_gb} GB")
    logging.info("  Streaming: enabled (always available)")
    logging.info(f"  Chunking: {'enabled' if enable_chunking else 'disabled'}")
    logging.info(
        f"  Progress reporting: {'enabled' if progress_reporting else 'disabled'}"
    )
    logging.info(
        f"  Mixed precision fallback: {'enabled' if mixed_precision_fallback else 'disabled'}"
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

    Examples
    --------
    >>> from nlsq.config import large_dataset_context, LargeDatasetConfig
    >>> temp_config = LargeDatasetConfig(enable_automatic_solver_selection=True)
    >>> with large_dataset_context(temp_config):
    ...     # Code here uses automatic solver selection
    ...     result = fit_large_dataset(func, x, y)
    """
    return JAXConfig.large_dataset_context(large_dataset_config)


# Mixed precision convenience functions
def get_mixed_precision_config():
    """Get the current mixed precision configuration.

    Returns
    -------
    MixedPrecisionConfig
        Current mixed precision configuration

    Examples
    --------
    >>> from nlsq.config import get_mixed_precision_config
    >>> config = get_mixed_precision_config()
    >>> print(config.gradient_explosion_threshold)
    10000000000.0
    """
    return JAXConfig.get_mixed_precision_config()


def set_mixed_precision_config(config):
    """Set the mixed precision configuration.

    Parameters
    ----------
    config : MixedPrecisionConfig
        New mixed precision configuration

    Examples
    --------
    >>> from nlsq.config import set_mixed_precision_config
    >>> from nlsq.precision.mixed_precision import MixedPrecisionConfig
    >>> config = MixedPrecisionConfig(verbose=True, max_degradation_iterations=10)
    >>> set_mixed_precision_config(config)
    """
    JAXConfig.set_mixed_precision_config(config)


def configure_mixed_precision(
    enable: bool = True,
    max_degradation_iterations: int = 5,
    gradient_explosion_threshold: float = 1e10,
    precision_limit_threshold: float = 1e-7,
    stall_window: int = 10,
    tolerance_relaxation_factor: float = 10.0,
    verbose: bool = False,
):
    """Configure mixed precision optimization behavior.

    This function provides a convenient way to customize mixed precision
    settings without directly constructing a MixedPrecisionConfig object.

    Parameters
    ----------
    enable : bool, optional
        Enable automatic mixed precision fallback (default: True)
    max_degradation_iterations : int, optional
        Number of iterations with issues before upgrading (default: 5)
    gradient_explosion_threshold : float, optional
        Gradient norm threshold for explosion detection (default: 1e10)
    precision_limit_threshold : float, optional
        Minimum parameter change in float32 (default: 1e-7)
    stall_window : int, optional
        Iterations to track for stall detection (default: 10)
    tolerance_relaxation_factor : float, optional
        Factor to multiply tolerances in fallback (default: 10.0)
    verbose : bool, optional
        Enable verbose logging (default: False)

    Examples
    --------
    Conservative configuration (upgrades quickly):

    >>> from nlsq.config import configure_mixed_precision
    >>> configure_mixed_precision(
    ...     max_degradation_iterations=2,
    ...     gradient_explosion_threshold=1e8,
    ...     verbose=True
    ... )

    Aggressive configuration (stays in float32 longer):

    >>> configure_mixed_precision(
    ...     max_degradation_iterations=10,
    ...     gradient_explosion_threshold=1e12,
    ...     precision_limit_threshold=1e-8
    ... )

    Disable mixed precision (pure float64):

    >>> configure_mixed_precision(enable=False)

    Notes
    -----
    Environment variables can override these settings:

    - NLSQ_MIXED_PRECISION_VERBOSE=1
    - NLSQ_GRADIENT_EXPLOSION_THRESHOLD=1e8
    - NLSQ_PRECISION_LIMIT_THRESHOLD=1e-8
    - NLSQ_STALL_WINDOW=15
    - NLSQ_MAX_DEGRADATION_ITERATIONS=3
    - NLSQ_TOLERANCE_RELAXATION_FACTOR=5.0
    - NLSQ_DISABLE_MIXED_PRECISION=1
    """
    from nlsq.precision.mixed_precision import MixedPrecisionConfig

    config = MixedPrecisionConfig(
        enable_mixed_precision_fallback=enable,
        max_degradation_iterations=max_degradation_iterations,
        gradient_explosion_threshold=gradient_explosion_threshold,
        precision_limit_threshold=precision_limit_threshold,
        stall_window=stall_window,
        tolerance_relaxation_factor=tolerance_relaxation_factor,
        verbose=verbose,
    )
    JAXConfig.set_mixed_precision_config(config)

    logging.info("Configured mixed precision optimization:")
    logging.info(f"  Enabled: {enable}")
    if enable:
        logging.info(f"  Max degradation iterations: {max_degradation_iterations}")
        logging.info(
            f"  Gradient explosion threshold: {gradient_explosion_threshold:.2e}"
        )
        logging.info(f"  Precision limit threshold: {precision_limit_threshold:.2e}")
        logging.info(f"  Stall window: {stall_window} iterations")
        logging.info(
            f"  Tolerance relaxation factor: {tolerance_relaxation_factor:.1f}x"
        )
        logging.info(f"  Verbose logging: {verbose}")


def mixed_precision_context(mixed_precision_config):
    """Context manager for temporarily changing mixed precision configuration.

    Parameters
    ----------
    mixed_precision_config : MixedPrecisionConfig
        Temporary mixed precision configuration

    Examples
    --------
    >>> from nlsq.config import mixed_precision_context
    >>> from nlsq.precision.mixed_precision import MixedPrecisionConfig
    >>> temp_config = MixedPrecisionConfig(verbose=True, max_degradation_iterations=2)
    >>> with mixed_precision_context(temp_config):
    ...     # Code here runs with custom mixed precision settings
    ...     result = curve_fit(func, x, y)
    >>> # Back to previous mixed precision settings
    """
    return JAXConfig.mixed_precision_context(mixed_precision_config)


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
        else:
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
                if "jacobian_mode" in config:
                    mode = config["jacobian_mode"]
                    if mode in ("auto", "fwd", "rev"):
                        return mode, "config file"
                    else:
                        warnings.warn(
                            f"Invalid jacobian_mode in config file: {mode}. Must be 'auto', 'fwd', or 'rev'. Using auto-default.",
                            stacklevel=2,
                        )
        except (OSError, json.JSONDecodeError) as e:
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
            f"Invalid jacobian_mode: {mode}. Must be 'auto', 'fwd', or 'rev'."
        )

    os.environ["NLSQ_JACOBIAN_MODE"] = mode
    logging.info(f"Set Jacobian mode to '{mode}' via environment variable")
