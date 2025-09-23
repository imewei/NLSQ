"""Central configuration management for NLSQ package."""
from typing import Optional
import os
from contextlib import contextmanager


class JAXConfig:
    """Singleton configuration manager for JAX settings.

    This class ensures that JAX configuration is set once and consistently
    across all NLSQ modules, avoiding duplicate configuration calls.
    """

    _instance: Optional['JAXConfig'] = None
    _x64_enabled: bool = False
    _initialized: bool = False

    def __new__(cls) -> 'JAXConfig':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize JAX configuration if not already done."""
        if not self._initialized:
            self._initialize_jax()
            self._initialized = True

    def _initialize_jax(self):
        """Initialize JAX with default NLSQ settings."""
        # Import here to avoid circular imports
        from jax import config

        # Force CPU backend if requested (useful for testing)
        if os.getenv('NLSQ_FORCE_CPU', '0') == '1' or os.getenv('JAX_PLATFORM_NAME') == 'cpu':
            config.update("jax_platform_name", "cpu")

        # Enable 64-bit precision by default for NLSQ
        if not self._x64_enabled and os.getenv('NLSQ_DISABLE_X64') != '1':
            config.update("jax_enable_x64", True)
            self._x64_enabled = True

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
