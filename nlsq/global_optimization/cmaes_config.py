"""CMA-ES configuration for global optimization.

This module provides configuration dataclasses and utilities for CMA-ES
optimization using the evosax library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

__all__ = [
    "CMAES_PRESETS",
    "CMAESConfig",
    "get_evosax_import_error",
    "is_evosax_available",
]

logger = logging.getLogger(__name__)

# Global cache for evosax availability check
_EVOSAX_AVAILABLE: bool | None = None
_EVOSAX_IMPORT_ERROR: str | None = None


def is_evosax_available() -> bool:
    """Check if evosax is available for import.

    Uses lazy import pattern to avoid import overhead until needed.
    Caches the result for subsequent calls.

    Returns
    -------
    bool
        True if evosax can be imported successfully, False otherwise.
    """
    global _EVOSAX_AVAILABLE, _EVOSAX_IMPORT_ERROR  # noqa: PLW0603

    if _EVOSAX_AVAILABLE is None:
        try:
            from evosax.algorithms import CMA_ES  # type: ignore[import-not-found]
            from evosax.core.restart import (  # type: ignore[import-not-found]
                cma_cond,
                spread_cond,
            )

            _EVOSAX_AVAILABLE = True
            _EVOSAX_IMPORT_ERROR = None
        except ImportError as e:
            _EVOSAX_AVAILABLE = False
            _EVOSAX_IMPORT_ERROR = str(e)
            logger.info(
                "evosax not available - CMA-ES will fall back to multi-start. "
                "Install with: pip install 'nlsq[global]'"
            )

    return _EVOSAX_AVAILABLE


def get_evosax_import_error() -> str | None:
    """Get the import error message if evosax is not available.

    Returns
    -------
    str | None
        The import error message, or None if evosax is available.
    """
    is_evosax_available()  # Ensure availability check has been performed
    return _EVOSAX_IMPORT_ERROR


@dataclass(slots=True)
class CMAESConfig:
    """Configuration for CMA-ES global optimization.

    Attributes
    ----------
    popsize : int | None
        Population size. If None, uses CMA-ES default: int(4 + 3 * log(n)).
    max_generations : int
        Maximum number of generations before stopping. Default: 100.
    sigma : float
        Initial step size (standard deviation). Default: 0.5.
    tol_fun : float
        Function value tolerance for convergence. Default: 1e-8.
    tol_x : float
        Parameter tolerance for convergence. Default: 1e-8.
    restart_strategy : Literal['none', 'bipop']
        Restart strategy. Default: 'bipop'.
    max_restarts : int
        Maximum restart attempts for BIPOP. Default: 9.
    refine_with_nlsq : bool
        Whether to refine best solution with NLSQ TRF. Default: True.
    seed : int | None
        Random seed for reproducibility. If None, uses random seed.

    Examples
    --------
    >>> config = CMAESConfig(popsize=32, max_generations=200)
    >>> config = CMAESConfig.from_preset('cmaes-global')
    """

    # Population and generations
    popsize: int | None = None  # None = auto: int(4 + 3 * log(n))
    max_generations: int = 100

    # Step size and tolerances
    sigma: float = 0.5
    tol_fun: float = 1e-8
    tol_x: float = 1e-8

    # Restart strategy (BIPOP enabled by default per spec)
    restart_strategy: Literal["none", "bipop"] = "bipop"
    max_restarts: int = 9

    # NLSQ refinement
    refine_with_nlsq: bool = True

    # Reproducibility
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.popsize is not None and self.popsize < 4:
            raise ValueError(f"popsize must be >= 4, got {self.popsize}")

        if self.max_generations < 1:
            raise ValueError(
                f"max_generations must be >= 1, got {self.max_generations}"
            )

        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

        if self.tol_fun <= 0:
            raise ValueError(f"tol_fun must be > 0, got {self.tol_fun}")

        if self.tol_x <= 0:
            raise ValueError(f"tol_x must be > 0, got {self.tol_x}")

        if self.max_restarts < 0:
            raise ValueError(f"max_restarts must be >= 0, got {self.max_restarts}")

        if self.restart_strategy not in ("none", "bipop"):
            raise ValueError(
                f"restart_strategy must be 'none' or 'bipop', "
                f"got '{self.restart_strategy}'"
            )

    @classmethod
    def from_preset(cls, preset_name: str) -> CMAESConfig:
        """Create a CMAESConfig from a named preset.

        Parameters
        ----------
        preset_name : str
            Name of the preset. One of 'cmaes-fast', 'cmaes', 'cmaes-global'.

        Returns
        -------
        CMAESConfig
            Configuration for the specified preset.

        Raises
        ------
        ValueError
            If preset_name is not recognized.

        Examples
        --------
        >>> config = CMAESConfig.from_preset('cmaes-fast')
        >>> config.max_generations
        50
        """
        if preset_name not in CMAES_PRESETS:
            available = ", ".join(sorted(CMAES_PRESETS.keys()))
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )

        preset_config = CMAES_PRESETS[preset_name]
        return cls(**preset_config)


# CMA-ES presets - source of truth for CMA-ES preset configurations
CMAES_PRESETS: dict[str, dict] = {
    "cmaes-fast": {
        "popsize": None,  # auto
        "max_generations": 50,
        "restart_strategy": "none",
        "max_restarts": 0,
    },
    "cmaes": {
        "popsize": None,  # auto
        "max_generations": 100,
        "restart_strategy": "bipop",
        "max_restarts": 9,
    },
    "cmaes-global": {
        "popsize": None,  # Will be doubled in optimizer (2x auto)
        "max_generations": 200,
        "restart_strategy": "bipop",
        "max_restarts": 9,
    },
}
