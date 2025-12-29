"""
Global Optimization Configuration
=================================

Configuration dataclass for multi-start optimization with Latin Hypercube Sampling.

This module provides the GlobalOptimizationConfig dataclass which controls all
aspects of multi-start global optimization, including sampling strategy, starting
point generation, and tournament selection for large datasets.

Examples
--------
Basic configuration with defaults:

>>> from nlsq.global_optimization import GlobalOptimizationConfig
>>> config = GlobalOptimizationConfig()
>>> config.n_starts
10

Using presets:

>>> config = GlobalOptimizationConfig.from_preset('robust')
>>> config.n_starts
5

>>> config = GlobalOptimizationConfig.from_preset('global')
>>> config.n_starts
20

Custom configuration:

>>> config = GlobalOptimizationConfig(
...     n_starts=30,
...     sampler='sobol',
...     center_on_p0=True,
...     scale_factor=0.5,
... )
"""

from dataclasses import dataclass, field
from typing import Any, Literal

# Preset configurations for common use cases
PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "n_starts": 0,
        "sampler": "lhs",
        "center_on_p0": False,
        "scale_factor": 1.0,
        "elimination_rounds": 0,
        "elimination_fraction": 0.5,
        "batches_per_round": 50,
    },
    "robust": {
        "n_starts": 5,
        "sampler": "lhs",
        "center_on_p0": True,
        "scale_factor": 1.0,
        "elimination_rounds": 2,
        "elimination_fraction": 0.5,
        "batches_per_round": 50,
    },
    "global": {
        "n_starts": 20,
        "sampler": "lhs",
        "center_on_p0": True,
        "scale_factor": 1.0,
        "elimination_rounds": 3,
        "elimination_fraction": 0.5,
        "batches_per_round": 100,
    },
    "thorough": {
        "n_starts": 50,
        "sampler": "lhs",
        "center_on_p0": True,
        "scale_factor": 1.0,
        "elimination_rounds": 4,
        "elimination_fraction": 0.5,
        "batches_per_round": 150,
    },
    "streaming": {
        "n_starts": 10,
        "sampler": "lhs",
        "center_on_p0": True,
        "scale_factor": 1.0,
        "elimination_rounds": 3,
        "elimination_fraction": 0.5,
        "batches_per_round": 50,
    },
}


@dataclass(slots=True)
class GlobalOptimizationConfig:
    """Configuration for multi-start global optimization.

    This configuration class controls all aspects of multi-start optimization
    with Latin Hypercube Sampling or other quasi-random samplers.

    Parameters
    ----------
    n_starts : int, default=10
        Number of starting points to generate. Set to 0 to disable multi-start.

    sampler : {'lhs', 'sobol', 'halton'}, default='lhs'
        Sampling strategy for generating starting points:

        - 'lhs': Latin Hypercube Sampling (recommended, stratified random)
        - 'sobol': Sobol quasi-random sequence (deterministic, low-discrepancy)
        - 'halton': Halton quasi-random sequence (deterministic, prime bases)

    center_on_p0 : bool, default=True
        Whether to center starting points around the initial parameter guess (p0).
        When True, samples are generated in a region around p0 rather than
        uniformly in the full parameter bounds.

    scale_factor : float, default=1.0
        Scale factor for exploration region when center_on_p0=True.
        Multiplier for the exploration range around p0.
        Smaller values (0.5) = tighter exploration around p0.
        Larger values (2.0) = wider exploration.

    elimination_rounds : int, default=3
        Number of tournament elimination rounds for large datasets.
        Each round eliminates a fraction of candidates based on loss.

    elimination_fraction : float, default=0.5
        Fraction of candidates to eliminate in each tournament round.
        Must be in (0, 1). Default 0.5 eliminates half in each round.

    batches_per_round : int, default=50
        Number of data batches to use for evaluation in each tournament round.
        More batches = more reliable selection but slower.

    Examples
    --------
    >>> config = GlobalOptimizationConfig(n_starts=20, sampler='sobol')
    >>> config.n_starts
    20

    >>> config = GlobalOptimizationConfig.from_preset('global')
    >>> config.n_starts
    20

    Notes
    -----
    - When n_starts=0, multi-start is disabled and standard single-start
      optimization is used.
    - Tournament selection (elimination_rounds > 0) is designed for streaming
      datasets where evaluating all candidates on the full dataset is impractical.
    - LHS provides better coverage guarantees than Sobol/Halton for small N,
      while Sobol/Halton are deterministic and may be preferred for reproducibility.

    See Also
    --------
    MultiStartOrchestrator : Orchestrates multi-start optimization
    TournamentSelector : Implements tournament selection for large datasets
    """

    # Sampling configuration
    n_starts: int = 10
    sampler: Literal["lhs", "sobol", "halton"] = "lhs"

    # Centering configuration
    center_on_p0: bool = True
    scale_factor: float = 1.0

    # Tournament selection for large datasets
    elimination_rounds: int = 3
    elimination_fraction: float = 0.5
    batches_per_round: int = 50

    # Private field for tracking preset origin (not user-configurable)
    _preset: str | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate n_starts
        if self.n_starts < 0:
            raise ValueError(f"n_starts must be non-negative, got {self.n_starts}")

        # Validate sampler
        valid_samplers = ("lhs", "sobol", "halton")
        if self.sampler.lower() not in valid_samplers:
            raise ValueError(
                f"sampler must be one of {valid_samplers}, got '{self.sampler}'"
            )
        # Normalize sampler to lowercase
        object.__setattr__(self, "sampler", self.sampler.lower())

        # Validate scale_factor
        if self.scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {self.scale_factor}")

        # Validate elimination_fraction
        if not 0 < self.elimination_fraction < 1:
            raise ValueError(
                f"elimination_fraction must be in (0, 1), got {self.elimination_fraction}"
            )

        # Validate elimination_rounds
        if self.elimination_rounds < 0:
            raise ValueError(
                f"elimination_rounds must be non-negative, got {self.elimination_rounds}"
            )

        # Validate batches_per_round
        if self.batches_per_round <= 0:
            raise ValueError(
                f"batches_per_round must be positive, got {self.batches_per_round}"
            )

        # Validate parameter combinations
        if self.n_starts == 0 and self.elimination_rounds > 0:
            # Tournament selection makes no sense with 0 starts
            # Silently set elimination_rounds to 0
            object.__setattr__(self, "elimination_rounds", 0)

    @classmethod
    def from_preset(cls, preset_name: str) -> "GlobalOptimizationConfig":
        """Create configuration from a named preset.

        Parameters
        ----------
        preset_name : str
            Name of the preset. One of: 'fast', 'robust', 'global',
            'thorough', 'streaming'.

        Returns
        -------
        GlobalOptimizationConfig
            Configuration instance with preset values.

        Raises
        ------
        ValueError
            If preset_name is not a known preset.

        Examples
        --------
        >>> config = GlobalOptimizationConfig.from_preset('robust')
        >>> config.n_starts
        5

        >>> config = GlobalOptimizationConfig.from_preset('global')
        >>> config.n_starts
        20
        """
        preset_name_lower = preset_name.lower()
        if preset_name_lower not in PRESETS:
            valid_presets = list(PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. Valid presets: {valid_presets}"
            )

        preset_values = PRESETS[preset_name_lower].copy()
        preset_values["_preset"] = preset_name_lower
        return cls(**preset_values)

    @property
    def is_multistart_enabled(self) -> bool:
        """Whether multi-start optimization is enabled.

        Returns
        -------
        bool
            True if n_starts > 0, False otherwise.
        """
        return self.n_starts > 0

    @property
    def preset(self) -> str | None:
        """The preset name if this config was created from a preset.

        Returns
        -------
        str or None
            Preset name ('fast', 'robust', etc.) or None if custom.
        """
        return self._preset

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary.

        Returns
        -------
        dict
            Dictionary representation suitable for JSON serialization
            or checkpoint saving.

        Examples
        --------
        >>> config = GlobalOptimizationConfig(n_starts=20)
        >>> d = config.to_dict()
        >>> d['n_starts']
        20
        """
        return {
            "n_starts": self.n_starts,
            "sampler": self.sampler,
            "center_on_p0": self.center_on_p0,
            "scale_factor": self.scale_factor,
            "elimination_rounds": self.elimination_rounds,
            "elimination_fraction": self.elimination_fraction,
            "batches_per_round": self.batches_per_round,
            "_preset": self._preset,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GlobalOptimizationConfig":
        """Deserialize configuration from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with configuration values.

        Returns
        -------
        GlobalOptimizationConfig
            Configuration instance.

        Examples
        --------
        >>> d = {'n_starts': 20, 'sampler': 'sobol'}
        >>> config = GlobalOptimizationConfig.from_dict(d)
        >>> config.n_starts
        20
        """
        # Filter to known fields
        known_fields = {
            "n_starts",
            "sampler",
            "center_on_p0",
            "scale_factor",
            "elimination_rounds",
            "elimination_fraction",
            "batches_per_round",
            "_preset",
        }
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def with_overrides(self, **kwargs: Any) -> "GlobalOptimizationConfig":
        """Create a new config with specified overrides.

        Parameters
        ----------
        **kwargs
            Configuration fields to override.

        Returns
        -------
        GlobalOptimizationConfig
            New configuration with overrides applied.

        Examples
        --------
        >>> config = GlobalOptimizationConfig.from_preset('robust')
        >>> config2 = config.with_overrides(n_starts=10)
        >>> config2.n_starts
        10
        """
        d = self.to_dict()
        d.update(kwargs)
        # Clear preset if we're overriding values
        if kwargs and "_preset" not in kwargs:
            d["_preset"] = None
        return self.from_dict(d)
