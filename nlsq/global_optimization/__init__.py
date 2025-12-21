"""
Global Optimization for NLSQ
============================

Multi-start optimization with Latin Hypercube Sampling and other quasi-random
samplers for global search in nonlinear least squares fitting.

This package provides tools for exploring parameter space to find global optima,
which is particularly useful for problems with multiple local minima.

Key Features
------------
- **Latin Hypercube Sampling (LHS)**: Stratified random sampling for better coverage
- **Quasi-random sequences**: Sobol and Halton for deterministic, low-discrepancy sampling
- **Multi-start orchestration**: Evaluate multiple starting points and select best
- **Tournament selection**: Memory-efficient selection for streaming datasets
- **Preset configurations**: 'fast', 'robust', 'global', 'thorough', 'streaming'

Examples
--------
Basic sampling:

>>> from nlsq.global_optimization import latin_hypercube_sample
>>> import jax
>>> samples = latin_hypercube_sample(10, 3, rng_key=jax.random.PRNGKey(42))

Using configuration presets:

>>> from nlsq.global_optimization import GlobalOptimizationConfig
>>> config = GlobalOptimizationConfig.from_preset('robust')
>>> print(config.n_starts)
5

Multi-start optimization:

>>> from nlsq.global_optimization import MultiStartOrchestrator
>>> orchestrator = MultiStartOrchestrator.from_preset('robust')
>>> result = orchestrator.fit(model, x, y, bounds=([0, 0], [10, 10]))

Tournament selection for large datasets:

>>> from nlsq.global_optimization import TournamentSelector, GlobalOptimizationConfig
>>> candidates = latin_hypercube_sample(20, 3)
>>> config = GlobalOptimizationConfig.from_preset('streaming')
>>> selector = TournamentSelector(candidates, config)
>>> best = selector.run_tournament(data_batch_iter, model, top_m=1)

Notes
-----
Multi-start optimization is integrated with the existing NLSQ infrastructure:

- For small datasets (<1M points): Full multi-start on complete data
- For medium datasets (1M-100M): Full multi-start, then chunked fit
- For large datasets (>100M): Tournament selection during streaming warmup

See Also
--------
nlsq.curve_fit : Main curve fitting function with multi-start support
nlsq.LargeDatasetFitter : Chunked processing for medium-large datasets
nlsq.AdaptiveHybridStreamingOptimizer : Streaming for very large datasets
"""

from nlsq.global_optimization.config import (
    PRESETS,
    GlobalOptimizationConfig,
)
from nlsq.global_optimization.multi_start import MultiStartOrchestrator
from nlsq.global_optimization.sampling import (
    center_samples_around_p0,
    get_sampler,
    halton_sample,
    latin_hypercube_sample,
    scale_samples_to_bounds,
    sobol_sample,
)
from nlsq.global_optimization.tournament import TournamentSelector

__all__ = [
    "PRESETS",
    # Configuration
    "GlobalOptimizationConfig",
    # Multi-start orchestration
    "MultiStartOrchestrator",
    # Tournament selection for large datasets
    "TournamentSelector",
    "center_samples_around_p0",
    "get_sampler",
    "halton_sample",
    # Sampling functions
    "latin_hypercube_sample",
    "scale_samples_to_bounds",
    "sobol_sample",
]
