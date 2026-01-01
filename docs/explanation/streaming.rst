Adaptive Hybrid Streaming Optimizer
===================================

NLSQ provides a single streaming optimizer for huge datasets:
``AdaptiveHybridStreamingOptimizer``. It combines parameter normalization,
L-BFGS warmup, streaming Gauss-Newton, and exact covariance accumulation.

Overview
--------

The adaptive hybrid optimizer runs in four phases:

1. **Normalization**: Rescales parameters for stable gradients.
2. **L-BFGS warmup**: Fast initial convergence with defense layers.
3. **Streaming Gauss-Newton**: Chunked Jacobian accumulation with bounded memory.
4. **Denormalization + covariance**: Returns parameters and uncertainties.

When to Use
-----------

- Datasets too large to keep in memory.
- Models with multi-scale parameters that need normalization.
- Cases where you need covariance estimates in streaming runs.

Quick Start
-----------

Use the high-level API:

.. code-block:: python

    from nlsq import curve_fit

    popt, pcov = curve_fit(
        model,
        x,
        y,
        p0=p0,
        method="hybrid_streaming",
        verbose=1,
    )

Or configure the optimizer directly:

.. code-block:: python

    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    config = HybridStreamingConfig(
        chunk_size=50000,
        gauss_newton_max_iterations=100,
        enable_checkpoints=True,
        checkpoint_frequency=100,
    )

    optimizer = AdaptiveHybridStreamingOptimizer(config)
    result = optimizer.fit((x, y), model, p0=p0, verbose=1)

    popt = result["x"]
    pcov = result["pcov"]

Defense Presets
---------------

Warmup defense presets tune L-BFGS behavior:

- ``HybridStreamingConfig.defense_strict()``: Warm-start refinement
- ``HybridStreamingConfig.defense_relaxed()``: Exploration
- ``HybridStreamingConfig.scientific_default()``: Production scientific
- ``HybridStreamingConfig.defense_disabled()``: Disable defense layers

Decision Guide
--------------

- Need accurate covariance estimates at scale? Use adaptive hybrid streaming.
- Want fewer manual tuning knobs? Use ``HybridStreamingConfig`` presets.
- Need fault tolerance? Enable checkpoints in the config.

See Also
--------

- :doc:`/api/nlsq.adaptive_hybrid_streaming` - API reference
- :doc:`/api/nlsq.hybrid_streaming_config` - Configuration reference
- :doc:`/howto/streaming_checkpoints` - Checkpointing guide
