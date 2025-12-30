nlsq.adaptive\_hybrid\_streaming module
=======================================

.. currentmodule:: nlsq.streaming.adaptive_hybrid

.. automodule:: nlsq.streaming.adaptive_hybrid
   :noindex:

Overview
--------

The ``nlsq.adaptive_hybrid_streaming`` module implements a four-phase hybrid
optimizer that solves three fundamental issues in streaming optimization:

1. **Weak gradient signals** from parameter scale imbalance (via normalization)
2. **Slow convergence** near optimum (via Gauss-Newton)
3. **Crude covariance estimation** (via exact J^T J accumulation)

**New in version 0.3.0**: Complete adaptive hybrid streaming optimizer.

Key Features
------------

- **Four-phase optimization**: Automatic phase transitions for optimal convergence
- **Parameter normalization**: Address gradient signal weakness from scale imbalance
- **Adam warmup**: First-order optimization with 4-layer divergence protection
- **Streaming Gauss-Newton**: Second-order convergence with streaming J^T J
- **Exact covariance**: Production-quality uncertainty estimates
- **Fault tolerance**: Checkpointing, validation, and automatic recovery
- **Multi-device support**: GPU/TPU parallelism for large datasets
- **Defense telemetry**: Production monitoring of warmup protection layers

**New in version 0.3.6**: 4-Layer Defense Strategy for Adam warmup divergence prevention.

Optimization Phases
-------------------

Phase 0: Normalization Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sets up parameter normalization to address gradient signal weakness:

- Determines normalization strategy (bounds-based, p0-based, or none)
- Creates ``ParameterNormalizer`` with scales and offsets
- Wraps user model for transparent normalized parameter space
- Transforms bounds to normalized space
- Stores normalization Jacobian for Phase 3

Phase 1: Adam Warmup with 4-Layer Defense
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First-order optimization with adaptive switching and divergence protection:

- Uses Optax Adam optimizer with configurable learning rate
- Optional learning rate schedule with warmup and decay
- **4-Layer Defense Strategy** (new in 0.3.6):

  1. **Warm Start Detection**: Skip warmup if initial loss already low
  2. **Adaptive Learning Rate**: Scale LR based on initial loss quality
  3. **Cost-Increase Guard**: Abort if loss increases beyond tolerance
  4. **Step Clipping**: Limit update magnitude for stability

- Monitors loss plateau and gradient norm for switch criteria
- Builds momentum and explores parameter space
- Switches to Phase 2 when ready for fine-tuning

.. seealso::

   :doc:`../explanation/how_fitting_works` for complete optimization strategy documentation.

Phase 2: Streaming Gauss-Newton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Second-order optimization with exact J^T J accumulation:

- Streams data in chunks for memory efficiency
- Accumulates exact J^T J matrix (not stochastic approximation)
- Trust region step control with Levenberg-Marquardt regularization
- Converges quickly near optimum with quadratic rate
- Provides production-quality parameter estimates

Phase 3: Denormalization and Covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finalizes optimization and computes uncertainties:

- Denormalizes parameters to original space
- Transforms covariance matrix using normalization Jacobian
- Returns final result with parameter uncertainties

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AdaptiveHybridStreamingOptimizer
   DefenseLayerTelemetry

.. autoclass:: AdaptiveHybridStreamingOptimizer
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: DefenseLayerTelemetry
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Functions
---------

.. autosummary::
   :toctree: generated/

   get_defense_telemetry
   reset_defense_telemetry

.. autofunction:: get_defense_telemetry
   :noindex:

.. autofunction:: reset_defense_telemetry
   :noindex:

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

Create an optimizer with default configuration:

.. code-block:: python

    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig
    import jax.numpy as jnp

    # Default configuration
    config = HybridStreamingConfig()
    optimizer = AdaptiveHybridStreamingOptimizer(config)


    # Define model
    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c


    # Prepare data
    x_data = jnp.linspace(0, 10, 1000000)  # 1M points
    y_data = 2.5 * jnp.exp(-0.3 * x_data) + 0.5 + 0.1 * jnp.random.normal(key, (1000000,))

    # Fit (API for full implementation)
    result = optimizer.fit(
        model=exponential,
        x_data=x_data,
        y_data=y_data,
        p0=[2.0, 0.5, 0.3],
    )

With Bounds-Based Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use parameter bounds for normalization:

.. code-block:: python

    config = HybridStreamingConfig(
        normalize=True,
        normalization_strategy="bounds",
    )
    optimizer = AdaptiveHybridStreamingOptimizer(config)

    # Parameters: amplitude [1, 10], decay [0.1, 1.0], offset [0, 2]
    bounds = (jnp.array([1.0, 0.1, 0.0]), jnp.array([10.0, 1.0, 2.0]))

    result = optimizer.fit(
        model=exponential,
        x_data=x_data,
        y_data=y_data,
        p0=[2.0, 0.5, 0.3],
        bounds=bounds,
    )

With Aggressive Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fast convergence with looser tolerances:

.. code-block:: python

    config = HybridStreamingConfig.aggressive()
    optimizer = AdaptiveHybridStreamingOptimizer(config)

    # Faster warmup, larger chunks, earlier switching
    result = optimizer.fit(model, x_data, y_data, p0)

With Conservative Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Higher quality with tighter tolerances:

.. code-block:: python

    config = HybridStreamingConfig.conservative()
    optimizer = AdaptiveHybridStreamingOptimizer(config)

    # Tighter tolerance, more Gauss-Newton iterations
    result = optimizer.fit(model, x_data, y_data, p0)

Memory-Optimized Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimize memory footprint:

.. code-block:: python

    config = HybridStreamingConfig.memory_optimized()
    optimizer = AdaptiveHybridStreamingOptimizer(config)

    # Smaller chunks, float32, frequent checkpoints
    result = optimizer.fit(model, x_data, y_data, p0)

Custom Switching Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~

Control when Phase 1 switches to Phase 2:

.. code-block:: python

    config = HybridStreamingConfig(
        warmup_iterations=300,
        max_warmup_iterations=800,
        loss_plateau_threshold=1e-5,  # Tighter plateau detection
        gradient_norm_threshold=1e-4,  # Lower gradient threshold
        active_switching_criteria=["plateau", "gradient"],  # Remove max_iter
    )
    optimizer = AdaptiveHybridStreamingOptimizer(config)

With Fault Tolerance
~~~~~~~~~~~~~~~~~~~~

Enable checkpointing for long optimizations:

.. code-block:: python

    config = HybridStreamingConfig(
        enable_checkpoints=True,
        checkpoint_frequency=50,
        checkpoint_dir="/path/to/checkpoints",
        validate_numerics=True,
        enable_fault_tolerance=True,
        max_retries_per_batch=3,
    )
    optimizer = AdaptiveHybridStreamingOptimizer(config)

    # Resume from checkpoint
    config_resume = HybridStreamingConfig(resume_from_checkpoint="/path/to/checkpoint.pkl")

Phase Tracking
~~~~~~~~~~~~~~

Monitor optimization progress:

.. code-block:: python

    optimizer = AdaptiveHybridStreamingOptimizer(config)

    # After fitting
    print(f"Final phase: {optimizer.current_phase}")
    print(f"Phase history: {optimizer.phase_history}")

    # Each entry in phase_history:
    # {'phase': 0, 'start_time': 1234567890.0, 'duration': 0.5}

Defense Layer Telemetry
~~~~~~~~~~~~~~~~~~~~~~~

**New in version 0.3.6**: Monitor defense layer activations.

Track when defense layers trigger across multiple fits:

.. code-block:: python

    from nlsq import get_defense_telemetry, reset_defense_telemetry

    # Reset telemetry counters
    reset_defense_telemetry()

    # Run multiple fits
    for dataset in datasets:
        result = optimizer.fit(model, x_data, y_data, p0)

    # Get telemetry summary
    telemetry = get_defense_telemetry()
    print(telemetry.get_summary())

    # Get activation rates
    rates = telemetry.get_trigger_rates()
    print(f"Warm start rate: {rates['layer1_warm_start_rate']:.1f}%")
    print(f"Cost guard rate: {rates['layer3_cost_guard_rate']:.1f}%")

    # Export for Prometheus/Grafana
    metrics = telemetry.export_metrics()

Defense Layer Presets
~~~~~~~~~~~~~~~~~~~~~

**New in version 0.3.6**: Sensitivity presets for defense layers.

.. code-block:: python

    # Strict defense for near-optimal scenarios
    config = HybridStreamingConfig.defense_strict()

    # Relaxed defense for exploration
    config = HybridStreamingConfig.defense_relaxed()

    # Disable all defense layers
    config = HybridStreamingConfig.defense_disabled()

    # Scientific computing optimized
    config = HybridStreamingConfig.scientific_default()

Architecture
------------

Optimization Flow
~~~~~~~~~~~~~~~~~

.. code-block:: text

    Input: model, x_data, y_data, p0, bounds
           │
           ▼
    ┌──────────────────────────────────────┐
    │  Phase 0: Normalization Setup        │
    │  - Create ParameterNormalizer        │
    │  - Wrap model function               │
    │  - Transform bounds                  │
    │  - Store normalization Jacobian      │
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │  Phase 1: Adam Warmup                │
    │  - Optax Adam optimizer              │
    │  - Monitor loss/gradient             │
    │  - Check switching criteria          │
    └──────────────────────────────────────┘
           │ (switch when criteria met)
           ▼
    ┌──────────────────────────────────────┐
    │  Phase 2: Streaming Gauss-Newton     │
    │  - Stream data in chunks             │
    │  - Accumulate exact J^T J            │
    │  - Trust region optimization         │
    └──────────────────────────────────────┘
           │ (converged)
           ▼
    ┌──────────────────────────────────────┐
    │  Phase 3: Denormalization            │
    │  - Denormalize parameters            │
    │  - Transform covariance: J @ C @ J.T │
    │  - Return final result               │
    └──────────────────────────────────────┘
           │
           ▼
    Output: OptimizeResult with popt, pcov

Key Attributes
~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Attribute
     - Description
   * - ``current_phase``
     - Current optimization phase (0, 1, 2, or 3)
   * - ``phase_history``
     - List of phase transitions with timing
   * - ``normalizer``
     - ParameterNormalizer instance
   * - ``normalized_model``
     - Wrapped model for normalized space
   * - ``normalized_bounds``
     - Bounds in normalized space
   * - ``normalization_jacobian``
     - Jacobian for covariance transform
   * - ``best_params_global``
     - Best parameters found (for fault tolerance)
   * - ``best_cost_global``
     - Best cost found (for fault tolerance)

Performance Characteristics
---------------------------

Convergence Speed
~~~~~~~~~~~~~~~~~

- **Phase 1 (Adam)**: Linear convergence, robust to initialization
- **Phase 2 (Gauss-Newton)**: Quadratic convergence near optimum
- **Overall**: Faster than pure first-order or pure second-order

Memory Usage
~~~~~~~~~~~~

- **Streaming**: Fixed memory footprint regardless of dataset size
- **J^T J accumulation**: O(n_params^2) memory
- **Chunks**: Configurable via ``chunk_size``

Throughput
~~~~~~~~~~

- CPU: 50,000 - 200,000 samples/second
- GPU: 500,000 - 2,000,000 samples/second
- Depends on model complexity

When to Use
-----------

**Use Adaptive Hybrid Streaming when:**

- Parameters span many orders of magnitude (gradient imbalance)
- Dataset is large (100K+ points)
- Need production-quality uncertainty estimates
- Standard optimizers converge slowly
- Memory is limited relative to dataset size

**Use standard curve_fit when:**

- Dataset fits in memory
- Parameters have similar scales
- Simple models with good initialization
- Don't need streaming

See Also
--------

- :doc:`nlsq.hybrid_streaming_config` : Configuration options
- :doc:`nlsq.parameter_normalizer` : Parameter normalization
- :doc:`nlsq.streaming_optimizer` : Original streaming optimizer
- :doc:`nlsq.stability` : Numerical stability utilities
- :doc:`../howto/handle_large_data` : Large dataset guide
