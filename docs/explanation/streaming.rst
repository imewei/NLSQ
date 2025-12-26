Streaming Optimizer Comparison
==============================

This guide compares NLSQ's two streaming optimizers for large datasets:
**StreamingOptimizer** and **AdaptiveHybridStreamingOptimizer**.

Overview
--------

Both optimizers handle datasets too large to fit in memory by processing data
in batches. However, they use fundamentally different optimization strategies:

.. list-table:: Quick Comparison
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - StreamingOptimizer
     - AdaptiveHybridStreamingOptimizer
   * - **Algorithm**
     - Pure Adam (gradient descent)
     - 4-phase hybrid (Adam → Gauss-Newton)
   * - **Convergence Rate**
     - Linear (slow near optimum)
     - Quadratic (fast near optimum)
   * - **Parameter Scaling**
     - None
     - Automatic normalization
   * - **Covariance Estimation**
     - Approximate (Hessian estimate)
     - Exact J^T J accumulation
   * - **Multi-start**
     - No
     - Yes (tournament selection)
   * - **Mixed Precision**
     - No
     - Yes (auto float32 → float64)
   * - **Multi-device**
     - No
     - Yes (JAX pmap support)
   * - **Defense Layers (v0.3.6+)**
     - No
     - Yes (4-layer warmup protection)

StreamingOptimizer
------------------

A production-ready streaming optimizer using Adam gradient descent with
comprehensive fault tolerance features.

**Key Features:**

- Pure Adam optimizer with adaptive learning rate
- Checkpoint save/resume for long-running fits
- NaN/Inf detection at three critical points
- Adaptive retry strategies for error recovery
- Success rate validation (configurable threshold)
- Detailed diagnostics and batch statistics

**Best For:**

- Simple models with well-scaled parameters
- Maximum speed when high precision is not critical
- Trusted data with minimal numerical issues

**Example:**

.. code-block:: python

    from nlsq import StreamingOptimizer, StreamingConfig
    import jax.numpy as jnp


    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c


    config = StreamingConfig(
        batch_size=50000,
        max_epochs=20,
        learning_rate=0.001,
        enable_fault_tolerance=True,
    )

    optimizer = StreamingOptimizer(config)
    result = optimizer.fit_streaming(model, data_source, p0=[1.0, 0.5, 0.1])

AdaptiveHybridStreamingOptimizer
--------------------------------

A four-phase hybrid optimizer that combines parameter normalization, Adam
warmup, streaming Gauss-Newton, and exact covariance computation.

**The Four Phases:**

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Phase
     - Name
     - Purpose
   * - 0
     - Setup
     - Parameter normalization and bounds transformation
   * - 1
     - Adam Warmup
     - Fast initial convergence with adaptive switching criteria
   * - 2
     - Gauss-Newton
     - Streaming GN with exact J^T J accumulation
   * - 3
     - Finalize
     - Denormalize parameters and transform covariance

**Key Advantages:**

1. **Parameter Normalization (Phase 0)**

   Automatically scales parameters to similar magnitudes, solving the common
   problem of weak gradient signals from multi-scale parameters.

   .. code-block:: python

       # Problem: amplitude ~1000, decay ~0.001, offset ~1
       # StreamingOptimizer: Gradients dominated by amplitude
       # AdaptiveHybrid: All parameters contribute equally


       def model(x, amplitude, decay, offset):
           return amplitude * jnp.exp(-decay * x) + offset

2. **Hybrid Algorithm (Phase 1 → Phase 2)**

   Combines the global convergence of Adam with the fast local convergence
   of Gauss-Newton:

   - **Phase 1 (Adam)**: Robust initial approach, handles non-convexity
   - **Phase 2 (Gauss-Newton)**: Quadratic convergence near optimum

   Adaptive switching detects when to transition based on:

   - Loss plateau (relative improvement below threshold)
   - Small gradient norm (indicates proximity to optimum)
   - Maximum iterations reached (safety limit)

3. **Exact Covariance Estimation**

   Streams through all data to accumulate the exact J^T J matrix,
   providing accurate parameter uncertainties even for huge datasets.

   .. code-block:: python

       # StreamingOptimizer: Approximate covariance from final batch
       # AdaptiveHybrid: Exact covariance from ALL data points

4. **Multi-Start with Tournament Selection**

   Generates multiple starting points and uses tournament elimination
   to find the best candidate, avoiding local minima.

   .. code-block:: python

       config = HybridStreamingConfig(
           enable_multistart=True,
           n_starts=10,  # 10 starting points
           multistart_sampler="lhs",  # Latin Hypercube Sampling
           elimination_rounds=3,  # Tournament rounds
           elimination_fraction=0.5,  # Eliminate half each round
       )

5. **Mixed Precision Strategy**

   Automatically uses float32 for fast warmup and float64 for accurate
   Gauss-Newton convergence:

   .. code-block:: python

       config = HybridStreamingConfig(precision="auto")
       # Phase 1: float32 (fast, memory-efficient)
       # Phase 2+: float64 (numerical stability)

6. **Multi-Device Support**

   Distributes Jacobian computation across multiple GPUs/TPUs:

   .. code-block:: python

       config = HybridStreamingConfig(enable_multi_device=True)
       # Uses JAX pmap for data-parallel computation

7. **4-Layer Defense Strategy (v0.3.6+)**

   Prevents Adam warmup divergence when initial parameters are near optimal.
   This is critical for warm-start refinement scenarios where a previous fit
   provides the starting point.

   The four layers:

   - **Layer 1 (Warm Start Detection)**: Skips warmup if initial loss < 1% of
     data variance
   - **Layer 2 (Adaptive Learning Rate)**: Scales LR based on fit quality
     (1e-6 to 0.001)
   - **Layer 3 (Cost-Increase Guard)**: Aborts if loss increases > 5%
   - **Layer 4 (Step Clipping)**: Limits parameter update magnitude (max
     norm 0.1)

   .. code-block:: python

       # Defense presets for common scenarios
       config = HybridStreamingConfig.defense_strict()  # Warm-start refinement
       config = HybridStreamingConfig.defense_relaxed()  # Exploration
       config = HybridStreamingConfig.scientific_default()  # Production scientific
       config = HybridStreamingConfig.defense_disabled()  # Pre-0.3.6 behavior

**Example:**

.. code-block:: python

    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig
    import jax.numpy as jnp


    def model(x, amplitude, decay, offset):
        return amplitude * jnp.exp(-decay * x) + offset


    config = HybridStreamingConfig(
        normalize=True,
        normalization_strategy="auto",
        warmup_iterations=200,
        warmup_learning_rate=0.001,
        gauss_newton_tol=1e-8,
        enable_multistart=True,
        n_starts=5,
    )

    optimizer = AdaptiveHybridStreamingOptimizer(config)

    # Or via curve_fit interface:
    from nlsq import curve_fit

    popt, pcov = curve_fit(model, x, y, p0=[1000, 0.001, 1], method="hybrid_streaming")

When to Use Which
-----------------

.. list-table:: Decision Guide
   :header-rows: 1
   :widths: 50 25 25

   * - Use Case
     - StreamingOptimizer
     - AdaptiveHybridStreaming
   * - Simple fits, well-scaled parameters
     - Recommended
     - Works
   * - Multi-scale parameters (e.g., 1e6 vs 1e-6)
     - Poor
     - **Recommended**
   * - Need accurate uncertainties (pcov)
     - Approximate
     - **Exact**
   * - Risk of local minima
     - No protection
     - **Multi-start**
   * - Multi-GPU/TPU deployment
     - Not supported
     - **Supported**
   * - Maximum speed, less accuracy needed
     - **Recommended**
     - Slower
   * - Production fault tolerance
     - **Full support**
     - Full support
   * - Checkpoint/resume
     - **Full support**
     - Full support
   * - Warm-start refinement (v0.3.6+)
     - No protection
     - **4-layer defense**

**Decision Flowchart:**

1. **Are you refining parameters from a previous fit (warm-start)?**

   - Yes → Use ``AdaptiveHybridStreamingOptimizer`` with ``defense_strict()``
   - No → Continue

2. **Are parameters on vastly different scales?**

   - Yes → Use ``AdaptiveHybridStreamingOptimizer``
   - No → Continue

3. **Do you need accurate parameter uncertainties?**

   - Yes → Use ``AdaptiveHybridStreamingOptimizer``
   - No → Continue

4. **Is there risk of local minima?**

   - Yes → Use ``AdaptiveHybridStreamingOptimizer`` with multi-start
   - No → Continue

5. **Do you have multiple GPUs/TPUs?**

   - Yes → Use ``AdaptiveHybridStreamingOptimizer``
   - No → Use ``StreamingOptimizer`` for maximum speed

Configuration Defaults
----------------------

**StreamingConfig Defaults:**

.. code-block:: python

    StreamingConfig(
        batch_size=10000,
        max_epochs=100,
        learning_rate=0.001,
        enable_fault_tolerance=True,
        validate_numerics=True,
        checkpoint_frequency=100,
        min_success_rate=0.5,
    )

**HybridStreamingConfig Defaults:**

.. code-block:: python

    HybridStreamingConfig(
        # Phase 0: Normalization
        normalize=True,
        normalization_strategy="auto",
        # Phase 1: Adam warmup
        warmup_iterations=200,
        max_warmup_iterations=500,
        warmup_learning_rate=0.001,
        loss_plateau_threshold=1e-4,
        gradient_norm_threshold=1e-3,
        # Phase 2: Gauss-Newton
        gauss_newton_max_iterations=100,
        gauss_newton_tol=1e-8,
        trust_region_initial=1.0,
        regularization_factor=1e-10,
        chunk_size=10000,
        # Fault tolerance
        enable_checkpoints=True,
        checkpoint_frequency=100,
        validate_numerics=True,
        # Multi-start (disabled by default)
        enable_multistart=False,
        n_starts=10,
        multistart_sampler="lhs",
    )

**Pre-built Profiles:**

.. code-block:: python

    # Aggressive: Faster convergence, more warmup
    config = HybridStreamingConfig.aggressive()

    # Conservative: More stable, slower convergence
    config = HybridStreamingConfig.conservative()

    # Balanced: Middle ground
    config = HybridStreamingConfig.balanced()

**Defense Presets (v0.3.6+):**

.. code-block:: python

    # Strictest protection for warm-start refinement
    config = HybridStreamingConfig.defense_strict()

    # More aggressive learning for exploration
    config = HybridStreamingConfig.defense_relaxed()

    # Balanced for production scientific computing
    config = HybridStreamingConfig.scientific_default()

    # Disable defense layers (pre-0.3.6 behavior)
    config = HybridStreamingConfig.defense_disabled()

Performance Characteristics
---------------------------

**Convergence Speed:**

- ``StreamingOptimizer``: Linear convergence, O(1/k) improvement per iteration
- ``AdaptiveHybridStreamingOptimizer``: Quadratic convergence near optimum after
  switching to Gauss-Newton

**Memory Usage:**

- ``StreamingOptimizer``: O(batch_size × n_params)
- ``AdaptiveHybridStreamingOptimizer``: O(batch_size × n_params) + O(n_params²)
  for J^T J accumulator

**Computational Cost per Iteration:**

- ``StreamingOptimizer``: Gradient computation only
- ``AdaptiveHybridStreamingOptimizer``: Jacobian computation + matrix operations
  (Phase 2)

Example: Multi-Scale Parameter Problem
--------------------------------------

This example demonstrates where ``AdaptiveHybridStreamingOptimizer`` excels:

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from nlsq import (
        StreamingOptimizer,
        StreamingConfig,
        AdaptiveHybridStreamingOptimizer,
        HybridStreamingConfig,
    )


    # Model with multi-scale parameters
    def multi_scale_model(x, amplitude, decay, offset):
        """
        Parameters on vastly different scales:
        - amplitude: ~1000
        - decay: ~0.001
        - offset: ~1
        """
        return amplitude * jnp.exp(-decay * x) + offset


    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 1000, 1_000_000)
    true_params = [1000.0, 0.001, 1.0]
    y_true = multi_scale_model(x, *true_params)
    y = y_true + 5 * np.random.randn(len(x))

    # Initial guess
    p0 = [800.0, 0.002, 2.0]

    # StreamingOptimizer: May struggle with scale imbalance
    stream_config = StreamingConfig(batch_size=50000, max_epochs=50)
    stream_opt = StreamingOptimizer(stream_config)
    # Gradients dominated by amplitude, slow convergence for decay

    # AdaptiveHybridStreamingOptimizer: Handles multi-scale automatically
    hybrid_config = HybridStreamingConfig(
        normalize=True,
        normalization_strategy="auto",
        warmup_iterations=100,
        gauss_newton_tol=1e-10,
    )
    hybrid_opt = AdaptiveHybridStreamingOptimizer(hybrid_config)
    # Parameters normalized → balanced gradients → fast convergence

See Also
--------

- :doc:`../howto/handle_large_data` - Large dataset handling guide
- :doc:`../reference/large_data` - Large data API reference
- :doc:`/api/nlsq.streaming_optimizer` - StreamingOptimizer API reference
- :doc:`/api/nlsq.adaptive_hybrid_streaming` - AdaptiveHybridStreamingOptimizer API reference
- :doc:`/api/nlsq.streaming_config` - StreamingConfig API reference
- :doc:`/api/nlsq.hybrid_streaming_config` - HybridStreamingConfig API reference
