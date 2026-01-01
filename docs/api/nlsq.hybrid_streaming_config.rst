nlsq.hybrid\_streaming\_config module
=====================================

.. currentmodule:: nlsq.streaming.hybrid_config

.. automodule:: nlsq.streaming.hybrid_config
   :noindex:

Overview
--------

The ``nlsq.hybrid_streaming_config`` module provides configuration options for the
four-phase adaptive hybrid streaming optimizer. This configuration controls all
aspects of the optimization process including parameter normalization, L-BFGS warmup,
streaming Gauss-Newton, and covariance computation.

**New in version 0.3.0**: Complete configuration for adaptive hybrid streaming.

Key Features
------------

- **Phase 0**: Parameter normalization configuration (bounds-based, p0-based, or none)
- **Phase 1**: L-BFGS warmup with configurable step sizes and switching criteria
- **4-Layer Defense Strategy** (new in 0.3.6): Protection against warmup divergence
- **Phase 2**: Streaming Gauss-Newton with trust region and regularization control
- **Phase 3**: Denormalization and covariance transform settings
- **Fault tolerance**: Checkpointing, validation, and retry configuration
- **Multi-device**: GPU/TPU parallelism settings
- **Presets**: Ready-to-use profiles for common use cases

**New in version 0.3.6**: 4-layer defense strategy parameters and sensitivity presets.

Classes
-------

.. autoclass:: HybridStreamingConfig
   :members:
   :special-members: __init__, __post_init__
   :undoc-members:
   :show-inheritance:
   :noindex:

Configuration Presets
---------------------

The ``HybridStreamingConfig`` class provides factory methods for common use cases.

Performance Profiles
~~~~~~~~~~~~~~~~~~~~

Aggressive Profile
^^^^^^^^^^^^^^^^^^

Fast convergence, more warmup, looser tolerances:

.. code-block:: python

    from nlsq import HybridStreamingConfig

    config = HybridStreamingConfig.aggressive()
    # Larger warmup: 300-800 iterations
    # Larger initial step size: 0.5
    # Larger chunks: 20000
    # Looser tolerances

Conservative Profile
^^^^^^^^^^^^^^^^^^^^

Slower but robust, tighter tolerances:

.. code-block:: python

    config = HybridStreamingConfig.conservative()
    # Smaller warmup: 100-300 iterations
    # Smaller initial step size: 0.05
    # Tighter tolerance: 1e-10
    # Smaller trust region: 0.5

Memory-Optimized Profile
^^^^^^^^^^^^^^^^^^^^^^^^^

Minimizes memory footprint:

.. code-block:: python

    config = HybridStreamingConfig.memory_optimized()
    # Smaller chunks: 5000
    # float32 precision
    # Frequent checkpoints: every 50 iterations

Defense Layer Sensitivity Presets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**New in version 0.3.6**

Defense Strict
^^^^^^^^^^^^^^

Maximum protection for near-optimal scenarios (warm starts, refinement):

.. code-block:: python

    config = HybridStreamingConfig.defense_strict()
    # Very low warm start threshold (1%)
    # Ultra-conservative step sizes
    # Tight cost guard tolerance (5%)
    # Very small step clipping (0.05)

Use when:

- Continuing from previous fit
- Refining near-optimal parameters
- Ill-conditioned problems
- Prioritizing stability over speed

Defense Relaxed
^^^^^^^^^^^^^^^

Relaxed protection for exploration-heavy scenarios:

.. code-block:: python

    config = HybridStreamingConfig.defense_relaxed()
    # High warm start threshold (50%)
    # Aggressive step sizes
    # Generous cost guard tolerance (50%)
    # Larger step clipping (0.5)

Use when:

- Starting from rough initial guess
- Exploring wide parameter space
- Problems with multiple local minima
- Speed more important than robustness

Defense Disabled
^^^^^^^^^^^^^^^^

Disable all defense layers (reverts to pre-0.3.6 behavior):

.. code-block:: python

    config = HybridStreamingConfig.defense_disabled()

.. warning::

   Use with caution! Removes protection against warmup divergence.

Use when:

- Debugging to isolate defense layer effects
- Benchmarking without defense overhead
- Backward compatibility required

Scientific Default
^^^^^^^^^^^^^^^^^^

Optimized for scientific computing workflows (XPCS, scattering, spectroscopy):

.. code-block:: python

    config = HybridStreamingConfig.scientific_default()
    # Balanced defense layers
    # Float64 precision
    # Tight Gauss-Newton tolerances (1e-10)
    # Enabled checkpoints

Use when:

- Fitting physics-based models
- Numerical precision is critical
- Parameters span multiple scales
- Reproducibility required

Usage Examples
--------------

Default Configuration
~~~~~~~~~~~~~~~~~~~~~

Create an optimizer with default settings:

.. code-block:: python

    from nlsq import HybridStreamingConfig, AdaptiveHybridStreamingOptimizer

    config = HybridStreamingConfig()
    optimizer = AdaptiveHybridStreamingOptimizer(config)

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

Fine-tune specific parameters:

.. code-block:: python

    config = HybridStreamingConfig(
        # Normalization
        normalize=True,
        normalization_strategy="bounds",  # 'auto', 'bounds', 'p0', 'none'
        # Phase 1: L-BFGS warmup
        warmup_iterations=300,
        max_warmup_iterations=800,
        lbfgs_history_size=15,
        lbfgs_initial_step_size=0.5,
        lbfgs_line_search="backtracking",
        lbfgs_exploration_step_size=0.1,
        lbfgs_refinement_step_size=1.0,
        loss_plateau_threshold=5e-4,
        gradient_norm_threshold=5e-3,
        # Phase 2: Gauss-Newton
        gauss_newton_max_iterations=150,
        gauss_newton_tol=1e-9,
        trust_region_initial=0.5,
        regularization_factor=1e-8,
        # Streaming
        chunk_size=20000,
        # Fault tolerance
        enable_checkpoints=True,
        checkpoint_frequency=50,
        validate_numerics=True,
        # Precision
        precision="float64",  # 'auto', 'float32', 'float64'
    )

Normalization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~

Configure how parameters are normalized:

.. code-block:: python

    # Auto-detect: use bounds if provided, else p0-based
    config = HybridStreamingConfig(normalization_strategy="auto")

    # Normalize to [0, 1] using parameter bounds
    config = HybridStreamingConfig(normalization_strategy="bounds")

    # Scale by initial parameter magnitudes
    config = HybridStreamingConfig(normalization_strategy="p0")

    # No normalization (identity transform)
    config = HybridStreamingConfig(normalization_strategy="none")

Switching Criteria
~~~~~~~~~~~~~~~~~~

Control when Phase 1 switches to Phase 2:

.. code-block:: python

    config = HybridStreamingConfig(
        # Any of these criteria can trigger switch
        active_switching_criteria=["plateau", "gradient", "max_iter"],
        # Loss plateau detection threshold
        loss_plateau_threshold=1e-4,
        # Gradient norm threshold
        gradient_norm_threshold=1e-3,
        # Maximum warmup iterations
        max_warmup_iterations=500,
    )

L-BFGS Options
~~~~~~~~~~~~~~

Configure L-BFGS behavior and line search:

.. code-block:: python

    config = HybridStreamingConfig(
        lbfgs_history_size=10,
        lbfgs_initial_step_size=0.1,
        lbfgs_line_search="wolfe",  # "wolfe", "strong_wolfe", "backtracking"
        lbfgs_exploration_step_size=0.1,
        lbfgs_refinement_step_size=1.0,
        gradient_clip_value=1.0,
    )

Configuration Parameters
------------------------

Phase 0: Normalization
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``normalize``
     - ``True``
     - Enable parameter normalization
   * - ``normalization_strategy``
     - ``'auto'``
     - Strategy: 'auto', 'bounds', 'p0', 'none'

Phase 1: L-BFGS Warmup
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``warmup_iterations``
     - 200
     - Initial warmup iterations before checking switch
   * - ``max_warmup_iterations``
     - 500
     - Maximum warmup before forced switch
   * - ``lbfgs_history_size``
     - 10
     - L-BFGS history size
   * - ``lbfgs_initial_step_size``
     - 0.1
     - Initial step size for L-BFGS line search
   * - ``lbfgs_line_search``
     - ``'wolfe'``
     - Line search strategy ('wolfe', 'strong_wolfe', 'backtracking')
   * - ``lbfgs_exploration_step_size``
     - 0.1
     - Step size for exploration mode
   * - ``lbfgs_refinement_step_size``
     - 1.0
     - Step size for refinement mode
   * - ``loss_plateau_threshold``
     - 1e-4
     - Relative loss improvement for plateau detection
   * - ``gradient_norm_threshold``
     - 1e-3
     - Gradient norm for early switch

4-Layer Defense Strategy (New in 0.3.6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - **Layer 1: Warm Start Detection**
     -
     -
   * - ``enable_warm_start_detection``
     - ``True``
     - Enable/disable warm start detection
   * - ``warm_start_threshold``
     - 0.01
     - Relative loss threshold (skip if < threshold)
   * - **Layer 2: Adaptive Step Size**
     -
     -
   * - ``enable_adaptive_warmup_lr``
     - ``True``
     - Enable/disable adaptive step size selection
   * - ``warmup_lr_refinement``
     - 1e-6
     - Step size for excellent fits (relative_loss < 0.1)
   * - ``warmup_lr_careful``
     - 1e-5
     - Step size for good fits (0.1 ≤ relative_loss < 1.0)
   * - **Layer 3: Cost-Increase Guard**
     -
     -
   * - ``enable_cost_guard``
     - ``True``
     - Enable/disable cost-increase guard
   * - ``cost_increase_tolerance``
     - 0.05
     - Max allowed loss increase (5%)
   * - **Layer 4: Step Clipping**
     -
     -
   * - ``enable_step_clipping``
     - ``True``
     - Enable/disable step clipping
   * - ``max_warmup_step_size``
     - 0.1
     - Maximum L2 norm of parameter update

.. seealso::

   :doc:`../explanation/how_fitting_works` for complete optimization strategy documentation.

Phase 2: Gauss-Newton
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``gauss_newton_max_iterations``
     - 100
     - Maximum Gauss-Newton iterations
   * - ``gauss_newton_tol``
     - 1e-8
     - Convergence tolerance
   * - ``trust_region_initial``
     - 1.0
     - Initial trust region radius
   * - ``regularization_factor``
     - 1e-10
     - Regularization for rank-deficient matrices

See Also
--------

- :doc:`nlsq.adaptive_hybrid_streaming` : Main optimizer class
- :doc:`nlsq.parameter_normalizer` : Parameter normalization implementation
- :doc:`../howto/handle_large_data` : Large dataset guide
