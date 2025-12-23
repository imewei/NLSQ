4-Layer Defense Strategy
========================

.. currentmodule:: nlsq

.. versionadded:: 0.3.6

The **4-Layer Defense Strategy** prevents Adam warmup divergence when initial
parameters are already near optimal. This is critical for warm-start scenarios,
refinement workflows, and multi-scale parameter fitting.

Overview
--------

When using the Adaptive Hybrid Streaming Optimizer (``method='hybrid_streaming'``),
Adam warmup can sometimes **push parameters away** from good initial guesses,
especially when:

- Continuing optimization from a previous fit (warm start)
- Initial parameters are already close to optimal
- Parameters span multiple scales (e.g., amplitude=1e6, rate=1e-6)
- Learning rate is too aggressive for near-optimal states

The 4-layer defense strategy automatically detects these scenarios and applies
appropriate protections to maintain stability while preserving convergence speed.

The Four Defense Layers
------------------------

Layer 1: Warm Start Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Skip Adam warmup entirely when initial parameters are already near optimal.

**How It Works**:

1. Compute initial loss and data variance
2. Calculate relative loss: ``initial_loss / data_variance``
3. If relative loss < ``warm_start_threshold``, skip Adam warmup
4. Jump directly to Gauss-Newton phase (Phase 2)

**Configuration**:

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enable_warm_start_detection``
     - ``True``
     - Enable/disable Layer 1
   * - ``warm_start_threshold``
     - ``0.01``
     - Relative loss threshold (1% of variance)

**Example**:

.. code-block:: python

    from nlsq import curve_fit, HybridStreamingConfig

    # Strict warm start detection (1% threshold)
    config = HybridStreamingConfig(
        enable_warm_start_detection=True,
        warm_start_threshold=0.01,  # Skip if loss < 1% of variance
    )

    popt, pcov = curve_fit(
        model, x, y, p0=near_optimal_guess, method="hybrid_streaming", config=config
    )

**When Layer 1 Triggers**:

.. code-block:: text

    Initial loss: 0.008  (relative to variance: 0.005)
    Threshold:    0.01
    Decision:     SKIP ADAM WARMUP (warm start detected)
    → Proceeding directly to Gauss-Newton phase

Layer 2: Adaptive Learning Rate Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Automatically scale learning rate based on initial loss quality.

**How It Works**:

Adam learning rate is selected based on relative loss:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Relative Loss
     - LR Mode
     - Learning Rate
   * - < 0.1
     - ``refinement``
     - ``warmup_lr_refinement`` (1e-6)
   * - 0.1 – 1.0
     - ``careful``
     - ``warmup_lr_careful`` (1e-5)
   * - ≥ 1.0
     - ``exploration``
     - ``warmup_learning_rate`` (0.001)

**Configuration**:

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enable_adaptive_warmup_lr``
     - ``True``
     - Enable/disable Layer 2
   * - ``warmup_lr_refinement``
     - ``1e-6``
     - Ultra-conservative LR for excellent fits
   * - ``warmup_lr_careful``
     - ``1e-5``
     - Conservative LR for good fits
   * - ``warmup_learning_rate``
     - ``0.001``
     - Exploration LR for poor fits

**Example**:

.. code-block:: python

    config = HybridStreamingConfig(
        enable_adaptive_warmup_lr=True,
        warmup_lr_refinement=1e-7,  # Even more conservative
        warmup_lr_careful=1e-6,
        warmup_learning_rate=0.001,
    )

**When Layer 2 Activates**:

.. code-block:: text

    Relative loss: 0.05 (excellent initial guess)
    LR mode:       refinement
    LR selected:   1e-6
    Reason:        Near-optimal, use ultra-conservative LR

Layer 3: Cost-Increase Guard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Abort warmup if loss increases beyond tolerance, preventing divergence.

**How It Works**:

1. Track initial loss at start of warmup
2. Monitor loss at each Adam iteration
3. Compute cost increase ratio: ``current_loss / initial_loss``
4. If ratio > ``1 + cost_increase_tolerance``, abort warmup
5. Return **best parameters** found (not diverged parameters)

**Configuration**:

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enable_cost_guard``
     - ``True``
     - Enable/disable Layer 3
   * - ``cost_increase_tolerance``
     - ``0.05``
     - Max allowed loss increase (5%)

**Example**:

.. code-block:: python

    config = HybridStreamingConfig(
        enable_cost_guard=True,
        cost_increase_tolerance=0.05,  # Abort if loss > initial * 1.05
    )

**When Layer 3 Triggers**:

.. code-block:: text

    Iteration 5:
        Current loss:  0.0125
        Initial loss:  0.0100
        Cost ratio:    1.25 (25% increase)
        Tolerance:     1.05 (5% allowed)
        Decision:      ABORT WARMUP (cost guard triggered)
    Returning best parameters from iteration 3 (loss: 0.0095)

Layer 4: Step Clipping
~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Limit Adam update magnitude to prevent large jumps that overshoot optimum.

**How It Works**:

1. Compute Adam update: ``Δp = optimizer.update(gradient)``
2. Calculate update norm: ``||Δp||₂``
3. If norm > ``max_warmup_step_size``, clip to max:

   .. math::

      \text{clipped}(\Delta p) = \Delta p \cdot \frac{\text{max\_norm}}{||\Delta p||_2}

4. Apply clipped update (preserves direction, scales magnitude)

**Configuration**:

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enable_step_clipping``
     - ``True``
     - Enable/disable Layer 4
   * - ``max_warmup_step_size``
     - ``0.1``
     - Maximum L2 norm of parameter update

**Example**:

.. code-block:: python

    config = HybridStreamingConfig(
        enable_step_clipping=True,
        max_warmup_step_size=0.05,  # Very conservative
    )

**When Layer 4 Clips**:

.. code-block:: text

    Adam update:     [0.5, -0.3, 0.2]
    Update norm:     0.62
    Max allowed:     0.1
    Clipping ratio:  0.161
    Clipped update:  [0.081, -0.048, 0.032]

Preset Configurations
---------------------

NLSQ provides four sensitivity presets for common scenarios.

Defense Strict
~~~~~~~~~~~~~~

**When to use**: Near-optimal scenarios, warm starts, refinement workflows.

Maximum protection against divergence:

- Very low warm start threshold (1% relative loss triggers skip)
- Ultra-conservative learning rates
- Tight cost guard tolerance (5% increase aborts)
- Very small step clipping (0.05 max norm)

.. code-block:: python

    from nlsq import HybridStreamingConfig

    config = HybridStreamingConfig.defense_strict()

    # Equivalent to:
    config = HybridStreamingConfig(
        enable_warm_start_detection=True,
        warm_start_threshold=0.01,
        enable_adaptive_warmup_lr=True,
        warmup_lr_refinement=1e-7,
        warmup_lr_careful=1e-6,
        warmup_learning_rate=0.0005,
        enable_cost_guard=True,
        cost_increase_tolerance=0.05,
        enable_step_clipping=True,
        max_warmup_step_size=0.05,
    )

**Best for**:

- Continuing optimization from previous fit
- Refining parameters already close to optimal
- Ill-conditioned problems
- Prioritizing stability over speed

Defense Relaxed
~~~~~~~~~~~~~~~

**When to use**: Exploration-heavy scenarios, rough initial guesses.

Reduced defense sensitivity for aggressive exploration:

- High warm start threshold (50% relative loss needed to skip)
- Aggressive learning rates for faster exploration
- Generous cost guard tolerance (50% increase allowed)
- Larger step clipping for faster convergence

.. code-block:: python

    config = HybridStreamingConfig.defense_relaxed()

    # Equivalent to:
    config = HybridStreamingConfig(
        enable_warm_start_detection=True,
        warm_start_threshold=0.5,
        enable_adaptive_warmup_lr=True,
        warmup_lr_refinement=1e-5,
        warmup_lr_careful=1e-4,
        warmup_learning_rate=0.003,
        enable_cost_guard=True,
        cost_increase_tolerance=0.5,
        enable_step_clipping=True,
        max_warmup_step_size=0.5,
    )

**Best for**:

- Starting from rough initial guess
- Exploring wide parameter space
- Problems with multiple local minima
- Speed more important than robustness

Defense Disabled
~~~~~~~~~~~~~~~~

**When to use**: Debugging, benchmarking, backward compatibility.

Completely disables all defense layers (reverts to pre-0.3.6 behavior):

.. code-block:: python

    config = HybridStreamingConfig.defense_disabled()

    # Equivalent to:
    config = HybridStreamingConfig(
        enable_warm_start_detection=False,
        enable_adaptive_warmup_lr=False,
        enable_cost_guard=False,
        enable_step_clipping=False,
    )

.. warning::

   Use with caution! This removes protection against Adam warmup divergence.

**Best for**:

- Isolating defense layer effects during debugging
- Performance benchmarking without defense overhead
- Backward compatibility with older code

Scientific Default
~~~~~~~~~~~~~~~~~~

**When to use**: Scientific computing workflows (XPCS, scattering, spectroscopy).

Balanced defense tuned for physics-based models:

- Float64 precision for numerical accuracy
- Moderate defense layers (not too aggressive, not too relaxed)
- Enabled checkpoints for long-running fits
- Tight Gauss-Newton tolerances

.. code-block:: python

    config = HybridStreamingConfig.scientific_default()

    # Includes:
    # - Balanced defense thresholds
    # - precision='float64'
    # - gauss_newton_tol=1e-10
    # - enable_checkpoints=True

**Best for**:

- Fitting physics-based models (XPCS, scattering, decay curves)
- Numerical precision is critical
- Parameters span multiple scales
- Reproducibility required

Telemetry and Monitoring
-------------------------

The defense layer system includes comprehensive telemetry for production monitoring.

DefenseLayerTelemetry Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tracks when each layer is triggered:

.. code-block:: python

    from nlsq import get_defense_telemetry, reset_defense_telemetry

    # Reset counters before monitoring session
    reset_defense_telemetry()

    # Run multiple fits
    for dataset in datasets:
        popt, pcov = curve_fit(model, x, y, method="hybrid_streaming")

    # Get telemetry summary
    telemetry = get_defense_telemetry()
    summary = telemetry.get_summary()

    print(summary)
    # Output:
    # Defense Layer Telemetry Summary:
    # ================================
    # Total warmup calls: 100
    #
    # Layer 1 (Warm Start):
    #   Triggers: 15 (15.0%)
    #
    # Layer 2 (Adaptive LR):
    #   Refinement: 20 (20.0%)
    #   Careful:    50 (50.0%)
    #   Exploration: 30 (30.0%)
    #
    # Layer 3 (Cost Guard):
    #   Triggers: 5 (5.0%)
    #
    # Layer 4 (Step Clipping):
    #   Clips: 75 (75.0%)

Trigger Rates
~~~~~~~~~~~~~

Get activation rates for each layer:

.. code-block:: python

    rates = telemetry.get_trigger_rates()

    print(f"Warm start rate: {rates['layer1_warm_start_rate']:.1f}%")
    print(f"Refinement LR rate: {rates['layer2_refinement_rate']:.1f}%")
    print(f"Cost guard rate: {rates['layer3_cost_guard_rate']:.1f}%")

Prometheus/Grafana Export
~~~~~~~~~~~~~~~~~~~~~~~~~~

Export metrics for monitoring dashboards:

.. code-block:: python

    metrics = telemetry.export_metrics()

    # Prometheus format:
    # nlsq_defense_layer1_warm_start_total 15
    # nlsq_defense_layer2_refinement_total 20
    # nlsq_defense_layer2_careful_total 50
    # nlsq_defense_layer2_exploration_total 30
    # nlsq_defense_layer3_cost_guard_total 5
    # nlsq_defense_layer4_clip_total 75
    # nlsq_defense_warmup_calls_total 100

Detailed Event Log
~~~~~~~~~~~~~~~~~~

Access recent events with context:

.. code-block:: python

    # Get last 10 events
    events = telemetry._event_log[-10:]

    for event in events:
        print(f"{event['type']}: {event['data']}")

    # Output:
    # layer1_warm_start: {'relative_loss': 0.005, 'threshold': 0.01}
    # layer2_lr_mode: {'mode': 'refinement', 'relative_loss': 0.005}
    # layer4_clip: {'original_norm': 0.25, 'max_norm': 0.1}

Practical Examples
------------------

Example 1: Warm Start Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refine parameters from a previous fit:

.. code-block:: python

    from nlsq import curve_fit, HybridStreamingConfig
    import jax.numpy as jnp


    def xpcs_model(x, g2_inf, beta, tau):
        return g2_inf + beta * jnp.exp(-2 * x / tau)


    # Initial fit
    popt_initial, _ = curve_fit(xpcs_model, t, g2_data, p0=[1.0, 0.5, 100.0])

    # Refinement with new data - use strict defense
    config = HybridStreamingConfig.defense_strict()

    popt_refined, pcov = curve_fit(
        xpcs_model,
        t_new,
        g2_new_data,
        p0=popt_initial,  # Warm start
        method="hybrid_streaming",
        config=config,
    )

    # Layer 1 will likely skip Adam warmup since p0 is already good

Example 2: Multi-Scale Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit model with parameters spanning many orders of magnitude:

.. code-block:: python

    def multi_scale_model(x, amplitude, rate, offset):
        # amplitude ~ 1e6, rate ~ 1e-6, offset ~ 100
        return amplitude * jnp.exp(-rate * x) + offset


    # Use scientific default with balanced defense
    config = HybridStreamingConfig.scientific_default()

    popt, pcov = curve_fit(
        multi_scale_model,
        x,
        y,
        p0=[1e6, 1e-6, 100],
        bounds=([1e5, 1e-7, 0], [1e7, 1e-5, 1000]),
        method="hybrid_streaming",
        config=config,
    )

Example 3: Monitoring Production Workloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor defense layer activation in production:

.. code-block:: python

    from nlsq import get_defense_telemetry, reset_defense_telemetry
    import time

    # Reset at start of monitoring window
    reset_defense_telemetry()
    start_time = time.time()

    # Process batch of fits
    for sample_id, (x, y) in enumerate(streaming_data):
        try:
            popt, pcov = curve_fit(
                model,
                x,
                y,
                method="hybrid_streaming",
                config=HybridStreamingConfig.scientific_default(),
            )
            save_results(sample_id, popt, pcov)
        except Exception as e:
            log_error(sample_id, e)

    # Report telemetry every hour
    if time.time() - start_time > 3600:
        telemetry = get_defense_telemetry()

        # Export to Prometheus
        metrics = telemetry.export_metrics()
        push_to_prometheus(metrics)

        # Log summary
        logger.info(telemetry.get_summary())

        # Reset for next window
        reset_defense_telemetry()
        start_time = time.time()

Performance Impact
------------------

The 4-layer defense strategy adds minimal overhead:

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Layer
     - Overhead
     - Notes
   * - Layer 1
     - ~0.1 ms
     - One-time check at warmup start
   * - Layer 2
     - ~0.05 ms
     - One-time LR selection
   * - Layer 3
     - ~0.2 ms/iter
     - Loss comparison per iteration
   * - Layer 4
     - ~0.5 ms/iter
     - Norm computation and clipping

**Total overhead**: < 10% of warmup time for typical problems.

**Benefits far outweigh costs**:

- Prevents divergence from good initial guesses
- Eliminates need for manual LR tuning
- Improves stability for multi-scale problems
- Provides production telemetry

Migration Guide
---------------

From Pre-0.3.6 Code
~~~~~~~~~~~~~~~~~~~

**Old code** (pre-0.3.6):

.. code-block:: python

    from nlsq import curve_fit

    popt, pcov = curve_fit(model, x, y, p0=initial_guess, method="hybrid_streaming")

**New behavior** (0.3.6+):

Defense layers are **enabled by default**. No code changes required.

**If you experience issues**:

.. code-block:: python

    from nlsq import HybridStreamingConfig

    # Option 1: Disable all defense layers
    config = HybridStreamingConfig.defense_disabled()
    popt, pcov = curve_fit(model, x, y, method="hybrid_streaming", config=config)

    # Option 2: Use relaxed defense
    config = HybridStreamingConfig.defense_relaxed()
    popt, pcov = curve_fit(model, x, y, method="hybrid_streaming", config=config)

Tuning Defense Sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**If warmup is skipped too often** (Layer 1 over-triggers):

.. code-block:: python

    config = HybridStreamingConfig(
        enable_warm_start_detection=True,
        warm_start_threshold=0.001,  # Lower threshold (0.1% instead of 1%)
    )

**If learning rate is too conservative** (Layer 2):

.. code-block:: python

    config = HybridStreamingConfig(
        enable_adaptive_warmup_lr=True,
        warmup_lr_refinement=1e-5,  # Increase from 1e-6
        warmup_lr_careful=1e-4,  # Increase from 1e-5
    )

**If cost guard aborts too early** (Layer 3):

.. code-block:: python

    config = HybridStreamingConfig(
        enable_cost_guard=True,
        cost_increase_tolerance=0.2,  # Allow 20% increase instead of 5%
    )

**If step clipping is too restrictive** (Layer 4):

.. code-block:: python

    config = HybridStreamingConfig(
        enable_step_clipping=True,
        max_warmup_step_size=0.5,  # Increase from 0.1
    )

Troubleshooting
---------------

Problem: Warmup Always Skipped
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Layer 1 triggers on every fit, Adam warmup never runs.

**Diagnosis**:

.. code-block:: python

    telemetry = get_defense_telemetry()
    rates = telemetry.get_trigger_rates()
    print(f"Warm start rate: {rates['layer1_warm_start_rate']:.1f}%")
    # Output: Warm start rate: 100.0%

**Solutions**:

1. **Lower the threshold**:

   .. code-block:: python

       config = HybridStreamingConfig(warm_start_threshold=0.001)

2. **Disable Layer 1**:

   .. code-block:: python

       config = HybridStreamingConfig(enable_warm_start_detection=False)

3. **Use relaxed preset**:

   .. code-block:: python

       config = HybridStreamingConfig.defense_relaxed()

Problem: Convergence Too Slow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Optimization takes many more iterations than expected.

**Diagnosis**:

Check if refinement LR mode is being used when exploration would be better:

.. code-block:: python

    telemetry = get_defense_telemetry()
    rates = telemetry.get_trigger_rates()
    print(f"Refinement: {rates['layer2_refinement_rate']:.1f}%")
    print(f"Exploration: {rates['layer2_exploration_rate']:.1f}%")

**Solutions**:

1. **Increase refinement/careful LRs**:

   .. code-block:: python

       config = HybridStreamingConfig(
           warmup_lr_refinement=1e-5,
           warmup_lr_careful=1e-4,
       )

2. **Disable adaptive LR**:

   .. code-block:: python

       config = HybridStreamingConfig(
           enable_adaptive_warmup_lr=False,
           warmup_learning_rate=0.01,  # Fixed LR
       )

Problem: Cost Guard Aborts Prematurely
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Layer 3 triggers early, preventing warmup completion.

**Diagnosis**:

.. code-block:: python

    rates = telemetry.get_trigger_rates()
    if rates["layer3_cost_guard_rate"] > 20:
        print("Cost guard triggering frequently")

**Solutions**:

1. **Increase tolerance**:

   .. code-block:: python

       config = HybridStreamingConfig(cost_increase_tolerance=0.2)

2. **Disable cost guard**:

   .. code-block:: python

       config = HybridStreamingConfig(enable_cost_guard=False)

3. **Check if initial guess is poor** - cost guard may be protecting you from divergence.

See Also
--------

- :doc:`../api/nlsq.adaptive_hybrid_streaming` : Main optimizer implementation
- :doc:`../api/nlsq.hybrid_streaming_config` : Full configuration reference
- :doc:`practical_workflows` : Real-world usage patterns
- :doc:`troubleshooting` : General troubleshooting guide
