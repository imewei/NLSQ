nlsq.mixed\_precision module
==============================

.. currentmodule:: nlsq.precision.mixed_precision

.. automodule:: nlsq.precision.mixed_precision
   :noindex:

Overview
--------

The ``nlsq.mixed_precision`` module provides automatic mixed precision management for
memory-constrained optimization. It dynamically switches between float32 and float64
precision based on convergence indicators, providing up to 50% memory savings while
maintaining numerical accuracy.

Key Features
------------

- **Automatic precision fallback** from float32 to float64 when convergence stalls
- **NaN/Inf detection** with three validation points (gradients, parameters, loss)
- **Memory efficiency** with 50% reduction in typical cases
- **Gradient explosion detection** with configurable thresholds
- **Detailed diagnostics** for debugging precision issues
- **Trust radius validation** to ensure optimization stability
- **Configurable fallback triggers** based on iteration count and convergence metrics
- **Production-ready error handling** with comprehensive state validation

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   MixedPrecisionManager

Usage Examples
--------------

Basic Mixed Precision Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable automatic precision management for memory-constrained environments:

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.config import configure_mixed_precision
    import jax.numpy as jnp
    import numpy as np

    # Enable mixed precision with default settings
    configure_mixed_precision(enable=True)


    # Define model function
    def exponential(x, a, b):
        return a * jnp.exp(-b * x)


    # Generate data
    x = np.linspace(0, 10, 10000)
    y = 2.5 * np.exp(-0.8 * x) + np.random.normal(0, 0.1, 10000)

    # Fit - starts in float32, upgrades to float64 if needed
    popt, pcov = curve_fit(exponential, x, y, p0=[2.0, 0.5])

    print(f"Fitted parameters: a={popt[0]:.4f}, b={popt[1]:.4f}")

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

Customize fallback behavior for specific use cases:

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.config import configure_mixed_precision
    import jax.numpy as jnp

    # Configure with custom thresholds
    configure_mixed_precision(
        enable=True,
        max_degradation_iterations=5,  # Fallback after 5 stalled iterations
        gradient_explosion_threshold=1e10,  # Detect gradient explosion
        verbose=True,  # Enable diagnostic messages
    )


    def gaussian(x, amplitude, mean, std):
        return amplitude * jnp.exp(-((x - mean) ** 2) / (2 * std**2))


    # Fit with custom configuration
    popt, pcov = curve_fit(
        gaussian,
        x_data,
        y_data,
        p0=[1.0, 5.0, 1.0],
    )

Accessing Diagnostics
~~~~~~~~~~~~~~~~~~~~~

Monitor precision fallback events and performance:

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.config import configure_mixed_precision
    import jax.numpy as jnp

    # Enable verbose mode for diagnostics
    configure_mixed_precision(
        enable=True,
        verbose=True,
    )

    # Perform fit
    popt, pcov = curve_fit(exponential, x, y, p0=[2.0, 0.5])

    # Diagnostics are logged automatically when verbose=True
    # Example output:
    # [Mixed Precision] Starting in float32
    # [Mixed Precision] No convergence after 5 iterations, falling back to float64
    # [Mixed Precision] Successfully converged in float64

Direct Manager Usage
~~~~~~~~~~~~~~~~~~~~

Use the manager directly for fine-grained control:

.. code-block:: python

    from nlsq.mixed_precision import (
        MixedPrecisionManager,
        MixedPrecisionConfig,
        ConvergenceMetrics,
        OptimizationState,
    )
    import jax.numpy as jnp

    # Create configuration
    config = MixedPrecisionConfig(
        max_degradation_iterations=3,
        gradient_explosion_threshold=1e8,
        enable_mixed_precision_fallback=True,
    )

    # Create manager
    manager = MixedPrecisionManager(config, verbose=False)

    # In optimization loop, report convergence metrics
    metrics = ConvergenceMetrics(
        iteration=6,
        residual_norm=0.5,
        gradient_norm=1e12,  # Large gradient
        parameter_change=0.01,
        cost=0.15,
        trust_radius=1.0,
        has_nan_inf=False,
    )
    manager.report_metrics(metrics)

    # Check if precision upgrade needed
    if manager.should_upgrade():
        print("Precision upgrade recommended")

        # Create optimization state for upgrade
        state = OptimizationState(
            x=jnp.array([1.0, 2.0]),
            f=jnp.array([0.1, 0.2]),
            J=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            g=jnp.array([0.5, 0.6]),
            cost=0.15,
            trust_radius=1.0,
            iteration=6,
            dtype=jnp.float32,
        )

        # Upgrade to float64
        upgraded_state = manager.upgrade_precision(state)
        print(f"Upgraded to dtype: {upgraded_state.dtype}")

Handling Gradient Explosion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect and recover from gradient explosion:

.. code-block:: python

    from nlsq.mixed_precision import (
        MixedPrecisionManager,
        MixedPrecisionConfig,
        ConvergenceMetrics,
        OptimizationState,
    )
    import jax.numpy as jnp

    # Configure with lower explosion threshold
    config = MixedPrecisionConfig(
        gradient_explosion_threshold=1e10,
        enable_mixed_precision_fallback=True,
    )
    manager = MixedPrecisionManager(config, verbose=True)

    # Report metrics with exploded gradients
    metrics = ConvergenceMetrics(
        iteration=2,
        residual_norm=1e5,
        gradient_norm=1e12,  # Exploded gradients
        parameter_change=0.001,
        cost=float("inf"),
        trust_radius=0.1,
        has_nan_inf=True,
    )
    manager.report_metrics(metrics)

    # Check if upgrade needed (will detect gradient explosion)
    if manager.should_upgrade():
        print("Gradient explosion detected, upgrading to float64")

        # Create state for upgrade
        state = OptimizationState(
            x=jnp.array([1.0, 2.0]),
            f=jnp.array([1e5, 1e5]),
            J=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            g=jnp.array([1e12, 1e13]),
            cost=float("inf"),
            trust_radius=0.1,
            iteration=2,
            dtype=jnp.float32,
        )

        # Upgrade precision
        upgraded_state = manager.upgrade_precision(state)
        print(f"State upgraded to {upgraded_state.dtype}")

State Validation
~~~~~~~~~~~~~~~~

Validate optimization state for NaN/Inf values:

.. code-block:: python

    from nlsq.mixed_precision import MixedPrecisionManager
    import jax.numpy as jnp

    manager = MixedPrecisionManager()

    # Create optimization state
    state = {
        "x": jnp.array([1.0, 2.0, 3.0]),
        "f": jnp.array([0.1, 0.2, 0.3]),
        "J": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        "g": jnp.array([0.5, 0.6, 0.7]),
        "cost": 0.25,
        "trust_radius": 1.5,
    }

    # Validate state
    is_valid, error_msg = manager.validate_state(state)

    if is_valid:
        print("State is valid")
    else:
        print(f"Invalid state: {error_msg}")
        # Apply corrective action

Integration with TRF Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mixed precision integrates automatically with the Trust Region Reflective solver:

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.config import configure_mixed_precision
    import jax.numpy as jnp
    import numpy as np

    # Enable mixed precision
    configure_mixed_precision(enable=True, verbose=True)


    # Complex model that may benefit from mixed precision
    def complex_model(x, a, b, c, d):
        return a * jnp.exp(-b * x) + c * jnp.sin(d * x)


    x = np.linspace(0, 20, 50000)
    y_true = 2.0 * np.exp(-0.5 * x) + 0.3 * np.sin(1.5 * x)
    y = y_true + np.random.normal(0, 0.05, 50000)

    # TRF solver automatically uses mixed precision
    popt, pcov = curve_fit(
        complex_model,
        x,
        y,
        p0=[2.0, 0.5, 0.3, 1.5],
        method="trf",  # Uses mixed precision when enabled
    )

Batch Processing with Mixed Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient batch processing with shared JIT compilation:

.. code-block:: python

    from nlsq import CurveFit
    from nlsq.config import configure_mixed_precision
    import jax.numpy as jnp
    import numpy as np

    # Enable mixed precision for all fits
    configure_mixed_precision(enable=True)


    # Define model
    def exponential(x, a, b):
        return a * jnp.exp(-b * x)


    # Create reusable fitter
    fitter = CurveFit(exponential)

    # Fit multiple datasets (JIT cached, mixed precision enabled)
    datasets = [
        (x1, y1, [2.0, 0.5]),
        (x2, y2, [3.0, 0.8]),
        (x3, y3, [1.5, 0.3]),
    ]

    results = []
    for x, y, p0 in datasets:
        popt, pcov = fitter.fit(x, y, p0=p0)
        results.append((popt, pcov))

Memory Savings Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Measure memory reduction from mixed precision:

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.config import configure_mixed_precision
    import jax.numpy as jnp
    import numpy as np

    # Large dataset
    n_points = 1_000_000
    x = np.linspace(0, 10, n_points)
    y = 2.5 * np.exp(-0.8 * x) + np.random.normal(0, 0.1, n_points)

    # Without mixed precision (pure float64)
    configure_mixed_precision(enable=False)
    popt_64, _ = curve_fit(exponential, x, y, p0=[2.0, 0.5])

    # With mixed precision (starts float32, upgrades if needed)
    configure_mixed_precision(enable=True)
    popt_mixed, _ = curve_fit(exponential, x, y, p0=[2.0, 0.5])

    # Memory savings: typically 50% for float32 phase
    # Performance: similar if no fallback, ~10% overhead if fallback occurs

Configuration
-------------

Mixed precision is configured globally via ``nlsq.config.configure_mixed_precision()``:

.. code-block:: python

    from nlsq.config import configure_mixed_precision

    configure_mixed_precision(
        enable=True,  # Enable/disable mixed precision
        max_degradation_iterations=5,  # Fallback after N stalled iterations
        gradient_explosion_threshold=1e10,  # Gradient magnitude threshold
        verbose=False,  # Enable diagnostic logging
    )

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Configuration can also be set via environment variables:

.. code-block:: bash

    # Enable mixed precision
    export NLSQ_MIXED_PRECISION_ENABLE=true

    # Set fallback iterations
    export NLSQ_MIXED_PRECISION_MAX_DEGRADATION_ITERATIONS=3

    # Set gradient threshold
    export NLSQ_MIXED_PRECISION_GRADIENT_EXPLOSION_THRESHOLD=1e8

    # Enable verbose logging
    export NLSQ_MIXED_PRECISION_VERBOSE=true

Performance Characteristics
---------------------------

**Memory Savings**:

- **50% reduction** when optimization stays in float32
- Typical for well-conditioned problems with good initial guesses
- Most effective for large datasets (>10K points)

**Convergence Behavior**:

- **No fallback**: 0-5% faster than pure float64 (less memory traffic)
- **With fallback**: 10-15% overhead from precision conversion and restart
- Fallback typically occurs in <5% of cases with default settings

**Fallback Triggers**:

1. **Stalled convergence**: No cost improvement for `max_degradation_iterations`
2. **Gradient explosion**: Gradient magnitude exceeds threshold
3. **NaN/Inf detection**: Invalid values in state variables
4. **Trust radius collapse**: Trust radius becomes too small

**When to Use Mixed Precision**:

+=============================+=====================+======================+
| Scenario                    | Recommendation      | Expected Benefit     |
+=============================+=====================+======================+
| Memory-constrained systems  | Strongly recommended| 50% memory savings   |
+-----------------------------+---------------------+----------------------+
| Large datasets (>100K pts)  | Recommended         | 40-50% memory savings|
+-----------------------------+---------------------+----------------------+
| GPU acceleration            | Recommended         | Improved throughput  |
+-----------------------------+---------------------+----------------------+
| Small datasets (<1K points) | Optional            | Minimal benefit      |
+-----------------------------+---------------------+----------------------+
| High-precision requirements | Use with care       | May trigger fallback |
+-----------------------------+---------------------+----------------------+

Best Practices
--------------

1. **Enable for large datasets**: Maximum benefit with >10K data points
2. **Monitor diagnostics**: Use ``verbose=True`` during development
3. **Tune thresholds**: Adjust for problem-specific convergence behavior
4. **Test convergence**: Validate results match float64-only mode
5. **Profile memory usage**: Measure actual savings for your use case
6. **Handle fallback gracefully**: Expect fallback in 5-10% of challenging problems

Common Issues
-------------

**Issue: Frequent fallbacks**

If fallback occurs too often:

- Reduce ``max_degradation_iterations`` (try 3 instead of 5)
- Lower ``gradient_explosion_threshold`` (try 1e8 instead of 1e10)
- Improve initial guess ``p0`` quality
- Check problem conditioning

**Issue: Numerical accuracy concerns**

If results differ significantly from float64:

- Disable mixed precision for critical calculations
- Reduce ``max_degradation_iterations`` to fallback sooner
- Validate with ``verbose=True`` to see when fallback occurs

**Issue: Memory not reduced**

If memory savings are not observed:

- Check if fallback occurs immediately (use ``verbose=True``)
- Ensure dataset is large enough (>10K points)
- Verify JAX is using float32 correctly (check dtypes)

Validation and Debugging
-------------------------

**Check configuration**:

.. code-block:: python

    from nlsq.config import get_mixed_precision_config

    config = get_mixed_precision_config()
    print(f"Enabled: {config.enable_mixed_precision_fallback}")
    print(f"Max iterations: {config.max_degradation_iterations}")
    print(f"Gradient threshold: {config.gradient_explosion_threshold}")

**Enable verbose logging**:

.. code-block:: python

    import logging

    # Set NLSQ logger to DEBUG level
    logging.basicConfig(level=logging.DEBUG)
    nlsq_logger = logging.getLogger("nlsq")
    nlsq_logger.setLevel(logging.DEBUG)

    # Configure verbose mode
    configure_mixed_precision(enable=True, verbose=True)

**Manual state inspection**:

.. code-block:: python

    from nlsq.mixed_precision import (
        MixedPrecisionManager,
        MixedPrecisionConfig,
        OptimizationState,
    )
    import jax.numpy as jnp

    # Create manager with default config
    config = MixedPrecisionConfig()
    manager = MixedPrecisionManager(config)

    # Create example optimization state
    state = OptimizationState(
        x=jnp.array([1.0, 2.0]),
        f=jnp.array([0.1, 0.2]),
        J=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        g=jnp.array([0.5, 0.6]),
        cost=0.15,
        trust_radius=1.0,
        iteration=5,
        dtype=jnp.float32,
    )

    # Validate state
    is_valid, error = manager.validate_state(state)

    if not is_valid:
        print(f"State validation failed: {error}")

        # Check individual components
        print(f"Parameters valid: {not jnp.any(jnp.isnan(state.x))}")
        print(f"Gradient valid: {not jnp.any(jnp.isnan(state.g))}")
        print(f"Cost valid: {not jnp.isnan(state.cost)}")

Algorithm Details
-----------------

**Fallback Decision Logic**:

1. **Convergence monitoring**: Track cost improvement over iterations
2. **Gradient check**: Monitor gradient magnitude for explosion
3. **State validation**: Check all state variables for NaN/Inf
4. **Trust radius check**: Ensure trust radius remains positive

**State Validation Components**:

- ``_validate_parameters(x)``: Check parameter vector
- ``_validate_residuals_jacobian(f, J)``: Check residuals and Jacobian
- ``_validate_gradient_cost(g, cost)``: Check gradient and cost
- ``_validate_trust_radius(trust_radius)``: Check trust radius

**Precision Conversion**:

When fallback triggers, all state arrays are converted to float64:

- Parameters ``x``
- Residuals ``f``
- Jacobian ``J``
- Gradient ``g``
- Cost function value ``cost``
- Trust radius ``trust_radius``

See Also
--------

- :doc:`nlsq.config` : Configuration management and mixed precision setup
- :doc:`nlsq.trf` : Trust Region Reflective algorithm integration
- :doc:`nlsq.memory_manager` : Memory management for large datasets
- :doc:`../howto/optimize_performance` : Performance optimization guide
- :doc:`../howto/advanced_api` : Advanced usage patterns
