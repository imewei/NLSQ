nlsq.diagnostics module
========================

.. currentmodule:: nlsq.utils.diagnostics

.. automodule:: nlsq.utils.diagnostics
   :noindex:

Overview
--------

The ``nlsq.diagnostics`` module provides comprehensive optimization monitoring and diagnostic
reporting. It helps track convergence, detect problems, and analyze optimization performance.

Key Features
------------

- **Real-time convergence monitoring** with pattern detection
- **Automatic problem detection** (oscillation, stagnation, divergence)
- **Performance metrics** tracking (timing, memory, function calls)
- **Detailed reporting** with summary statistics
- **Convergence visualization** with matplotlib integration

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ConvergenceMonitor
   OptimizationDiagnostics

Functions
---------

.. autosummary::
   :toctree: generated/

   get_diagnostics
   reset_diagnostics

Usage Examples
--------------

Basic Diagnostics
~~~~~~~~~~~~~~~~~

Use the global diagnostics instance to track optimization:

.. code-block:: python

    from nlsq.diagnostics import get_diagnostics, reset_diagnostics
    import jax.numpy as jnp

    # Reset diagnostics for new optimization
    reset_diagnostics()
    diag = get_diagnostics()

    # Start tracking
    diag.start_optimization(p0, problem_name="My Optimization")

    # Record iterations (done automatically by curve_fit)
    # ... optimization happens ...

    # Get summary
    stats = diag.get_summary_statistics()
    print(f"Total iterations: {stats['total_iterations']}")
    print(f"Final cost: {stats['final_cost']:.6e}")

    # Generate report
    report = diag.generate_report()
    print(report)

Convergence Monitoring
~~~~~~~~~~~~~~~~~~~~~~

Monitor convergence patterns during optimization:

.. code-block:: python

    from nlsq.diagnostics import ConvergenceMonitor

    monitor = ConvergenceMonitor(window_size=10, sensitivity=1.0)

    # Update with each iteration
    for iteration in range(max_iter):
        # ... perform optimization step ...
        monitor.update(cost, params, gradient, step_size)

        # Check for problems
        is_oscillating, osc_score = monitor.detect_oscillation()
        if is_oscillating:
            print(f"Warning: Oscillation detected (score={osc_score:.2f})")

        is_stagnant, stag_score = monitor.detect_stagnation()
        if is_stagnant:
            print(f"Warning: Stagnation detected (score={stag_score:.2f})")

        is_diverging, div_score = monitor.detect_divergence()
        if is_diverging:
            print(f"Warning: Divergence detected (score={div_score:.2f})")

Custom Diagnostics
~~~~~~~~~~~~~~~~~~

Create custom diagnostics instance for detailed tracking:

.. code-block:: python

    from nlsq.diagnostics import OptimizationDiagnostics

    diag = OptimizationDiagnostics(enable_plotting=True)
    diag.start_optimization(p0, problem_name="Exponential Fit")

    # Record iteration data manually
    for i in range(n_iterations):
        diag.record_iteration(
            iteration=i,
            x=params,
            cost=cost_value,
            gradient=gradient,
            jacobian=jacobian,
            step_size=step_size,
        )

    # Get statistics
    stats = diag.get_summary_statistics()
    print(f"Convergence rate: {stats.get('convergence_rate', 'N/A')}")
    print(f"Peak memory: {stats['peak_memory_mb']:.1f} MB")

    # Generate visualizations
    diag.plot_convergence(save_path="convergence_plot.png")

Event Recording
~~~~~~~~~~~~~~~

Record special events during optimization:

.. code-block:: python

    from nlsq.diagnostics import get_diagnostics

    diag = get_diagnostics()

    # Record recovery events
    diag.record_event("recovery_attempt", {"strategy": "perturbation"})
    diag.record_event("recovery_success", {"strategy": "perturbation", "cost": 0.005})

    # Record failures
    diag.record_event("jacobian_error", {"iteration": 42, "error": "singular"})

    # Check warnings
    stats = diag.get_summary_statistics()
    if stats["warnings_issued"]:
        print("Warnings:", stats["warnings_issued"])

Convergence Rate Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze convergence rate to assess optimization progress:

.. code-block:: python

    from nlsq.diagnostics import ConvergenceMonitor

    monitor = ConvergenceMonitor()

    # Update during optimization
    for i in range(n_iterations):
        monitor.update(costs[i], params[i])

    # Get convergence rate
    rate = monitor.get_convergence_rate()
    if rate is not None:
        if rate > 0:
            print(f"Converging with rate: {rate:.4f}")
        else:
            print(f"Diverging with rate: {rate:.4f}")

Performance Analysis
~~~~~~~~~~~~~~~~~~~~

Analyze optimization performance and resource usage:

.. code-block:: python

    from nlsq.diagnostics import OptimizationDiagnostics

    diag = OptimizationDiagnostics()
    diag.start_optimization(p0)

    # ... run optimization ...

    stats = diag.get_summary_statistics()

    print(f"Function evaluations: {stats['function_evaluations']}")
    print(f"Jacobian evaluations: {stats['jacobian_evaluations']}")
    print(f"Total time: {stats['total_time_seconds']:.2f}s")
    print(f"Time per iteration: {stats['time_per_iteration'] * 1000:.1f}ms")
    print(f"Memory increase: {stats['memory_increase_mb']:.1f} MB")

Diagnostic Metrics
------------------

The diagnostics system tracks these key metrics:

**Cost Reduction:**
- Initial and final cost values
- Absolute and relative cost reduction
- Min/max cost during optimization

**Convergence:**
- Convergence rate estimation
- Gradient norm progression
- Oscillation, stagnation, divergence detection

**Performance:**
- Total iterations
- Function and Jacobian evaluation counts
- Time per iteration
- Memory usage and peak memory

**Numerical Stability:**
- Jacobian condition numbers
- Gradient finiteness checks
- Parameter stability tracking

Pattern Detection
-----------------

The convergence monitor detects these patterns:

**Oscillation:**
Detected when parameters or cost values alternate up/down frequently.
Suggests step size is too large or problem is poorly scaled.

**Stagnation:**
Detected when cost barely changes over multiple iterations.
Suggests convergence has plateaued or is stuck in local minimum.

**Divergence:**
Detected when cost increases over time.
Suggests initial guess is poor or algorithm is unstable.

See Also
--------

- :doc:`nlsq.callbacks` : Callback functions for monitoring
- :doc:`nlsq.stability` : Numerical stability analysis
- :doc:`nlsq.recovery` : Optimization recovery strategies
