nlsq.fallback module
=====================

.. currentmodule:: nlsq.stability.fallback

.. automodule:: nlsq.stability.fallback
   :noindex:

Overview
--------

The ``nlsq.fallback`` module (located at ``nlsq.stability.fallback``) provides
automatic fallback strategies for recovering from failed optimizations. When
curve fitting fails, the module intelligently tries alternative approaches to
achieve convergence.

Key Features
------------

- **Automatic method selection** - try alternative optimization methods
- **Initial guess perturbation** - escape local minima with random perturbations
- **Tolerance adjustment** - relax tolerances for difficult problems
- **Parameter bound inference** - infer reasonable bounds from data
- **Robust loss functions** - use soft_l1 or huber for outlier resistance
- **Problem rescaling** - improve conditioning through rescaling

Classes
-------

Main Classes
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FallbackOrchestrator
   FallbackResult
   FallbackStrategy

Strategy Classes
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AlternativeMethodStrategy
   PerturbInitialGuessStrategy
   AdjustTolerancesStrategy
   AddParameterBoundsStrategy
   UseRobustLossStrategy
   RescaleProblemStrategy

Usage Examples
--------------

Basic Usage with FallbackOrchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq.stability.fallback import FallbackOrchestrator
    import jax.numpy as jnp


    def model(x, a, b):
        return a * jnp.exp(-b * x)


    # Create orchestrator with verbose output
    orchestrator = FallbackOrchestrator(verbose=True)

    # Fit with automatic fallback
    result = orchestrator.fit_with_fallback(
        model, xdata, ydata, p0=[1.0, 0.5], max_attempts=5
    )

    if result.success:
        print(f"Fitted parameters: {result.popt}")
        print(f"Strategy used: {result.fallback_strategy_used}")
    else:
        print(f"All strategies failed after {result.attempts} attempts")

Custom Strategy Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq.stability.fallback import (
        FallbackOrchestrator,
        PerturbInitialGuessStrategy,
        AdjustTolerancesStrategy,
    )

    # Create orchestrator with specific strategies
    orchestrator = FallbackOrchestrator(
        strategies=[
            PerturbInitialGuessStrategy(),
            AdjustTolerancesStrategy(),
        ],
        verbose=True,
    )

    result = orchestrator.fit_with_fallback(model, x, y, p0=[1.0, 0.5])

Available Strategies
--------------------

The following fallback strategies are available, ordered by default priority:

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Strategy
     - Priority
     - Description
   * - ``AlternativeMethodStrategy``
     - 10
     - Try alternative optimization methods
   * - ``PerturbInitialGuessStrategy``
     - 8
     - Perturb initial parameters with random noise
   * - ``AdjustTolerancesStrategy``
     - 6
     - Relax convergence tolerances
   * - ``AddParameterBoundsStrategy``
     - 4
     - Infer and add parameter bounds
   * - ``UseRobustLossStrategy``
     - 2
     - Switch to robust loss functions
   * - ``RescaleProblemStrategy``
     - 1
     - Rescale problem for better conditioning

See Also
--------

- :doc:`../howto/troubleshooting` : Troubleshooting guide
- :doc:`nlsq.stability` : Numerical stability module
- :doc:`nlsq.recovery` : Optimization recovery
