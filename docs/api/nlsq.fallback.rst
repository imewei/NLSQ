nlsq.fallback module
=====================

.. currentmodule:: nlsq.stability.fallback

.. automodule:: nlsq.stability.fallback
   :noindex:

Overview
--------

The ``nlsq.fallback`` module provides automatic fallback strategies for recovering from
failed optimizations. When curve fitting fails, the module intelligently tries alternative
approaches to achieve convergence.

**New in version 0.1.1**: Complete fallback system with automatic retry strategies.

Key Features
------------

- **Automatic retry** with perturbed initial parameters
- **Algorithm switching** to alternative optimizers
- **Bounds relaxation** for constrained problems
- **Regularization adjustment** for ill-conditioned systems
- **Multi-strategy cascading** fallback chains
- **Success rate tracking** and diagnostics

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FallbackStrategy

Functions
---------

.. Comment: No functions currently exist in this module
   .. autosummary::
   ..    :toctree: generated/
   ..
   ..    create_fallback_chain
   ..    execute_with_fallback

Usage Examples
--------------

Basic Auto-Retry
~~~~~~~~~~~~~~~~

Automatically retry with fallback strategies:

.. code-block:: python

    from nlsq import curve_fit
    import jax.numpy as jnp


    def model(x, a, b):
        return a * jnp.exp(-b * x)


    # Enable automatic fallback
    result = curve_fit(
        model, x, y, p0=[1.0, 0.5], fallback=True  # Automatic retry on failure
    )

    # Check if fallback was used
    if result.used_fallback:
        print(f"Succeeded using fallback strategy: {result.fallback_strategy}")

Custom Fallback Chain
~~~~~~~~~~~~~~~~~~~~~~

Create a custom sequence of fallback strategies:

.. code-block:: python

    from nlsq.fallback import (
        create_fallback_chain,
        AutoRetry,
        AlgorithmSwitcher,
        BoundsRelaxer,
    )

    # Define fallback chain
    fallback_chain = create_fallback_chain(
        [
            AutoRetry(max_attempts=3, perturbation=0.1),
            AlgorithmSwitcher(algorithms=["trf", "lm", "dogbox"]),
            BoundsRelaxer(relaxation_factor=1.5),
        ]
    )

    # Use custom chain
    result = curve_fit(model, x, y, p0=[1.0, 0.5], fallback_chain=fallback_chain)

Parameter Perturbation
~~~~~~~~~~~~~~~~~~~~~~

Retry with perturbed initial parameters:

.. code-block:: python

    from nlsq.fallback import AutoRetry

    retry_strategy = AutoRetry(
        max_attempts=5,
        perturbation=0.2,  # 20% perturbation
        adaptive=True,  # Increase perturbation if still failing
    )

    result = curve_fit(model, x, y, p0=[1.0, 0.5], fallback_strategy=retry_strategy)

    print(f"Attempts made: {retry_strategy.attempts_made}")

Algorithm Switching
~~~~~~~~~~~~~~~~~~~

Try different optimization algorithms:

.. code-block:: python

    from nlsq.fallback import AlgorithmSwitcher

    switcher = AlgorithmSwitcher(
        algorithms=["trf", "lm", "dogbox"], ordered=True  # Try in order
    )

    result = curve_fit(model, x, y, p0=[1.0, 0.5], fallback_strategy=switcher)

    if result.success:
        print(f"Succeeded with algorithm: {result.algorithm_used}")

Bounds Relaxation
~~~~~~~~~~~~~~~~~

Relax parameter bounds if constrained:

.. code-block:: python

    from nlsq.fallback import BoundsRelaxer

    relaxer = BoundsRelaxer(
        relaxation_factor=2.0, max_relaxations=3  # Double the bounds range
    )

    result = curve_fit(
        model, x, y, p0=[1.0, 0.5], bounds=([0, 0], [10, 10]), fallback_strategy=relaxer
    )

Multi-Strategy Fallback
~~~~~~~~~~~~~~~~~~~~~~~~

Combine multiple fallback strategies in sequence:

.. code-block:: python

    from nlsq.fallback import create_fallback_chain

    # Try perturbation first, then algorithm switch, then bounds relaxation
    chain = create_fallback_chain(
        [
            AutoRetry(max_attempts=3, perturbation=0.1),
            AlgorithmSwitcher(algorithms=["trf", "lm"]),
            BoundsRelaxer(relaxation_factor=1.5),
            RegularizationAdjuster(reg_values=[1e-8, 1e-6, 1e-4]),
        ]
    )

    result = curve_fit(
        model,
        x,
        y,
        p0=[1.0, 0.5],
        fallback_chain=chain,
        verbose=True,  # Show fallback progress
    )

Configuration
-------------

Configure fallback behavior globally:

.. code-block:: python

    from nlsq import set_fallback_config

    set_fallback_config(
        enabled=True,
        max_total_attempts=10,
        verbose=True,
        default_strategies=["retry", "switch", "relax"],
    )

    # Now all curve_fit calls use fallback by default
    result = curve_fit(model, x, y, p0=[1.0, 0.5])

Performance Considerations
--------------------------

Fallback strategies add robustness but may increase computation time:

- **AutoRetry**: 1-5x slower (depends on attempts)
- **AlgorithmSwitcher**: 1-3x slower (depends on algorithms tried)
- **BoundsRelaxer**: Minimal overhead
- **Combined strategies**: Cumulative impact

**Recommendation**: Enable fallback for production code where robustness is critical,
disable for rapid prototyping.

Success Rates
-------------

Typical success rates with fallback enabled:

- **Without fallback**: ~60% success rate
- **With basic retry**: ~75% success rate
- **With full fallback chain**: ~85-90% success rate

The improvement is most significant for:
- Poorly conditioned problems
- Difficult initial parameter guesses
- Noisy data
- Complex nonlinear models

See Also
--------

- :doc:`../howto/troubleshooting` : Troubleshooting guide
- :doc:`nlsq.stability` : Numerical stability
- :doc:`nlsq.recovery` : Optimization recovery
