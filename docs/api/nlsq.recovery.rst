nlsq.recovery module
=====================

.. currentmodule:: nlsq.stability.recovery

.. automodule:: nlsq.stability.recovery
   :noindex:

Overview
--------

The ``nlsq.stability.recovery`` module provides automatic recovery mechanisms for handling optimization
failures. When curve fitting encounters problems like convergence failures, numerical instabilities,
or ill-conditioned systems, the recovery system intelligently applies various strategies to
achieve successful optimization.

**New in version 0.1.1**: Complete recovery system with multi-strategy fallback chains.

Key Features
------------

- **Automatic parameter perturbation** to escape local minima
- **Algorithm switching** between TRF and LM (LM is skipped for finite bounds)
- **Regularization adjustment** for ill-conditioned problems
- **Problem reformulation** with data normalization and bound relaxation
- **Multi-start optimization** from different initial points
- **Recovery history tracking** for diagnostics and debugging
- **Configurable retry strategies** with customizable priorities

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   OptimizationRecovery

Recovery Strategies
-------------------

The ``OptimizationRecovery`` class implements five recovery strategies, tried in order:

1. **Parameter Perturbation**: Add Gaussian noise to escape local minima
2. **Algorithm Switching**: Try alternative optimization algorithms (TRF → LM → Dogbox)
3. **Regularization Adjustment**: Increase regularization for numerical stability
4. **Problem Reformulation**: Normalize data and relax bounds
5. **Multi-start**: Generate new random starting points

Usage Examples
--------------

The optimization function passed to ``recover_from_failure`` is called as
``optimization_func(**modified_state, **kwargs)``, where ``modified_state`` is a copy of
the state dict with the current strategy's changes applied (for example perturbed
``params``/``p0``, a different ``method``, new ``ftol``/``xtol``, ``regularization``,
or ``loss``). It must therefore accept these keys as keyword arguments, and return a
dict (or an object) with ``success``, ``cost`` and the parameters under ``x`` or
``params``. A result with non-finite parameters or cost, or outside the ``bounds``
given in the state, is rejected.

The examples below share this setup:

.. code-block:: python

    import jax.numpy as jnp
    import numpy as np

    from nlsq import curve_fit
    from nlsq.stability.recovery import OptimizationRecovery


    def model(x, a, b):
        return a * jnp.exp(-b * x)


    x = np.linspace(0, 10, 200)
    y = 2.5 * np.exp(-0.5 * x)


    def optimize(params, method="trf", bounds=(-np.inf, np.inf), **state):
        """Retry callback: ``state`` absorbs keys the fit does not need."""
        kwargs = {} if method == "trf" else {"method": method}
        popt, pcov = curve_fit(model, x, y, p0=params, bounds=bounds, **kwargs)
        cost = 0.5 * float(np.sum((y - np.asarray(model(x, *popt))) ** 2))
        return {"x": np.asarray(popt), "cost": cost, "success": True}

Basic Recovery
~~~~~~~~~~~~~~

Use recovery to automatically retry a failed optimization:

.. code-block:: python

    recovery = OptimizationRecovery(max_retries=3, enable_diagnostics=True)

    initial_params = np.array([1.0, 0.1])
    state = {"params": initial_params, "method": "trf"}

    success, result = recovery.recover_from_failure(
        failure_type="convergence",
        optimization_state=state,
        optimization_func=optimize,
    )

    if success:
        print("Recovery succeeded!", result["x"])
    else:
        print(result["error"])

Advanced Recovery with Custom State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide more state for better recovery. Extra keyword arguments to
``recover_from_failure`` are forwarded to the optimization function unchanged:

.. code-block:: python

    recovery = OptimizationRecovery(max_retries=5)

    optimization_state = {
        "params": initial_params,
        "p0": initial_params,
        "method": "trf",
        "bounds": ([0.0, 0.0], [10.0, 5.0]),  # solutions outside are rejected
        "iteration": 150,
        "cost": 1.5e-3,
        "has_outliers": True,  # enables the robust-loss progression
        "regularization": 1e-8,
    }

    success, result = recovery.recover_from_failure(
        failure_type="numerical",
        optimization_state=optimization_state,
        optimization_func=optimize,
    )

    if success:
        print(f"Recovered with cost: {result['cost']}")
        print(f"Recovery history: {recovery.recovery_history}")

Recovery from Different Failure Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``failure_type`` selects which strategies adapt. All strategies are tried in
order for every failure type; ``"numerical"`` and ``"ill_conditioned"`` additionally
increase ``regularization``, and ``"outliers"`` (or ``has_outliers=True`` in the
state) moves ``loss`` through a robust progression starting at ``"huber"``:

.. code-block:: python

    state = {"params": initial_params, "method": "trf"}

    # Convergence failure
    success, result = recovery.recover_from_failure("convergence", state, optimize)

    # Numerical issues - increase regularization
    success, result = recovery.recover_from_failure("numerical", state, optimize)

    # Ill-conditioned system
    success, result = recovery.recover_from_failure("ill_conditioned", state, optimize)

    # Outlier-contaminated data - switch to a robust loss
    success, result = recovery.recover_from_failure(
        "outliers", {**state, "has_outliers": True}, optimize
    )

Diagnostics and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each call appends an entry to ``recovery_history``. With ``enable_diagnostics=True``,
failed recoveries are also recorded in the diagnostics warnings:

.. code-block:: python

    recovery = OptimizationRecovery(max_retries=3, enable_diagnostics=True)
    success, result = recovery.recover_from_failure("convergence", state, optimize)

    for attempt in recovery.recovery_history:
        print(f"Failure type: {attempt['failure_type']}")
        print(f"Iteration: {attempt['iteration']}")
        print(f"Cost: {attempt['cost']}")

    if recovery.diagnostics is not None:
        print(recovery.diagnostics.warnings_issued)

Integration with curve_fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recovery can be integrated into curve fitting workflows:

.. code-block:: python

    def fit_with_recovery(f, xdata, ydata, p0, **kwargs):
        """Curve fit with automatic recovery."""
        recovery = OptimizationRecovery(max_retries=3)

        try:
            return curve_fit(f, xdata, ydata, p0=p0, **kwargs)
        except Exception as e:
            state = {"params": np.asarray(p0, dtype=float), "method": "trf"}

            def opt_func(params, **_):
                popt, pcov = curve_fit(f, xdata, ydata, p0=params, **kwargs)
                return {"x": popt, "pcov": pcov, "success": True}

            success, result = recovery.recover_from_failure("convergence", state, opt_func)

            if success:
                return result["x"], result["pcov"]
            raise RuntimeError("Recovery failed") from e


    popt, pcov = fit_with_recovery(model, x, y, p0=[1.0, 0.5])

Performance Considerations
--------------------------

Recovery adds robustness but increases computation time. Strategies are tried in
order for up to ``max_retries`` rounds, so the worst case is
``max_retries * 5`` re-optimizations before ``(False, {"error": ...})`` is returned.

**Best practices**:

- Set ``max_retries`` based on problem complexity (3-5 is typical)
- Provide good initial guesses to minimize recovery attempts
- Pass ``bounds`` in the state so that out-of-range recoveries are rejected
- Disable diagnostics (``enable_diagnostics=False``) if the bookkeeping is not needed

Recovery is most useful for:

- Poorly conditioned Jacobians
- Bad initial parameter guesses
- Noisy or outlier-contaminated data
- Multi-modal objective functions

See Also
--------

- :doc:`nlsq.fallback` : High-level fallback strategies
- :doc:`nlsq.stability` : Numerical stability analysis
- :doc:`nlsq.diagnostics` : Optimization diagnostics
- :doc:`../howto/troubleshooting` : Troubleshooting guide
