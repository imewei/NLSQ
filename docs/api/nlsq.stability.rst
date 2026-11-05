nlsq.stability module
======================

.. currentmodule:: nlsq.stability.guard

.. automodule:: nlsq.stability.guard
   :noindex:
   :exclude-members: eps, max_float, min_float, condition_threshold, regularization_factor

Overview
--------

The ``nlsq.stability`` module provides numerical stability analysis and automatic fixes
for common numerical issues in curve fitting. It helps prevent and resolve problems like
ill-conditioning, singular matrices, and overflow/underflow.

**New in version 0.1.1**: Complete numerical stability system with automatic detection and fixes.

Key Features
------------

- **Pre-flight problem diagnosis** (``check_problem_stability``) for ill-conditioned data, parameter scale mismatch and collinearity
- **Condition number estimation** for the independent variable data
- **Optional data rescaling** (``apply_automatic_fixes``) and NaN/Inf replacement in the data arrays
- **Safe math operations** and Jacobian checks inside ``NumericalStabilityGuard``
- **Fallback strategies** in ``nlsq.stability.fallback`` (see :doc:`nlsq.fallback`)

Classes
-------

.. autoclass:: NumericalStabilityGuard
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: eps, max_float, min_float, condition_threshold, regularization_factor

Functions
---------

.. autofunction:: check_problem_stability

.. autofunction:: apply_automatic_fixes

.. autofunction:: estimate_condition_number

.. autofunction:: detect_parameter_scale_mismatch

.. autofunction:: detect_collinearity

Usage Examples
--------------

Automatic Stability Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``curve_fit`` accepts ``stability="auto"`` (check and fix what is safe to fix),
``stability="check"`` (warn only) or ``stability=False`` (default, no checks). Ill-conditioned
data, large data ranges and parameter scale mismatches are reported as warnings but not
rescaled, because rescaling cannot be inverted soundly for an arbitrary model.
Use ``x_scale="jac"`` for solver-internal scaling. Non-finite data is controlled by
``nan_policy`` and ``check_finite``, not by ``stability``.

.. code-block:: python

    import jax.numpy as jnp
    import numpy as np

    from nlsq import curve_fit


    def model(x, a, b):
        return a * jnp.exp(-b * x)


    x = np.linspace(0, 10, 200)
    y = 2.5 * np.exp(-0.5 * x)

    popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5], stability="auto")

Manual Stability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Manually check problem stability before fitting. The report is a ``dict``:

.. code-block:: python

    from nlsq.stability import check_problem_stability

    report = check_problem_stability(x, y, p0=[1.0, 0.5])

    print(f"Severity: {report['severity']}")  # 'ok', 'warning' or 'critical'
    print(f"Condition number: {report['condition_number']:.2e}")
    print(f"Parameter scale ratio: {report['parameter_scale_ratio']}")
    print(f"Has collinearity: {report['has_collinearity']}")

    for issue_type, message, severity in report["issues"]:
        print(f"{severity}: {message}")
    print(f"Recommendations: {report['recommendations']}")

Applying Fixes
~~~~~~~~~~~~~~

``apply_automatic_fixes`` rescales the data when the report calls for it and
replaces non-finite values. The returned ``fix_info`` records the transformation so
that it can be undone:

.. code-block:: python

    from nlsq.stability import apply_automatic_fixes

    x_fixed, y_fixed, p0_fixed, fix_info = apply_automatic_fixes(
        x, y, p0=[1.0, 0.5], stability_report=report
    )

    print(fix_info["applied_fixes"])
    print(fix_info["x_scale"], fix_info["y_scale"])

    # Fits on rescaled data return parameters in the rescaled coordinate system
    popt, pcov = curve_fit(model, x_fixed, y_fixed, p0=p0_fixed)

Set ``rescale_data=False`` to keep physical units and ``replace_nonfinite=False`` to
leave NaN/Inf handling to ``curve_fit``.

Condition Number Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimate the condition of the data, and check a Jacobian with the guard:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    from nlsq.stability import NumericalStabilityGuard, estimate_condition_number

    cond = estimate_condition_number(x)
    print(f"Condition number of [1, x, x^2]: {cond:.2e}")

    guard = NumericalStabilityGuard()
    J = jax.jacfwd(lambda p: model(x, *p))(jnp.array([1.0, 0.5]))
    J_fixed, issues = guard.check_and_fix_jacobian(J)
    print(issues["condition_number"])

Scale and Collinearity Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq.stability import detect_collinearity, detect_parameter_scale_mismatch

    # Parameters spanning many orders of magnitude
    has_mismatch, ratio = detect_parameter_scale_mismatch(np.array([1e-3, 1e4, 1.0]))
    print(f"Mismatch: {has_mismatch}, ratio: {ratio:.2e}")

    # Highly correlated independent variables (short axis = variables)
    x1 = np.linspace(0, 10, 100)
    x2 = 2 * x1 + 0.01 * np.random.default_rng(0).normal(size=100)
    has_coll, pairs = detect_collinearity(np.vstack([x1, x2]))
    print(f"Collinear: {has_coll}, pairs: {pairs}")

Common Issues and Solutions
---------------------------

Ill-Conditioned Matrices
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Jacobian matrix has very large condition number (>1e10)

**Symptoms**:

- Unreliable parameter estimates
- Large parameter uncertainties
- Optimization fails to converge

**Solution**:

.. code-block:: python

    # Warn about conditioning problems
    popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5], stability="check")

    # Solver-internal scaling (no change of coordinates for the caller)
    popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5], x_scale="jac")

Overflow/Underflow
~~~~~~~~~~~~~~~~~~

**Problem**: Numerical overflow or underflow during evaluation

**Symptoms**:

- NaN or Inf values in results
- Optimization fails with numerical errors

**Solution**: Use the guard's safe operations inside the model, or handle
non-finite data before fitting:

.. code-block:: python

    guard = NumericalStabilityGuard()

    safe = guard.safe_exp(jnp.array([1.0, 1000.0]))  # no overflow to inf
    print(safe)

    # Drop non-finite data points instead of failing
    y_bad = y.copy()
    y_bad[5] = np.nan
    popt, pcov = curve_fit(model, x, y_bad, p0=[1.0, 0.5], nan_policy="omit")

Singular Matrices
~~~~~~~~~~~~~~~~~

**Problem**: Jacobian matrix is singular or near-singular

**Symptoms**:

- LinAlgError during optimization
- Infinite parameter uncertainties

**Solution**: Use ``NumericalStabilityGuard.check_and_fix_jacobian`` (which
regularizes ill-conditioned Jacobians) or ``OptimizationRecovery`` (see
:doc:`nlsq.recovery`), which increases regularization across retries.

Configuration
-------------

``NumericalStabilityGuard`` takes a single option; its thresholds are attributes:

.. code-block:: python

    guard = NumericalStabilityGuard(max_jacobian_elements_for_svd=1_000_000)

    print(guard.condition_threshold)  # 1e12: above this a matrix is ill-conditioned
    print(guard.regularization_factor)  # 1e-10: default regularization

For ``curve_fit``, the only switch is the ``stability`` argument described above.

See Also
--------

- :doc:`../howto/troubleshooting` : Troubleshooting guide
- :doc:`nlsq.validators` : Input validation
- :doc:`nlsq.fallback` : Fallback strategies
