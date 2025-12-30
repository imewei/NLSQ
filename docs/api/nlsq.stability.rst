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

- **Automatic problem diagnosis** for ill-conditioned systems
- **Intelligent fixes** for numerical stability issues
- **Condition number monitoring** for matrix health
- **Scale detection** and normalization
- **Overflow/underflow prevention**
- **Singular matrix handling**

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

Usage Examples
--------------

Automatic Stability Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the stability guard to automatically detect and fix issues:

.. code-block:: python

    from nlsq import curve_fit
    import jax.numpy as jnp


    def model(x, a, b):
        return a * jnp.exp(-b * x)


    # Enable automatic stability checking
    result = curve_fit(
        model, x, y, p0=[1.0, 0.5], stability="auto"  # Automatic detection and fixes
    )

Manual Stability Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Manually check problem stability before fitting:

.. code-block:: python

    from nlsq.stability import check_problem_stability

    # Analyze problem stability
    report = check_problem_stability(model, x, y, p0=[1.0, 0.5])

    print(f"Condition number: {report.condition_number:.2e}")
    print(f"Is ill-conditioned: {report.is_ill_conditioned}")
    print(f"Recommended fixes: {report.recommendations}")

    if report.is_ill_conditioned:
        # Apply recommended fixes
        result = curve_fit(
            model,
            x,
            y,
            p0=report.rescaled_p0,
            x_scale=report.x_scale,
            bounds=report.adjusted_bounds,
        )

Numerical Issue Fixes
~~~~~~~~~~~~~~~~~~~~~

Apply specific fixes for numerical problems:

.. code-block:: python

    from nlsq.stability import fix_numerical_issues

    # Detect and fix issues automatically
    fixed_x, fixed_y, fixed_p0 = fix_numerical_issues(
        x, y, p0, fix_overflow=True, fix_scaling=True, fix_singularities=True
    )

    # Use fixed inputs
    result = curve_fit(model, fixed_x, fixed_y, p0=fixed_p0)

Condition Number Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor matrix condition during optimization:

.. code-block:: python

    from nlsq.stability import compute_condition_number, NumericalStabilityGuard

    # Create stability guard
    guard = NumericalStabilityGuard(
        max_condition_number=1e10, warn_threshold=1e8, auto_fix=True
    )

    # Use guard during fitting
    result = curve_fit(model, x, y, p0=[1.0, 0.5], callback=guard)

    # Check stability report
    if guard.stability_report:
        print(f"Max condition number: {guard.stability_report.max_condition:.2e}")
        print(f"Applied fixes: {guard.stability_report.applied_fixes}")

Scaling Detection
~~~~~~~~~~~~~~~~~

Detect and fix scaling issues:

.. code-block:: python

    from nlsq.stability import detect_scaling_issues

    # Analyze scaling
    scaling_report = detect_scaling_issues(x, y, p0)

    if scaling_report.needs_rescaling:
        print(f"X scale factor: {scaling_report.x_scale:.2e}")
        print(f"Y scale factor: {scaling_report.y_scale:.2e}")

        # Apply scaling
        x_scaled = x * scaling_report.x_scale
        y_scaled = y * scaling_report.y_scale

        result = curve_fit(model, x_scaled, y_scaled, p0=p0)

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

    # Let NLSQ handle it automatically
    result = curve_fit(model, x, y, p0=p0, stability="auto")

    # Or manually rescale
    from nlsq.stability import check_problem_stability

    report = check_problem_stability(model, x, y, p0)
    result = curve_fit(model, x, y, p0=report.rescaled_p0)

Overflow/Underflow
~~~~~~~~~~~~~~~~~~

**Problem**: Numerical overflow or underflow during evaluation

**Symptoms**:
- NaN or Inf values in results
- Optimization fails with numerical errors

**Solution**:

.. code-block:: python

    from nlsq.stability import fix_numerical_issues

    # Automatically fix overflow/underflow
    fixed_x, fixed_y, fixed_p0 = fix_numerical_issues(
        x, y, p0, fix_overflow=True, log_transform_if_needed=True
    )

    result = curve_fit(model, fixed_x, fixed_y, p0=fixed_p0)

Singular Matrices
~~~~~~~~~~~~~~~~~

**Problem**: Jacobian matrix is singular or near-singular

**Symptoms**:
- LinAlgError during optimization
- Infinite parameter uncertainties

**Solution**:

.. code-block:: python

    # Use automatic regularization
    result = curve_fit(
        model,
        x,
        y,
        p0=p0,
        stability="auto",
        regularization=1e-8,  # Add small regularization
    )

Configuration
-------------

Configure stability checking behavior:

.. code-block:: python

    from nlsq.stability import StabilityConfig

    config = StabilityConfig(
        max_condition_number=1e10,
        warn_threshold=1e8,
        auto_fix=True,
        fix_overflow=True,
        fix_scaling=True,
        fix_singularities=True,
        verbose=True,
    )

    result = curve_fit(model, x, y, p0=p0, stability_config=config)

See Also
--------

- :doc:`../howto/troubleshooting` : Troubleshooting guide
- :doc:`nlsq.validators` : Input validation
- :doc:`nlsq.fallback` : Fallback strategies
