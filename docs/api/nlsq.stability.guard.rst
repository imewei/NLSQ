nlsq.stability.guard module
============================

.. automodule:: nlsq.stability.guard
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``guard`` module provides comprehensive numerical stability monitoring and
correction capabilities for the NLSQ package, ensuring robust optimization even
with ill-conditioned problems or extreme parameter values.

Stability Modes
---------------

The ``stability`` parameter in :func:`~nlsq.core.minpack.curve_fit` controls
numerical stability behavior:

- ``stability=False`` (default): No stability checks. Maximum performance.
- ``stability='check'``: Check for issues and warn, but don't modify data.
- ``stability='auto'``: Automatically detect and fix numerical issues.

Key Features
------------

- **Condition number estimation** via singular values (avoids full SVD overhead)
- **Automatic data rescaling** for ill-conditioned problems
- **Collinearity detection** among model parameters
- **Parameter scale mismatch detection**
- **Cholesky-with-fallback linear solver** for JIT-compatible robustness

Classes
-------

.. autoclass:: nlsq.stability.guard.NumericalStabilityGuard
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: nlsq.stability.guard.apply_automatic_fixes

.. autofunction:: nlsq.stability.guard.check_problem_stability

.. autofunction:: nlsq.stability.guard.detect_collinearity

.. autofunction:: nlsq.stability.guard.detect_parameter_scale_mismatch

.. autofunction:: nlsq.stability.guard.estimate_condition_number

.. autofunction:: nlsq.stability.guard.solve_with_cholesky_fallback

.. autodata:: nlsq.stability.guard.stability_guard
   :annotation: = NumericalStabilityGuard()

See Also
--------

- :mod:`nlsq.stability.svd_fallback` - GPU/CPU fallback SVD
- :doc:`nlsq.utils.diagnostics` - Optimization diagnostics and monitoring
