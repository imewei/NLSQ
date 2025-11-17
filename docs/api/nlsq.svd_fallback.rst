nlsq.svd\_fallback module
==========================

.. automodule:: nlsq.svd_fallback
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``svd_fallback`` module provides SVD fallback strategies for handling numerical instability.

Key Features
------------

- **Automatic fallback** to stable algorithms
- **Precision switching** (float64 → float32 → relaxed tolerances)
- **Regularization strategies** for ill-conditioned matrices
- **Detailed error diagnostics**

Functions
---------

.. autofunction:: nlsq.svd_fallback.svd_with_fallback
   :noindex:
.. autofunction:: nlsq.svd_fallback.check_svd_convergence
   :noindex:

Example Usage
-------------

.. code-block:: python

   from nlsq.svd_fallback import svd_with_fallback
   import jax.numpy as jnp

   # Matrix that might cause numerical issues
   A = jnp.array([[1e10, 1.0], [1.0, 1e-10]])

   # SVD with automatic fallback
   U, s, Vt, info = svd_with_fallback(A, return_info=True)

   print(f"Method used: {info['method']}")
   print(f"Fallback triggered: {info['fallback_triggered']}")
   print(f"Condition number: {info['condition_number']:.2e}")

Fallback Sequence
-----------------

The module tries the following sequence:

1. **Native JAX SVD** (jnp.linalg.svd)
2. **Regularized SVD** (A + λI)
3. **Mixed precision** (float32 with relaxed tolerances)
4. **Pseudo-inverse** (via Moore-Penrose)

.. code-block:: python

   # Custom fallback configuration
   U, s, Vt = svd_with_fallback(
       A, regularization=1e-8, fallback_dtype=jnp.float32, max_condition_number=1e12
   )

See Also
--------

- :doc:`nlsq.robust_decomposition` - Robust decomposition algorithms
- :doc:`nlsq.stability` - Numerical stability utilities
- :doc:`nlsq.mixed_precision` - Mixed precision management
