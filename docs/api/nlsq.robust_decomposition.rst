nlsq.robust\_decomposition module
==================================

.. automodule:: nlsq.stability.robust_decomposition
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``robust_decomposition`` module provides robust matrix decomposition algorithms with numerical stability enhancements.

Key Features
------------

- **SVD with fallback strategies** for ill-conditioned matrices
- **QR decomposition** with pivoting
- **Cholesky decomposition** with regularization
- **Condition number monitoring**

Functions
---------

.. autofunction:: nlsq.robust_decomposition.robust_svd
   :noindex:
.. autofunction:: nlsq.robust_decomposition.robust_qr
   :noindex:
.. autofunction:: nlsq.robust_decomposition.robust_cholesky
   :noindex:

Example Usage
-------------

.. code-block:: python

   from nlsq.robust_decomposition import robust_svd
   import jax.numpy as jnp

   # Ill-conditioned matrix
   A = jnp.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]])

   # Robust SVD with fallback
   U, s, Vt = robust_svd(A, fallback=True)

   # Check condition number
   condition_number = s[0] / s[-1]
   print(f"Condition number: {condition_number:.2e}")

Fallback Strategies
-------------------

The module implements multiple fallback strategies for numerical stability:

1. **Standard decomposition** (JAX native)
2. **Regularized decomposition** (add small diagonal term)
3. **Mixed precision** (try float32 if float64 fails)
4. **Iterative refinement** (improve solution accuracy)

.. code-block:: python

   # Configure fallback behavior
   U, s, Vt = robust_svd(A, fallback=True, regularization=1e-10, max_attempts=3)

See Also
--------

- :doc:`nlsq.svd_fallback` - SVD fallback utilities
- :doc:`nlsq.stability` - Numerical stability checks
- :doc:`nlsq.trf` - Uses robust decomposition
