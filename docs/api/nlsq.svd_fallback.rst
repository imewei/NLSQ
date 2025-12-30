nlsq.svd\_fallback module
==========================

.. automodule:: nlsq.stability.svd_fallback
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``svd_fallback`` module provides full deterministic SVD computation with
GPU/CPU/NumPy fallback chain for robust numerical precision.

**Changed in version 0.3.5**: Randomized SVD has been completely removed.
All SVD operations now use full deterministic SVD for numerical precision
and reproducibility in optimization.

Key Features
------------

- **Full deterministic SVD** for numerical precision
- **GPU/CPU fallback** for robust SVD computation
- **NumPy fallback** as last resort if JAX fails
- **Reproducible results** across runs and platforms

Functions
---------

.. autosummary::
   :toctree: generated/

   compute_svd_with_fallback
   compute_svd_adaptive

compute_svd_with_fallback (Primary API)
---------------------------------------

The recommended function for SVD in NLSQ:

.. code-block:: python

   from nlsq.svd_fallback import compute_svd_with_fallback
   import jax.numpy as jnp

   # Matrix of any size
   A = jnp.ones((100_000, 50))

   # Full deterministic SVD with automatic fallback
   U, s, V = compute_svd_with_fallback(A, full_matrices=False)

   print(f"U shape: {U.shape}")  # (100000, 50)
   print(f"s shape: {s.shape}")  # (50,)
   print(f"V shape: {V.shape}")  # (50, 50)

**Fallback Sequence**:

1. **JAX GPU SVD** (jax.scipy.linalg.svd on GPU)
2. **JAX CPU SVD** (automatic fallback if GPU fails with cuSolver error)
3. **NumPy SVD** (last resort if JAX CPU also fails)

compute_svd_adaptive (Deprecated)
---------------------------------

.. deprecated:: 0.3.5
   This function now always uses full deterministic SVD.
   The ``use_randomized`` and ``n_components`` parameters are ignored.
   Use ``compute_svd_with_fallback`` directly instead.

.. code-block:: python

   from nlsq.svd_fallback import compute_svd_adaptive
   import jax.numpy as jnp

   # compute_svd_adaptive now always uses full SVD
   A = jnp.ones((100_000, 50))
   U, s, V = compute_svd_adaptive(A)  # Same as compute_svd_with_fallback

   # This will issue a DeprecationWarning:
   U, s, V = compute_svd_adaptive(A, use_randomized=True)

GPU/CPU Fallback
----------------

Handle GPU failures gracefully:

.. code-block:: python

   from nlsq.svd_fallback import compute_svd_with_fallback
   import jax.numpy as jnp

   # Matrix that might cause numerical issues
   A = jnp.array([[1e10, 1.0], [1.0, 1e-10]])

   # SVD with automatic GPU→CPU→NumPy fallback
   U, s, V = compute_svd_with_fallback(A, full_matrices=False)

   print(f"Singular values: {s}")

Why Full SVD Only?
------------------

Randomized SVD (available in v0.3.1-v0.3.4) was removed because it caused
optimization divergence in iterative least-squares solvers:

- **Approximation error accumulates** across trust-region iterations
- **Early termination** at worse local minima
- **3-25x worse fitting errors** in sensitive applications

Evidence from XPCS fitting (50K points, 13 params):

+-------------------+------------+--------------+------------+
| SVD Method        | D0 Error   | Alpha Error  | Iterations |
+===================+============+==============+============+
| Full SVD (v0.3.0) | 9.74%      | 0.59%        | 15         |
+-------------------+------------+--------------+------------+
| Randomized SVD    | 30.18%     | 14.66%       | 6          |
+-------------------+------------+--------------+------------+

See ``tests/test_svd_regression.py`` for detailed regression tests.

See Also
--------

- :doc:`nlsq.robust_decomposition` - Robust decomposition algorithms
- :doc:`nlsq.stability` - Numerical stability utilities
- :doc:`nlsq.mixed_precision` - Mixed precision management
- :doc:`../howto/troubleshooting` - Stability and troubleshooting guide
