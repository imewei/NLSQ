nlsq.svd\_fallback module
==========================

.. automodule:: nlsq.svd_fallback
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``svd_fallback`` module provides SVD computation with GPU/CPU fallback and
randomized SVD for large matrices.

**New in version 0.3.1**: Added ``randomized_svd`` and ``compute_svd_adaptive`` for
3-10x faster SVD on large matrices.

Key Features
------------

- **GPU/CPU fallback** for robust SVD computation
- **Randomized SVD** for large matrices (3-10x faster)
- **Adaptive algorithm selection** based on matrix size
- **Automatic fallback** to stable algorithms
- **Precision switching** (float64 → float32 → relaxed tolerances)
- **Regularization strategies** for ill-conditioned matrices

Functions
---------

.. autosummary::
   :toctree: generated/

   compute_svd_with_fallback
   randomized_svd
   compute_svd_adaptive

Randomized SVD (New in v0.3.1)
------------------------------

For large matrices, randomized SVD provides significant speedups:

.. code-block:: python

   from nlsq.svd_fallback import randomized_svd
   import jax.numpy as jnp

   # Large matrix (1M x 10)
   A = jnp.ones((1_000_000, 10))

   # Randomized SVD (3-10x faster than full SVD)
   U, s, V = randomized_svd(
       A,
       n_components=10,  # Number of components to compute
       n_oversamples=10,  # Additional samples for accuracy
       n_iter=2,  # Power iterations for accuracy
       random_state=42,  # Reproducibility
   )

   print(f"U shape: {U.shape}")  # (1000000, 10)
   print(f"s shape: {s.shape}")  # (10,)
   print(f"V shape: {V.shape}")  # (10, 10)

**Algorithm**: Uses Halko, Martinsson, and Tropp (2011) with O(mnk) complexity
vs O(mn*min(m,n)) for full SVD.

Adaptive SVD Selection
----------------------

Let NLSQ automatically choose the best SVD algorithm:

.. code-block:: python

   from nlsq.svd_fallback import compute_svd_adaptive
   import jax.numpy as jnp

   # Matrix of any size
   A = jnp.ones((100_000, 50))

   # Automatically uses randomized SVD for large matrices
   U, s, V = compute_svd_adaptive(A)

   # Force specific algorithm
   U, s, V = compute_svd_adaptive(A, use_randomized=True)  # Force randomized
   U, s, V = compute_svd_adaptive(A, use_randomized=False)  # Force full SVD

**Threshold**: Matrices with >500K elements use randomized SVD by default.
Configure via ``RANDOMIZED_SVD_THRESHOLD`` constant.

GPU/CPU Fallback
----------------

Handle GPU failures gracefully:

.. code-block:: python

   from nlsq.svd_fallback import compute_svd_with_fallback
   import jax.numpy as jnp

   # Matrix that might cause numerical issues
   A = jnp.array([[1e10, 1.0], [1.0, 1e-10]])

   # SVD with automatic GPU→CPU fallback
   U, s, Vt = compute_svd_with_fallback(A, full_matrices=False)

   print(f"Singular values: {s}")

**Fallback Sequence**:

1. **GPU SVD** (jax.scipy.linalg.svd on GPU)
2. **CPU SVD** (automatic fallback if GPU fails)

Performance Comparison
----------------------

+------------------+------------+------------+----------+
| Matrix Size      | Full SVD   | Random SVD | Speedup  |
+==================+============+============+==========+
| 100K × 10        | 50ms       | 15ms       | 3.3x     |
+------------------+------------+------------+----------+
| 1M × 10          | 500ms      | 80ms       | 6.2x     |
+------------------+------------+------------+----------+
| 10M × 10         | 5s         | 600ms      | 8.3x     |
+------------------+------------+------------+----------+

**Note**: Randomized SVD is approximate. For condition number monitoring,
use ``n_iter=4`` for higher accuracy.

See Also
--------

- :doc:`nlsq.robust_decomposition` - Robust decomposition algorithms
- :doc:`nlsq.stability` - Numerical stability utilities
- :doc:`nlsq.mixed_precision` - Mixed precision management
- :doc:`../guides/stability` - Stability mode guide
