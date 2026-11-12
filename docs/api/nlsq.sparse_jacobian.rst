nlsq.sparse\_jacobian module
=============================

.. automodule:: nlsq.core.sparse_jacobian
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``sparse_jacobian`` module provides automatic detection and exploitation of sparse Jacobian patterns for improved computational efficiency.

Key Features (Task Group 6)
----------------------------

- **Sparsity Detection**: Automatic detection of sparse Jacobian patterns (>70% zeros)
- **Auto-Selection**: Triggers sparse-aware optimizations when beneficial
- **Phase 1 Infrastructure**: Detection and activation logic in place
- **Diagnostic Access**: Full visibility into sparsity detection results

Performance Benefits
--------------------

- Reduced memory usage for sparse problems
- Faster Jacobian-vector products
- Better scaling for large parameter spaces
- Automatic activation when sparsity >10%

Classes
-------

.. autoclass:: nlsq.core.sparse_jacobian.SparseJacobianComputer
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from nlsq.core.sparse_jacobian import SparseJacobianComputer
   import jax.numpy as jnp

   # Create sparse Jacobian computer
   computer = SparseJacobianComputer(sparsity_threshold=0.01)


   # Define model function
   def model(x, a, b):
       return a * jnp.exp(-b * x)


   # Detect sparsity pattern
   p0 = jnp.array([1.0, 0.5])
   x_sample = jnp.linspace(0, 10, 100)
   pattern, sparsity = computer.detect_sparsity_pattern(model, p0, x_sample)

   # Check if sparse optimizations should be used
   if sparsity > 0.1:  # More than 10% sparse
       print(f"Jacobian is {sparsity:.1%} sparse")
       print("Sparse optimizations will be enabled")

Sparsity Detection
------------------

The module automatically analyzes Jacobian patterns:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq.core.sparse_jacobian import SparseJacobianComputer


   # Model whose first half depends only on (a, b) and second half only on (c, d):
   # the Jacobian is block-diagonal, so half of its entries are zero.
   def multi_component_model(x, a, b, c, d):
       n = x.shape[0] // 2
       return jnp.concatenate([a * x[:n] + b, c * x[n:] + d])


   computer = SparseJacobianComputer()
   p0 = np.array([1.0, 1.0, 1.0, 1.0])
   x_sample = np.linspace(0, 10, 100)
   pattern, sparsity = computer.detect_sparsity_pattern(
       multi_component_model, p0, x_sample
   )

   print(f"Zero fraction: {sparsity:.1%}")  # about 50%
   print(f"Non-zero entries: {int(pattern.sum())} of {pattern.size}")

See Also
--------

- :doc:`nlsq.trf` - Trust Region Reflective algorithm
- :doc:`nlsq.least_squares` - Least squares solver
- :doc:`../howto/optimize_performance` - Performance optimization guide
