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

.. autoclass:: nlsq.sparse_jacobian.SparseJacobianComputer
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

   from nlsq.sparse_jacobian import SparseJacobianComputer
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

   # Analyze sparsity for different model functions
   def multi_component_model(x, *params):
       # Model with multiple independent components
       # Often results in block-diagonal sparse Jacobian
       pass


   computer = SparseJacobianComputer()
   pattern, sparsity = computer.detect_sparsity_pattern(
       multi_component_model, p0, x_sample
   )

   # Get detailed diagnostics
   diagnostics = computer.get_diagnostics()
   print(f"Non-zero elements: {diagnostics['nnz']}")
   print(f"Sparsity pattern: {diagnostics['pattern']}")

See Also
--------

- :doc:`nlsq.trf` - Trust Region Reflective algorithm
- :doc:`nlsq.least_squares` - Least squares solver
- :doc:`../howto/optimize_performance` - Performance optimization guide
