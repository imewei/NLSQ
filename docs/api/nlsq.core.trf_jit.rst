nlsq.core.trf_jit
=================

JIT-compiled Trust Region Reflective helper functions.

.. versionadded:: 1.2.0
   Extracted from ``nlsq.core.trf`` for better code organization.

This module contains the JIT-compiled helper functions used by the Trust Region
Reflective optimizer. These functions are performance-critical and benefit from
JAX's just-in-time compilation.

Key Components
--------------

TrustRegionJITFunctions
~~~~~~~~~~~~~~~~~~~~~~~

A dataclass containing JIT-compiled functions for:

- Gradient computation with automatic differentiation
- SVD-based trust region subproblem solving
- Conjugate gradient (CG) solver for large problems
- Trust region step computation

Module Contents
---------------

.. automodule:: nlsq.core.trf_jit
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`nlsq.trf` - Main Trust Region Reflective optimizer
- :doc:`nlsq.core.profiler` - Performance profiling for TRF
