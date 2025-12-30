nlsq.streaming.validators
=========================

Configuration validators for hybrid streaming optimization.

.. versionadded:: 1.2.0
   Extracted from ``nlsq.streaming.hybrid_config`` for maintainability.

This module provides validation functions for HybridStreamingConfig parameters.
Each validator ensures configuration values are within acceptable ranges and
provides informative error messages.

Validation Functions
--------------------

The module provides validators for:

- **Chunk size**: Ensures reasonable memory usage
- **Learning rates**: Validates warmup learning rate bounds
- **Tolerance values**: Checks convergence tolerances
- **Iteration limits**: Validates max iteration counts
- **Defense layer thresholds**: Ensures layer thresholds are sensible

Module Contents
---------------

.. automodule:: nlsq.streaming.validators
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from nlsq.streaming.validators import (
       validate_chunk_size,
       validate_learning_rate,
       validate_tolerance,
   )

   # Validate configuration values
   chunk_size = validate_chunk_size(10000)  # Returns validated value
   lr = validate_learning_rate(0.001, "warmup_lr")
   tol = validate_tolerance(1e-8, "ftol")

See Also
--------

- :doc:`nlsq.hybrid_streaming_config` - Configuration class
- :doc:`nlsq.adaptive_hybrid_streaming` - Main hybrid optimizer
