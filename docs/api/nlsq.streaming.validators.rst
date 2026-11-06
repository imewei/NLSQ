nlsq.streaming.validators
=========================

Configuration validators for hybrid streaming optimization.

Extracted from ``nlsq.streaming.hybrid_config`` for maintainability.

This module provides validation functions for HybridStreamingConfig parameters.
Each validator ensures configuration values are within acceptable ranges and
provides informative error messages.

Validation Functions
--------------------

The module provides:

- **Scalar checks**: ``validate_positive``, ``validate_positive_int``, ``validate_non_negative``, ``validate_range``, ``validate_less_than_or_equal``, ``validate_enum_value``
- **Strategy checks**: normalization, loop, L-BFGS line search and multi-start sampler names
- **Per-phase config checks**: warmup, L-BFGS, Gauss-Newton, CG, group variance, error handling, learning-rate schedule, gradient clip, residual weighting and multi-start
- **Defense layer thresholds**: ``validate_defense_layer_config``
- **Streaming settings**: ``validate_streaming_config`` (chunk size, checkpoint and callback frequency)

All validators return ``None`` and raise ``ConfigValidationError`` when a value is invalid.

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
       ConfigValidationError,
       validate_non_negative,
       validate_positive,
       validate_positive_int,
       validate_range,
   )

   # Validators return None and raise ConfigValidationError on invalid values
   validate_positive_int(10000, "chunk_size")
   validate_positive(0.001, "warmup_learning_rate")
   validate_non_negative(0.0, "regularization")
   validate_range(0.5, 0.0, 1.0, "momentum", inclusive_max=False)

   try:
       validate_positive(-1.0, "ftol")
   except ConfigValidationError as err:
       print(err)  # ftol must be positive and finite

See Also
--------

- :doc:`nlsq.hybrid_streaming_config` - Configuration class
- :doc:`nlsq.adaptive_hybrid_streaming` - Main hybrid optimizer
