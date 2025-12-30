nlsq.error\_messages module
============================

.. automodule:: nlsq.utils.error_messages
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``error_messages`` module provides standardized error messages with helpful debugging information.

Key Features
------------

- **Standardized error messages** across the library
- **Helpful context** for debugging
- **Actionable suggestions** for fixing errors
- **Error classification** by category

Functions
---------

.. autofunction:: nlsq.error_messages.format_error_message
   :noindex:
.. autofunction:: nlsq.error_messages.get_suggestion
   :noindex:

Example Usage
-------------

.. code-block:: python

   from nlsq.error_messages import format_error_message

   try:
       # Some operation that might fail
       result = fit_function(data)
   except ValueError as e:
       # Get formatted error with helpful context
       error_msg = format_error_message(
           error_type="ValueError", message=str(e), context={"data_shape": data.shape}
       )
       raise ValueError(error_msg) from e

Error Categories
----------------

- **Input Validation Errors**: Invalid input data or parameters
- **Convergence Errors**: Optimization failed to converge
- **Numerical Errors**: Numerical stability issues
- **JAX Errors**: JAX-specific errors with workarounds

See Also
--------

- :doc:`nlsq.validators` - Input validation
- :doc:`../howto/troubleshooting` - Troubleshooting guide
