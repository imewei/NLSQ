nlsq.async\_logger module
==========================

.. automodule:: nlsq.utils.async_logger
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``async_logger`` module provides JAX-aware asynchronous logging infrastructure
to prevent GPU-CPU synchronization during optimization. This eliminates host-device
blocking that can degrade performance.

Key Features
------------

* **Zero-overhead logging**: Uses ``jax.debug.callback`` for non-blocking execution
* **JAX array detection**: Automatically identifies JAX arrays to prevent blocking
* **Verbosity control**: Configure logging frequency (0=off, 1=every 10th, 2=all)
* **Performance impact**: <5% overhead for async logging

Examples
--------

Basic usage with curve fitting::

    from nlsq import curve_fit
    import jax.numpy as jnp

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    # Async logging enabled with verbose=2
    popt, pcov = curve_fit(
        model, x, y,
        p0=[1.0, 0.5],
        verbose=2  # Logs all iterations asynchronously
    )

Integration with TRF optimizer::

    from nlsq.async_logger import log_iteration_async

    # Called automatically by TRF when verbose > 0
    # No manual integration needed

Type Detection::

    from nlsq.async_logger import is_jax_array
    import jax.numpy as jnp
    import numpy as np

    jax_arr = jnp.array([1, 2, 3])
    numpy_arr = np.array([1, 2, 3])

    print(is_jax_array(jax_arr))   # True
    print(is_jax_array(numpy_arr))  # False

Performance Characteristics
----------------------------

.. list-table:: Async Logging Performance
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - Blocking (Old)
     - Async (New)
   * - Iteration logging
     - ~10ms GPU sync
     - <0.5ms callback
   * - Large array logging
     - Transfer to CPU
     - Reference only
   * - Overall overhead
     - 15-25%
     - <5%

See Also
--------

* :mod:`nlsq.profiling` - Performance profiling infrastructure
* :mod:`nlsq.diagnostics` - Optimization diagnostics and monitoring
* :doc:`/howto/optimize_performance` - Complete performance optimization guide

Version History
---------------

.. versionadded:: 0.3.0-beta.3
   Initial async logging implementation with JAX integration
