Advanced Customization Guide
============================

**When should I read this?** Read this guide if you need to extend NLSQ
beyond configuration files, implement custom logic, or integrate deeply
with the optimization pipeline.

Custom Callbacks
----------------

Create custom callbacks by defining a function with signature:

.. code:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   def exponential(x, a, b):
       return a * jnp.exp(-b * x)


   x = np.linspace(0, 5, 100)
   y = 2.5 * np.exp(-1.3 * x) + 0.1 * np.random.randn(100)


   def custom_callback(iteration, cost, params, info):
       """
       Parameters
       ----------
       iteration : int
           Current iteration number (0-indexed)
       cost : float
           Current cost function value
       params : ndarray
           Current parameter values
       info : dict
           Additional information (gradient norm, step norm, etc.)

       Returns
       -------
       stop : bool
           True to stop optimization early, False to continue
       """
       if iteration > 50 and cost < 0.01:
           print("Good enough! Stopping early.")
           return True

       if iteration % 10 == 0:
           print(f"Iter {iteration}: cost={cost:.6f}, params={params}")

       return False


   popt, pcov = curve_fit(
       exponential, x, y, p0=[2, 1], callback=custom_callback, max_nfev=100
   )

Mixed Precision Fallback
------------------------

NLSQ includes automatic mixed precision management that can reduce memory
usage by starting optimization in float32 and upgrading to float64 when
convergence stalls.

Enabling mixed precision
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from nlsq import curve_fit
   from nlsq.config import configure_mixed_precision
   import jax.numpy as jnp
   import numpy as np

   configure_mixed_precision(enable=True)


   def exponential(x, a, b):
       return a * jnp.exp(-b * x)


   x = np.linspace(0, 10, 100000)
   y = 2.5 * np.exp(-0.8 * x) + np.random.normal(0, 0.1, 100000)

   popt, pcov = curve_fit(exponential, x, y, p0=[2.0, 0.5])

Custom configuration
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   configure_mixed_precision(
       enable=True,
       max_degradation_iterations=5,
       gradient_explosion_threshold=1e10,
       verbose=True,
   )

Environment variables
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   export NLSQ_MIXED_PRECISION_ENABLE=true
   export NLSQ_MIXED_PRECISION_MAX_DEGRADATION_ITERATIONS=3
   export NLSQ_MIXED_PRECISION_GRADIENT_EXPLOSION_THRESHOLD=1e8
   export NLSQ_MIXED_PRECISION_VERBOSE=true

Diagnostic Monitoring
---------------------

Monitor optimization health and numerical stability.

.. code:: python

   from nlsq.diagnostics import DiagnosticMonitor

   monitor = DiagnosticMonitor(
       check_condition_number=True,
       check_gradient_norm=True,
       check_step_quality=True,
       log_level="INFO",
   )

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 0.5], diagnostics=monitor)

   print(monitor.summary())

Sparse Jacobian Optimization
----------------------------

For models with sparse Jacobian structure, provide a sparsity pattern for
significant speedups.

.. code:: python

   import scipy.sparse as sp


   def complex_model(x, *params): ...


   sparsity = sp.lil_matrix((len(x), len(p0)))
   sparsity[0:50, 0:2] = 1
   sparsity[50:100, 2:4] = 1

   popt, pcov = curve_fit(complex_model, x, y, p0=p0, jac_sparsity=sparsity)

Streaming Optimization
----------------------

For online learning or real-time fitting scenarios.

.. code:: python

   from nlsq.streaming.optimizer import OnlineOptimizer

   optimizer = OnlineOptimizer(
       model=exponential, p0=[2, 0.5], learning_rate=0.01, momentum=0.9
   )

   for x_batch, y_batch in data_stream:
       popt = optimizer.update(x_batch, y_batch)
       print(f"Current estimate: {popt}")

   popt_final = optimizer.get_parameters()
   pcov_final = optimizer.get_covariance()

Related documentation
---------------------

- :doc:`../reference/configuration` - Configuration reference
- :doc:`../api/index` - Complete API documentation
- :doc:`../api/nlsq.streaming_optimizer` - Streaming optimizer API
- :doc:`../api/nlsq.mixed_precision` - Mixed precision API
