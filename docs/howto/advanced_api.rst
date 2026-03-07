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

Adaptive Hybrid Streaming
-------------------------

For huge datasets that require streaming Gauss-Newton updates:

.. code:: python

   from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

   config = HybridStreamingConfig(chunk_size=50000, gauss_newton_max_iterations=20)
   optimizer = AdaptiveHybridStreamingOptimizer(config)

   result = optimizer.fit((x, y), exponential, p0=[2, 0.5], verbose=1)
   popt_final = result["x"]
   pcov_final = result["pcov"]

Related documentation
---------------------

- :doc:`../reference/configuration` - Configuration reference
- :doc:`../api/index` - Complete API documentation
- :doc:`../api/nlsq.adaptive_hybrid_streaming` - Streaming optimizer API
