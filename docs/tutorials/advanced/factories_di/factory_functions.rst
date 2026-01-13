Factory Functions
=================

Factory functions enable runtime composition of optimization pipelines.

create_optimizer()
------------------

Creates a configured optimizer instance:

.. code-block:: python

   from nlsq.core.factories import create_optimizer

   # Basic optimizer
   optimizer = create_optimizer()
   popt, pcov = optimizer.fit(model, x, y)

   # With global optimization
   optimizer = create_optimizer(global_optimization=True, n_starts=20)

   # With full features
   optimizer = create_optimizer(
       global_optimization=True, diagnostics=True, recovery=True, n_starts=20
   )

**Parameters:**

.. code-block:: python

   create_optimizer(
       global_optimization=False,  # Enable multi-start/CMA-ES
       diagnostics=False,  # Collect convergence metrics
       recovery=False,  # Auto-recovery from issues
       n_starts=10,  # Starts for global optimization
       **kwargs  # Passed to underlying fit
   )

configure_curve_fit()
---------------------

Returns a configured fit function with preset defaults:

.. code-block:: python

   from nlsq.core.factories import configure_curve_fit

   # High-precision fit function
   high_precision_fit = configure_curve_fit(
       ftol=1e-12, xtol=1e-12, gtol=1e-12, enable_diagnostics=True
   )

   # Use like regular fit
   popt, pcov = high_precision_fit(model, x, y, p0=[...])

   # Fast fit function
   fast_fit = configure_curve_fit(ftol=1e-6, xtol=1e-6, gtol=1e-6, max_nfev=100)

**Parameters:**

.. code-block:: python

   configure_curve_fit(
       enable_diagnostics=False,  # Collect metrics
       enable_recovery=False,  # Auto-recovery
       enable_caching=True,  # JIT caching
       ftol=1e-8,  # Function tolerance
       xtol=1e-8,  # Parameter tolerance
       gtol=1e-8,  # Gradient tolerance
       **defaults  # Merged into every call
   )

Use Cases
---------

**Application-specific presets:**

.. code-block:: python

   # Spectroscopy fitting
   spectroscopy_fit = configure_curve_fit(
       ftol=1e-10, enable_diagnostics=True, stability="strict"
   )

   # Quick exploratory fits
   quick_fit = configure_curve_fit(ftol=1e-4, max_nfev=50)

**Batch processing:**

.. code-block:: python

   optimizer = create_optimizer(diagnostics=True)

   results = []
   for data_file in files:
       x, y = load(data_file)
       popt, pcov = optimizer.fit(model, x, y, p0=[...])
       results.append(popt)

**Testing configurations:**

.. code-block:: python

   # Production config
   prod_optimizer = create_optimizer(global_optimization=True, recovery=True)

   # Test config (faster, no recovery)
   test_optimizer = create_optimizer(global_optimization=False, recovery=False)

Complete Example
----------------

.. code-block:: python

   from nlsq.core.factories import create_optimizer, configure_curve_fit
   import jax.numpy as jnp
   import numpy as np


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Create test data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y = 2.5 * np.exp(-0.5 * x) + 0.3 + 0.1 * np.random.randn(100)

   # Configure different optimizers
   fast = configure_curve_fit(ftol=1e-6, xtol=1e-6, gtol=1e-6)
   precise = configure_curve_fit(ftol=1e-12, xtol=1e-12, gtol=1e-12)
   global_opt = create_optimizer(global_optimization=True, n_starts=10)

   # Compare
   print("Fast fit:")
   popt, _ = fast(model, x, y, p0=[2, 0.5, 0])
   print(f"  {popt}")

   print("Precise fit:")
   popt, _ = precise(model, x, y, p0=[2, 0.5, 0])
   print(f"  {popt}")

   print("Global fit:")
   popt, _ = global_opt.fit(model, x, y, p0=[2, 0.5, 0], bounds=([0, 0, -1], [10, 5, 1]))
   print(f"  {popt}")

Next Steps
----------

- :doc:`protocols` - Interface definitions
- :doc:`dependency_injection` - DI patterns
