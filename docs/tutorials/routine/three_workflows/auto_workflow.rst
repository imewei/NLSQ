workflow="auto" - Local Optimization
=====================================

The ``auto`` workflow is NLSQ's default and recommended starting point for most
curve fitting tasks. It provides memory-aware local optimization using the
Trust Region Reflective (TRF) algorithm.

When to Use
-----------

Use ``auto`` workflow when:

- You have a reasonable initial guess for parameters
- Your model has a single clear minimum
- You want the fastest possible fit
- You're not sure which workflow to use (start here)

Basic Usage
-----------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Fit with auto workflow (default - no need to specify)
   popt, pcov = fit(model, xdata, ydata, p0=[2.0, 0.5, 0.0])

   # Explicit workflow specification (same result)
   popt, pcov = fit(model, xdata, ydata, p0=[2.0, 0.5, 0.0], workflow="auto")

With Optional Bounds
--------------------

Bounds constrain parameters to valid ranges but don't enable global search:

.. code-block:: python

   # Constrain a > 0, b > 0, -1 < c < 1
   popt, pcov = fit(
       model,
       xdata,
       ydata,
       p0=[2.0, 0.5, 0.0],
       workflow="auto",
       bounds=([0, 0, -1], [np.inf, np.inf, 1]),
   )

With bounds, the optimizer respects constraints but still performs local
optimization starting from ``p0``.

Tolerance Control
-----------------

Adjust convergence tolerances for speed vs precision trade-off:

.. code-block:: python

   # Fast fitting (looser tolerances)
   popt, pcov = fit(model, x, y, p0=[...], ftol=1e-6, xtol=1e-6, gtol=1e-6)

   # High precision (tighter tolerances)
   popt, pcov = fit(model, x, y, p0=[...], ftol=1e-10, xtol=1e-10, gtol=1e-10)

**Tolerance meanings:**

- ``ftol``: Relative change in cost function
- ``xtol``: Relative change in parameters
- ``gtol``: Gradient norm threshold

Memory Strategy Selection
-------------------------

The ``auto`` workflow automatically selects the optimal memory strategy:

**STANDARD (typical)**:

- Data and Jacobian fit in memory
- Fastest option, used for most datasets

**CHUNKED (medium datasets)**:

- Data fits but full Jacobian exceeds memory
- Jacobian computed in chunks
- Slightly slower, but handles larger problems

**STREAMING (large datasets)**:

- Data itself exceeds available memory
- Adaptive batch processing
- Handles datasets up to 100M+ points

You don't need to configure this - it's automatic.

Override Memory Detection
-------------------------

Force a specific memory limit if needed:

.. code-block:: python

   # Force conservative memory usage (4GB limit)
   popt, pcov = fit(model, x, y, p0=[...], workflow="auto", memory_limit_gb=4.0)

Complete Example
----------------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp
   import numpy as np


   # Model: Gaussian peak
   def gaussian(x, amplitude, center, width, offset):
       return amplitude * jnp.exp(-0.5 * ((x - center) / width) ** 2) + offset


   # Generate synthetic data
   np.random.seed(42)
   x = np.linspace(0, 10, 200)
   y_true = 5.0 * np.exp(-0.5 * ((x - 5.0) / 1.0) ** 2) + 0.5
   y = y_true + 0.2 * np.random.normal(size=len(x))

   # Fit with auto workflow
   p0 = [4.0, 5.0, 1.5, 0.0]  # Initial guess
   popt, pcov = fit(gaussian, x, y, p0=p0)

   # Results
   print("Fitted parameters:")
   print(f"  Amplitude: {popt[0]:.3f} (true: 5.0)")
   print(f"  Center:    {popt[1]:.3f} (true: 5.0)")
   print(f"  Width:     {popt[2]:.3f} (true: 1.0)")
   print(f"  Offset:    {popt[3]:.3f} (true: 0.5)")

   # Uncertainties
   perr = np.sqrt(np.diag(pcov))
   print("\nUncertainties:")
   for name, val, err in zip(["Amp", "Ctr", "Wid", "Off"], popt, perr):
       print(f"  {name}: +/- {err:.3f}")

When Auto Workflow May Fail
---------------------------

The ``auto`` workflow may not find the global optimum if:

1. **Poor initial guess**: Starting far from the solution
2. **Multiple local minima**: Model has several valid fits
3. **Highly nonlinear**: Model has complex parameter landscape

In these cases, switch to :doc:`auto_global_workflow`.

Troubleshooting
---------------

**Fit doesn't converge:**

- Try a better initial guess (closer to expected values)
- Increase ``max_nfev`` (maximum function evaluations)
- Loosen tolerances initially

**Results seem wrong:**

- Plot the fitted curve against data
- Check if the model is appropriate
- Try bounds to constrain parameters

**Memory errors:**

- Reduce ``memory_limit_gb`` to force chunked/streaming
- Check available system memory

Next Steps
----------

- :doc:`auto_global_workflow` - For global optimization
- :doc:`../data_handling/bounds` - More on parameter bounds
- :doc:`../troubleshooting/common_issues` - Debugging tips
