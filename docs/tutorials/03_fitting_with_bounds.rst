Tutorial 3: Fitting with Bounds
================================

In this tutorial, you'll learn how to constrain parameter values using bounds,
which is essential for physical models where parameters must be positive,
within specific ranges, or otherwise constrained.

What You'll Learn
-----------------

- When and why to use bounds
- How to specify parameter bounds
- Handling common bound-related issues
- Best practices for bounded optimization

Prerequisites
-------------

- Completed :doc:`01_first_fit` and :doc:`02_understanding_results`

Why Use Bounds?
---------------

Bounds constrain parameters to physically meaningful ranges:

- **Positive amplitudes**: Intensity can't be negative
- **Rate constants**: Decay rates are typically positive
- **Fractions**: Must be between 0 and 1
- **Physical limits**: Temperature > 0 K, concentration ≥ 0

Bounds also help optimization by:

- Preventing numerical overflow/underflow
- Avoiding unphysical regions where the algorithm might get stuck
- Improving convergence for ill-posed problems

Basic Syntax
------------

Bounds are specified as a tuple of two arrays: ``(lower_bounds, upper_bounds)``

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   def exponential_decay(x, A, k):
       return A * jnp.exp(-k * x)


   # Data
   np.random.seed(42)
   x = np.linspace(0, 10, 50)
   y = 2.5 * np.exp(-0.5 * x) + 0.1 * np.random.normal(size=len(x))

   # Define bounds: A > 0, k > 0
   bounds = (
       [0, 0],  # Lower bounds: [A_min, k_min]
       [np.inf, np.inf],  # Upper bounds: [A_max, k_max]
   )

   popt, pcov = curve_fit(exponential_decay, x, y, bounds=bounds)
   print(f"A = {popt[0]:.4f}, k = {popt[1]:.4f}")

Using -inf and inf
------------------

Use ``-np.inf`` and ``np.inf`` for unbounded parameters:

.. code-block:: python

   # A: between 0 and 10
   # k: positive (no upper limit)
   bounds = ([0, 0], [10, np.inf])  # Lower bounds  # Upper bounds

Example: Gaussian Peak Fitting
------------------------------

Fitting a Gaussian peak with physically meaningful constraints:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   def gaussian(x, amplitude, center, sigma, baseline):
       """Gaussian peak with baseline offset."""
       return amplitude * jnp.exp(-((x - center) ** 2) / (2 * sigma**2)) + baseline


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = 5.0 * np.exp(-((x - 5.0) ** 2) / (2 * 1.0**2)) + 0.5
   y = y_true + 0.2 * np.random.normal(size=len(x))

   # Define bounds
   bounds = (
       [0, 0, 0.1, -np.inf],  # Lower: amp>0, center>0, sigma>0.1, baseline any
       [np.inf, 10, 5, np.inf],  # Upper: amp any, center<10, sigma<5, baseline any
   )

   # Initial guess
   p0 = [3.0, 5.0, 1.0, 0.0]

   # Fit with bounds
   popt, pcov = curve_fit(gaussian, x, y, p0=p0, bounds=bounds)

   print("Fitted parameters:")
   print(f"  Amplitude = {popt[0]:.3f}")
   print(f"  Center    = {popt[1]:.3f}")
   print(f"  Sigma     = {popt[2]:.3f}")
   print(f"  Baseline  = {popt[3]:.3f}")

Bounds vs Initial Guesses
-------------------------

Both bounds and initial guesses (``p0``) help the optimizer, but differently:

- **Bounds**: Hard constraints that parameters cannot violate
- **p0**: Starting point for optimization (can be outside final values)

.. code-block:: python

   # Both bounds and initial guess
   bounds = ([0, 0], [10, 5])
   p0 = [2.0, 1.0]  # Must be within bounds!

   popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds)

.. warning::

   If ``p0`` is outside the bounds, NLSQ will clip it to the nearest bound.

Common Issues and Solutions
---------------------------

Issue 1: Parameter Stuck at Bound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a fitted parameter equals its bound, the bound may be too restrictive:

.. code-block:: python

   # Problem: sigma stuck at lower bound
   bounds = ([0, 0, 1.0, -np.inf], [np.inf, 10, 5, np.inf])
   popt, pcov = curve_fit(gaussian, x, y, bounds=bounds)

   if popt[2] == 1.0:  # sigma at lower bound
       print("Warning: sigma at lower bound, consider relaxing")
       # Try again with looser bounds
       bounds = ([0, 0, 0.1, -np.inf], [np.inf, 10, 5, np.inf])

Issue 2: Bounds Too Tight
~~~~~~~~~~~~~~~~~~~~~~~~~

Overly restrictive bounds can prevent convergence:

.. code-block:: python

   # Too tight - may fail
   bounds = ([2.45, 0.49], [2.55, 0.51])

   # Better - give algorithm room to explore
   bounds = ([0, 0], [10, 5])

Issue 3: Infinite Covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a parameter is at its bound, its uncertainty may be infinite or undefined:

.. code-block:: python

   popt, pcov = curve_fit(model, x, y, bounds=bounds)

   # Check for infinite variances
   perr = np.sqrt(np.diag(pcov))
   for i, err in enumerate(perr):
       if not np.isfinite(err):
           print(f"Warning: Parameter {i} has undefined uncertainty")
           print(f"  Value: {popt[i]}, may be at bound")

Best Practices
--------------

1. **Start Without Bounds**

   First try fitting without bounds to see natural parameter ranges:

   .. code-block:: python

      popt_free, _ = curve_fit(model, x, y)
      print(f"Free fit parameters: {popt_free}")
      # Then set bounds based on this

2. **Use Physical Intuition**

   Set bounds based on what's physically possible:

   .. code-block:: python

      # Decay rate: positive, probably < 10 s^-1
      # Amplitude: positive, probably < 100
      bounds = ([0, 0], [100, 10])

3. **Provide Good Initial Guesses**

   Combine bounds with reasonable starting points:

   .. code-block:: python

      # Estimate from data
      A_guess = np.max(y) - np.min(y)
      x_peak = x[np.argmax(y)]

      p0 = [A_guess, x_peak, 1.0, np.min(y)]
      bounds = ([0, 0, 0.1, -10], [2 * A_guess, 10, 5, 10])

      popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds)

4. **Use Symmetric Log Bounds for Scale Parameters**

   For parameters spanning orders of magnitude:

   .. code-block:: python

      # Rate constant could be 0.001 to 1000
      bounds = ([0, 1e-4], [np.inf, 1e4])

Complete Example: Bi-exponential Decay
--------------------------------------

Fitting a sum of two exponentials (common in fluorescence lifetime):

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   def biexponential(x, A1, k1, A2, k2, offset):
       """Sum of two exponential decays."""
       return A1 * jnp.exp(-k1 * x) + A2 * jnp.exp(-k2 * x) + offset


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = 3.0 * np.exp(-0.8 * x) + 1.5 * np.exp(-0.2 * x) + 0.1
   y = y_true + 0.05 * np.random.normal(size=len(x))

   # Bounds: all amplitudes positive, rates positive,
   # and k1 > k2 (fast component first)
   bounds = (
       [0, 0.5, 0, 0, -1],  # Lower bounds
       [10, 10, 10, 0.5, 1],  # Upper bounds (k2 < 0.5 to ensure k1 > k2)
   )

   # Initial guess
   p0 = [2.0, 1.0, 1.0, 0.3, 0.0]

   # Fit
   popt, pcov = curve_fit(biexponential, x, y, p0=p0, bounds=bounds)
   perr = np.sqrt(np.diag(pcov))

   print("Bi-exponential Fit Results:")
   print("-" * 40)
   print(
       f"Fast component:  A1={popt[0]:.3f}±{perr[0]:.3f}, k1={popt[1]:.3f}±{perr[1]:.3f}"
   )
   print(
       f"Slow component:  A2={popt[2]:.3f}±{perr[2]:.3f}, k2={popt[3]:.3f}±{perr[3]:.3f}"
   )
   print(f"Offset:          {popt[4]:.3f}±{perr[4]:.3f}")

Key Takeaways
-------------

1. **Bounds** constrain parameters to valid ranges
2. Use ``(lower_bounds, upper_bounds)`` tuple syntax
3. Use ``np.inf`` and ``-np.inf`` for unbounded directions
4. Combine bounds with good initial guesses for best results
5. Check if fitted parameters are at bounds - may indicate too-tight constraints

Next Steps
----------

In :doc:`04_multiple_parameters`, you'll learn how to:

- Fit complex models with many parameters
- Handle parameter correlations
- Use multi-peak fitting

.. seealso::

   - :doc:`/explanation/trust_region` - How TRF handles bounds
   - :doc:`/howto/debug_bad_fits` - When fitting fails
