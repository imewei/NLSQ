Parameter Bounds
================

Bounds constrain parameters to physically meaningful or valid ranges.

Basic Syntax
------------

Bounds are specified as two arrays: lower bounds and upper bounds.

.. code-block:: python

   from nlsq import fit
   import numpy as np

   # bounds = ([lower1, lower2, ...], [upper1, upper2, ...])
   bounds = ([0, 0, -1], [10, 5, 1])

   popt, pcov = fit(model, x, y, p0=[1, 0.5, 0], bounds=bounds)

For a 3-parameter model:

- Parameter 1: 0 <= p1 <= 10
- Parameter 2: 0 <= p2 <= 5
- Parameter 3: -1 <= p3 <= 1

Unbounded Parameters
--------------------

Use ``np.inf`` for unbounded parameters:

.. code-block:: python

   import numpy as np

   # Only constrain first parameter (must be positive)
   bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

   # Only upper bound on second parameter
   bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, 100, np.inf])

Common Bound Patterns
---------------------

**Positive parameters:**

.. code-block:: python

   # Amplitude, rate, offset must be positive
   bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

**Constrained angles:**

.. code-block:: python

   # Phase angle in [-pi, pi]
   bounds = ([-np.pi], [np.pi])

**Relative constraints:**

.. code-block:: python

   # center1 < center2 (enforce ordering)
   # Use separate fits or reparameterize the model

**Physical limits:**

.. code-block:: python

   # Concentration (0-100%), temperature > 0
   bounds = ([0, 0], [100, np.inf])

When to Use Bounds
------------------

**Use bounds when:**

- Parameters have physical constraints (e.g., positive amplitude)
- Using ``workflow='auto_global'`` (required for global search)
- Preventing the optimizer from exploring invalid regions
- Model is undefined outside certain ranges

**Don't use bounds when:**

- Parameters can take any value
- Bounds are unnecessarily tight (may prevent convergence)
- Using bounds to "fix" a bad fit (fix the model instead)

Bounds and Global Optimization
------------------------------

For ``workflow='auto_global'``, bounds define the search space:

.. code-block:: python

   # Global search requires bounds
   popt, pcov = fit(
       model, x, y, p0=[1, 0.5], workflow="auto_global", bounds=([0, 0], [10, 5])
   )

Wider bounds = larger search space = slower but more thorough.

Effect on Results
-----------------

Bounds affect optimization behavior:

1. **Parameters at bounds**: May indicate bounds too tight
2. **Large covariance**: May indicate constraint effects
3. **Slow convergence**: May indicate bounds issues

.. code-block:: python

   # Check if parameters are at bounds
   lower, upper = bounds
   at_lower = np.isclose(popt, lower, rtol=1e-3)
   at_upper = np.isclose(popt, upper, rtol=1e-3)

   if any(at_lower) or any(at_upper):
       print("Warning: Some parameters at bounds")

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import fit


   # Model: Gaussian peak
   def gaussian(x, amplitude, center, width, offset):
       return amplitude * jnp.exp(-0.5 * ((x - center) / width) ** 2) + offset


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = 3.0 * np.exp(-0.5 * ((x - 5) / 1.2) ** 2) + 0.5
   y = y_true + 0.2 * np.random.randn(len(x))

   # Physical constraints:
   # - amplitude > 0
   # - center within data range
   # - width > 0
   # - offset >= 0
   bounds = ([0, 0, 0.1, 0], [10, 10, 5, 5])  # Lower bounds  # Upper bounds

   # Initial guess
   p0 = [2.5, 5, 1, 0.5]

   # Fit with bounds
   popt, pcov = fit(gaussian, x, y, p0=p0, bounds=bounds)

   # Results
   names = ["amplitude", "center", "width", "offset"]
   for name, val, lo, hi in zip(names, popt, bounds[0], bounds[1]):
       at_bound = (
           "AT LOWER" if np.isclose(val, lo) else "AT UPPER" if np.isclose(val, hi) else ""
       )
       print(f"{name:10s}: {val:6.3f}  [{lo}, {hi}]  {at_bound}")

Troubleshooting
---------------

**Parameter stuck at bound:**

- Bounds may be too tight
- Initial guess may be at bound
- Model may not fit the data

**Slow convergence with bounds:**

- Start with wider bounds
- Use better initial guess
- Check if model is appropriate

**Fit fails with bounds:**

- Ensure bounds are valid (lower < upper)
- Ensure p0 is within bounds
- Try without bounds first to verify model

Next Steps
----------

- :doc:`large_datasets` - Handle millions of points
- :doc:`../three_workflows/auto_global_workflow` - Global optimization
