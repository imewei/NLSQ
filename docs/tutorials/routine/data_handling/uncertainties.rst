Uncertainties (Sigma)
=====================

Using measurement uncertainties improves fit quality and provides
meaningful parameter errors.

What is Sigma?
--------------

``sigma`` represents the uncertainty (standard deviation) of each data point.
Points with larger uncertainties have less influence on the fit.

.. code-block:: python

   from nlsq import fit

   # Each y value has an associated uncertainty
   popt, pcov = fit(model, x, y, p0=[...], sigma=uncertainties)

Uniform Uncertainties
---------------------

If all points have the same uncertainty:

.. code-block:: python

   import numpy as np

   # Same uncertainty for all points
   sigma = np.ones(len(y)) * 0.1  # 0.1 uncertainty
   popt, pcov = fit(model, x, y, p0=[...], sigma=sigma)

   # Or just a scalar (applied to all)
   popt, pcov = fit(model, x, y, p0=[...], sigma=0.1)

Variable Uncertainties
----------------------

Different uncertainties for different points:

.. code-block:: python

   # From measurement instrument
   sigma = measurement_errors

   # Poisson counting (proportional to sqrt(y))
   sigma = np.sqrt(np.abs(y))

   # Proportional to y (relative error)
   sigma = 0.05 * np.abs(y)  # 5% relative uncertainty

   popt, pcov = fit(model, x, y, p0=[...], sigma=sigma)

Absolute vs Relative Sigma
--------------------------

The ``absolute_sigma`` parameter affects how uncertainties are interpreted:

**absolute_sigma=True:**

- ``sigma`` represents actual measurement uncertainties
- Covariance matrix gives true parameter uncertainties
- Use when you know the measurement precision

.. code-block:: python

   # Measurement uncertainty is 0.1 units
   popt, pcov = fit(model, x, y, p0=[...], sigma=0.1, absolute_sigma=True)

**absolute_sigma=False (default):**

- ``sigma`` represents relative weights
- Covariance is scaled by residual variance
- Use when exact uncertainty is unknown

.. code-block:: python

   # Default behavior: sigma as weights
   popt, pcov = fit(model, x, y, p0=[...], sigma=weights)

Effect on Parameter Errors
--------------------------

With ``absolute_sigma=True``, parameter errors are calibrated:

.. code-block:: python

   import numpy as np

   # Known measurement uncertainty
   sigma = 0.1

   popt, pcov = fit(model, x, y, p0=[...], sigma=sigma, absolute_sigma=True)

   # Parameter errors are meaningful
   perr = np.sqrt(np.diag(pcov))
   print(f"Parameter errors: {perr}")

   # Chi-squared should be ~1 for good fit
   y_fit = model(x, *popt)
   chi_sq = np.sum(((y - y_fit) / sigma) ** 2)
   dof = len(y) - len(popt)
   reduced_chi_sq = chi_sq / dof
   print(f"Reduced chi-squared: {reduced_chi_sq:.2f}")

Using Weights Instead
---------------------

If you don't know uncertainties but want to weight points:

.. code-block:: python

   # Weight by inverse variance
   weights = 1.0 / np.var(y)

   # Or custom weights (higher = more influence)
   weights = np.where(x < 5, 1.0, 0.5)  # Trust early data more

   popt, pcov = fit(model, x, y, p0=[...], sigma=1 / weights)

Note: NLSQ uses sigma (not weights), so ``weight = 1/sigma``.

Estimating Uncertainties
------------------------

If uncertainties are unknown, estimate from data:

.. code-block:: python

   # From repeated measurements at each point
   y_mean = y_measurements.mean(axis=0)
   y_std = y_measurements.std(axis=0)
   sigma = y_std

   # From residuals of preliminary fit
   popt_init, _ = fit(model, x, y, p0=[...])
   residuals = y - model(x, *popt_init)
   sigma = np.abs(residuals) * 1.5  # Rough estimate

   # Refit with estimated uncertainties
   popt, pcov = fit(model, x, y, p0=popt_init, sigma=sigma)

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import fit


   # Model
   def exponential(x, A, k, c):
       return A * jnp.exp(-k * x) + c


   # Generate data with known uncertainties
   np.random.seed(42)
   x = np.linspace(0, 10, 50)
   y_true = 2.5 * np.exp(-0.5 * x) + 0.3

   # Measurement uncertainty
   sigma = 0.1

   # Add noise according to uncertainty
   y = y_true + sigma * np.random.randn(len(x))

   # Fit with uncertainty
   popt, pcov = fit(exponential, x, y, p0=[2, 0.5, 0], sigma=sigma, absolute_sigma=True)

   # Extract results
   A, k, c = popt
   perr = np.sqrt(np.diag(pcov))

   print("Results:")
   print(f"  A = {A:.4f} +/- {perr[0]:.4f} (true: 2.5)")
   print(f"  k = {k:.4f} +/- {perr[1]:.4f} (true: 0.5)")
   print(f"  c = {c:.4f} +/- {perr[2]:.4f} (true: 0.3)")

   # Goodness of fit
   y_fit = exponential(x, *popt)
   chi_sq = np.sum(((y - y_fit) / sigma) ** 2)
   dof = len(y) - len(popt)
   print(f"\nReduced chi-squared: {chi_sq/dof:.2f}")

Best Practices
--------------

1. **Always provide uncertainties** when you have them
2. **Use absolute_sigma=True** when uncertainties are calibrated
3. **Check reduced chi-squared**: should be ~1.0 for good fit
4. **Be consistent**: same units for y and sigma

Next Steps
----------

- :doc:`bounds` - Constrain parameters
- :doc:`large_datasets` - Handle large data
