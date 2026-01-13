Understanding Results
=====================

This tutorial explains how to interpret the output from ``fit()`` and calculate
parameter uncertainties.

The Two Return Values
---------------------

``fit()`` returns two values:

.. code-block:: python

   popt, pcov = fit(model, x, y, p0=[...])

- **popt**: Optimal parameters (1D array)
- **pcov**: Covariance matrix (2D array)

Optimal Parameters (popt)
-------------------------

``popt`` contains the fitted parameter values in the same order as your model:

.. code-block:: python

   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   popt, pcov = fit(model, x, y, p0=[1, 0.5, 0])

   # popt[0] = a (amplitude)
   # popt[1] = b (decay rate)
   # popt[2] = c (offset)

   a_fit, b_fit, c_fit = popt

Covariance Matrix (pcov)
------------------------

``pcov`` is a square matrix where:

- **Diagonal elements**: Variance of each parameter
- **Off-diagonal elements**: Covariance between parameters

.. code-block:: text

   pcov = [[var(a),    cov(a,b), cov(a,c)],
           [cov(b,a),  var(b),   cov(b,c)],
           [cov(c,a),  cov(c,b), var(c)  ]]

Calculating Parameter Uncertainties
-----------------------------------

The standard error (1-sigma uncertainty) of each parameter is the square root
of the diagonal:

.. code-block:: python

   import numpy as np

   # Standard errors (1-sigma)
   perr = np.sqrt(np.diag(pcov))

   a_fit, b_fit, c_fit = popt
   a_err, b_err, c_err = perr

   print(f"a = {a_fit:.4f} +/- {a_err:.4f}")
   print(f"b = {b_fit:.4f} +/- {b_err:.4f}")
   print(f"c = {c_fit:.4f} +/- {c_err:.4f}")

Confidence Intervals
--------------------

For different confidence levels, multiply the standard error:

- **68.3% (1-sigma)**: ``perr``
- **95.4% (2-sigma)**: ``2 * perr``
- **99.7% (3-sigma)**: ``3 * perr``

.. code-block:: python

   # 95% confidence interval
   ci_95 = 1.96 * perr

   print(f"a = {a_fit:.4f} +/- {ci_95[0]:.4f} (95% CI)")

Parameter Correlations
----------------------

The correlation matrix shows how parameters are related:

.. code-block:: python

   # Correlation matrix (normalized covariance)
   d = np.sqrt(np.diag(pcov))
   corr = pcov / np.outer(d, d)

   print("Correlation matrix:")
   print(corr)

- **+1**: Parameters are perfectly correlated
- **-1**: Parameters are perfectly anti-correlated
- **0**: Parameters are independent

High correlations (absolute value > 0.9) indicate the parameters may be difficult
to determine independently.

Goodness of Fit
---------------

Compute residuals and chi-squared:

.. code-block:: python

   # Compute residuals
   y_fit = model(x, *popt)
   residuals = y - y_fit

   # Sum of squared residuals
   ss_res = np.sum(residuals**2)

   # Chi-squared (if sigma is provided)
   if sigma is not None:
       chi_sq = np.sum((residuals / sigma) ** 2)
       dof = len(y) - len(popt)  # degrees of freedom
       reduced_chi_sq = chi_sq / dof
       print(f"Reduced chi-squared: {reduced_chi_sq:.3f}")
       # Good fit: reduced chi-sq ~ 1.0

   # R-squared
   ss_tot = np.sum((y - np.mean(y)) ** 2)
   r_squared = 1 - (ss_res / ss_tot)
   print(f"R-squared: {r_squared:.4f}")

Complete Example
----------------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp
   import numpy as np


   # Model
   def exponential(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = 2.5 * np.exp(-0.5 * x) + 0.3
   sigma = 0.1
   y = y_true + sigma * np.random.normal(size=len(x))

   # Fit
   popt, pcov = fit(exponential, x, y, p0=[2, 0.5, 0], sigma=sigma)

   # Extract results
   a, b, c = popt
   perr = np.sqrt(np.diag(pcov))

   print("Fitted parameters:")
   print(f"  a = {a:.4f} +/- {perr[0]:.4f}")
   print(f"  b = {b:.4f} +/- {perr[1]:.4f}")
   print(f"  c = {c:.4f} +/- {perr[2]:.4f}")

   # Goodness of fit
   y_fit = exponential(x, *popt)
   residuals = y - y_fit
   chi_sq = np.sum((residuals / sigma) ** 2)
   dof = len(y) - len(popt)
   print(f"\nReduced chi-squared: {chi_sq/dof:.3f}")

When pcov is inf
----------------

If ``pcov`` contains ``inf`` values, the covariance could not be estimated.
This typically means:

1. **Poor fit**: The model doesn't describe the data well
2. **Ill-conditioned**: Parameters are highly correlated or unidentifiable
3. **Insufficient data**: Not enough points to constrain all parameters

Solutions:

- Check if the model is appropriate
- Provide better initial guesses
- Add bounds to constrain parameters
- Use ``workflow='auto_global'`` for robust fitting

Absolute vs Relative Sigma
--------------------------

The ``absolute_sigma`` parameter affects uncertainty scaling:

.. code-block:: python

   # Sigma represents actual measurement uncertainties
   popt, pcov = fit(model, x, y, p0=[...], sigma=sigma, absolute_sigma=True)

   # Sigma represents relative weights (default)
   popt, pcov = fit(model, x, y, p0=[...], sigma=sigma, absolute_sigma=False)

With ``absolute_sigma=False`` (default), uncertainties are scaled based on
the residual variance, which may be more appropriate when actual measurement
errors are unknown.

Next Steps
----------

Now that you understand the basics, continue to :doc:`../three_workflows/index`
to learn about NLSQ's 3-workflow system for handling different fitting scenarios.
