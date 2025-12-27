Tutorial 2: Understanding Fit Results
======================================

In this tutorial, you'll learn how to interpret the output of ``curve_fit``
and assess the quality of your fit.

What You'll Learn
-----------------

- What the covariance matrix tells you
- How to calculate parameter uncertainties
- How to compute goodness-of-fit metrics
- How to visualize residuals

Prerequisites
-------------

- Completed :doc:`01_first_fit`

The Covariance Matrix
---------------------

When you call ``curve_fit``, you get two outputs:

.. code-block:: python

   popt, pcov = curve_fit(model, x, y)

- **popt**: The optimal parameter values
- **pcov**: The covariance matrix

The covariance matrix ``pcov`` is a square matrix where:

- Diagonal elements (``pcov[i,i]``) are the **variances** of each parameter
- Off-diagonal elements (``pcov[i,j]``) are the **covariances** between parameters

Parameter Uncertainties
-----------------------

The standard error (uncertainty) of each parameter is the square root of
its variance:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   # Define model
   def exponential_decay(x, A, k):
       return A * jnp.exp(-k * x)


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 50)
   y = 2.5 * np.exp(-0.5 * x) + 0.1 * np.random.normal(size=len(x))

   # Fit
   popt, pcov = curve_fit(exponential_decay, x, y)

   # Calculate uncertainties (standard errors)
   perr = np.sqrt(np.diag(pcov))

   # Print results with uncertainties
   print("Fitted Parameters:")
   print(f"  A = {popt[0]:.4f} ± {perr[0]:.4f}")
   print(f"  k = {popt[1]:.4f} ± {perr[1]:.4f}")

Expected output:

.. code-block:: text

   Fitted Parameters:
     A = 2.4892 ± 0.0312
     k = 0.4987 ± 0.0089

Confidence Intervals
--------------------

For a 95% confidence interval, multiply the standard error by 1.96:

.. code-block:: python

   from scipy import stats

   # Degrees of freedom
   n = len(x)
   p = len(popt)
   dof = n - p

   # 95% confidence interval multiplier
   t_value = stats.t.ppf(0.975, dof)

   # Calculate confidence intervals
   for i, (param, err) in enumerate(zip(popt, perr)):
       lower = param - t_value * err
       upper = param + t_value * err
       print(f"Parameter {i}: {param:.4f} [{lower:.4f}, {upper:.4f}]")

Using the Result Object
-----------------------

NLSQ returns an enhanced result object with built-in statistics:

.. code-block:: python

   # Get full result object (don't unpack to tuple)
   result = curve_fit(exponential_decay, x, y)

   # Access parameters (same as popt)
   print(f"Parameters: {result.popt}")

   # Access covariance (same as pcov)
   print(f"Covariance shape: {result.pcov.shape}")

   # Built-in confidence intervals
   ci = result.confidence_intervals(alpha=0.95)
   print(f"95% CI for A: [{ci[0,0]:.4f}, {ci[0,1]:.4f}]")
   print(f"95% CI for k: [{ci[1,0]:.4f}, {ci[1,1]:.4f}]")

Goodness-of-Fit Metrics
-----------------------

R-squared (Coefficient of Determination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

R² measures how well the model explains the variance in the data:

.. code-block:: python

   # Calculate R²
   y_pred = exponential_decay(x, *popt)
   ss_res = np.sum((y - y_pred) ** 2)
   ss_tot = np.sum((y - np.mean(y)) ** 2)
   r_squared = 1 - (ss_res / ss_tot)

   print(f"R² = {r_squared:.4f}")

- R² = 1.0: Perfect fit
- R² = 0.0: Model is no better than mean
- R² > 0.9: Generally considered a good fit

Root Mean Square Error (RMSE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RMSE gives the typical size of the residuals:

.. code-block:: python

   rmse = np.sqrt(np.mean((y - y_pred) ** 2))
   print(f"RMSE = {rmse:.4f}")

Using Built-in Metrics
~~~~~~~~~~~~~~~~~~~~~~

The result object has these metrics built-in:

.. code-block:: python

   result = curve_fit(exponential_decay, x, y)

   print(f"R² = {result.r_squared:.4f}")
   print(f"Adjusted R² = {result.adj_r_squared:.4f}")
   print(f"RMSE = {result.rmse:.4f}")
   print(f"MAE = {result.mae:.4f}")
   print(f"AIC = {result.aic:.2f}")
   print(f"BIC = {result.bic:.2f}")

Visualizing Residuals
---------------------

Residuals are the differences between data and model. Good fits have
random, normally distributed residuals.

.. code-block:: python

   import matplotlib.pyplot as plt

   # Calculate residuals
   residuals = y - exponential_decay(x, *popt)

   # Create figure with subplots
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # Plot 1: Data and fit
   ax1 = axes[0, 0]
   ax1.scatter(x, y, alpha=0.7, label="Data")
   x_smooth = np.linspace(x.min(), x.max(), 200)
   ax1.plot(x_smooth, exponential_decay(x_smooth, *popt), "r-", label="Fit", linewidth=2)
   ax1.set_xlabel("x")
   ax1.set_ylabel("y")
   ax1.set_title("Data and Fit")
   ax1.legend()

   # Plot 2: Residuals vs x
   ax2 = axes[0, 1]
   ax2.scatter(x, residuals, alpha=0.7)
   ax2.axhline(y=0, color="r", linestyle="--")
   ax2.set_xlabel("x")
   ax2.set_ylabel("Residuals")
   ax2.set_title("Residuals vs x")

   # Plot 3: Residual histogram
   ax3 = axes[1, 0]
   ax3.hist(residuals, bins=15, edgecolor="black", alpha=0.7)
   ax3.set_xlabel("Residual value")
   ax3.set_ylabel("Frequency")
   ax3.set_title("Residual Distribution")

   # Plot 4: Q-Q plot
   ax4 = axes[1, 1]
   stats.probplot(residuals, dist="norm", plot=ax4)
   ax4.set_title("Q-Q Plot")

   plt.tight_layout()
   plt.show()

What to Look For
~~~~~~~~~~~~~~~~

- **Random scatter**: Residuals should show no pattern
- **Normal distribution**: Histogram should be roughly bell-shaped
- **Q-Q plot**: Points should fall on the diagonal line

Warning signs:

- **Curved pattern**: Model may be wrong
- **Increasing spread**: Heteroscedasticity (may need weighted fit)
- **Outliers**: Consider robust fitting

Quick Summary Method
--------------------

The result object has a ``summary()`` method for a complete overview:

.. code-block:: python

   result = curve_fit(exponential_decay, x, y)
   result.summary()

This prints a formatted table with:

- Parameter values and uncertainties
- Confidence intervals
- Goodness-of-fit metrics
- Convergence information

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   # Model
   def exponential_decay(x, A, k):
       return A * jnp.exp(-k * x)


   # Data
   np.random.seed(42)
   x = np.linspace(0, 10, 50)
   y = 2.5 * np.exp(-0.5 * x) + 0.1 * np.random.normal(size=len(x))

   # Fit and get result object
   result = curve_fit(exponential_decay, x, y)

   # Uncertainties
   perr = np.sqrt(np.diag(result.pcov))

   # Print summary
   print("=" * 50)
   print("FIT RESULTS")
   print("=" * 50)
   print(f"\nParameters:")
   print(f"  A = {result.popt[0]:.4f} ± {perr[0]:.4f}")
   print(f"  k = {result.popt[1]:.4f} ± {perr[1]:.4f}")
   print(f"\nGoodness of Fit:")
   print(f"  R² = {result.r_squared:.4f}")
   print(f"  RMSE = {result.rmse:.4f}")

Key Takeaways
-------------

1. **Covariance matrix** contains variance (diagonal) and covariances
2. **Standard errors** are square roots of diagonal elements
3. **R²** measures explained variance (closer to 1 is better)
4. **Residual plots** reveal model problems
5. **result.summary()** gives a complete overview

Next Steps
----------

In :doc:`03_fitting_with_bounds`, you'll learn how to:

- Constrain parameters to physical ranges
- Use bounds to improve convergence
- Handle edge cases

.. seealso::

   - :doc:`/explanation/how_fitting_works` - Mathematical details
   - :doc:`/howto/debug_bad_fits` - Troubleshooting poor fits
