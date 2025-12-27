How to Debug Bad Fits
=====================

When curve fitting fails or produces poor results, this guide helps you
diagnose and fix the problem.

Common Symptoms
---------------

1. **Convergence failure**: Fit doesn't complete
2. **Wrong parameters**: Results are obviously incorrect
3. **Large uncertainties**: Parameter errors are huge
4. **Poor R²**: Low coefficient of determination
5. **Patterned residuals**: Systematic errors in residual plot

Diagnosis Flowchart
-------------------

::

   Fit fails?
   ├── Yes → Check error message → See "Convergence Failures"
   └── No → Check results
            ├── Parameters at bounds? → Relax bounds
            ├── Large uncertainties? → See "Poor Parameter Estimates"
            ├── Low R²? → See "Poor Fit Quality"
            └── Patterned residuals? → See "Model Mismatch"

Convergence Failures
--------------------

Error: "Optimal parameters not found"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Algorithm couldn't find a minimum.

**Solutions**:

1. Provide better initial guesses:

   .. code-block:: python

      # Estimate from data
      A_guess = np.max(y) - np.min(y)
      k_guess = 1.0 / (x[np.argmax(y)] - x[0])

      popt, pcov = curve_fit(model, x, y, p0=[A_guess, k_guess])

2. Use global optimization:

   .. code-block:: python

      from nlsq import fit

      popt, pcov = fit(model, x, y, preset="global")

3. Check data quality:

   .. code-block:: python

      # Check for NaN/Inf
      print(f"NaN in x: {np.any(np.isnan(x))}")
      print(f"NaN in y: {np.any(np.isnan(y))}")
      print(f"Inf in y: {np.any(np.isinf(y))}")

Error: "Maximum iterations reached"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Fit needs more iterations.

**Solutions**:

.. code-block:: python

   # Increase max iterations
   popt, pcov = curve_fit(model, x, y, max_nfev=10000)

Error: "Jacobian is singular"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Model is ill-conditioned or parameters are redundant.

**Solutions**:

1. Simplify the model
2. Fix some parameters
3. Rescale data

.. code-block:: python

   # Rescale data
   x_scale = np.max(np.abs(x))
   y_scale = np.max(np.abs(y))

   x_scaled = x / x_scale
   y_scaled = y / y_scale

   popt_scaled, pcov = curve_fit(model, x_scaled, y_scaled)

   # Unscale parameters as needed

Poor Parameter Estimates
------------------------

Parameters Have Large Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Parameters are poorly constrained by data.

**Diagnosis**:

.. code-block:: python

   perr = np.sqrt(np.diag(pcov))
   for i, (p, e) in enumerate(zip(popt, perr)):
       relative_error = abs(e / p) if p != 0 else float("inf")
       print(f"p{i}: {p:.4f} ± {e:.4f} ({relative_error*100:.1f}%)")

**Solutions**:

1. Need more data, especially in sensitive regions
2. Fix some parameters if known
3. Simplify the model

Parameters at Bounds
~~~~~~~~~~~~~~~~~~~~

**Cause**: True value is outside allowed range, or bound is too restrictive.

**Diagnosis**:

.. code-block:: python

   lower, upper = bounds
   for i, p in enumerate(popt):
       if np.isclose(p, lower[i]) or np.isclose(p, upper[i]):
           print(f"Parameter {i} is at bound: {p}")

**Solutions**:

1. Relax bounds
2. Check if bounds are physically realistic
3. Reconsider model

Highly Correlated Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Parameters trade off against each other.

**Diagnosis**:

.. code-block:: python

   perr = np.sqrt(np.diag(pcov))
   correlation = pcov / np.outer(perr, perr)

   for i in range(len(popt)):
       for j in range(i + 1, len(popt)):
           if abs(correlation[i, j]) > 0.9:
               print(f"High correlation: p{i} and p{j}: {correlation[i,j]:.3f}")

**Solutions**:

1. Reparameterize the model
2. Fix one of the correlated parameters
3. Acquire more diverse data

Poor Fit Quality
----------------

Low R² Value
~~~~~~~~~~~~

**Cause**: Model doesn't explain the data well.

**Solutions**:

1. Check if model is appropriate for data:

   .. code-block:: python

      # Visualize data and model
      plt.scatter(x, y, alpha=0.5, label="Data")
      plt.plot(x, model(x, *popt), "r-", label="Fit")
      plt.legend()
      plt.show()

2. Consider different models (see :doc:`choose_model`)

3. Check for outliers:

   .. code-block:: python

      residuals = y - model(x, *popt)
      z_scores = (residuals - np.mean(residuals)) / np.std(residuals)
      outliers = np.abs(z_scores) > 3

      if np.any(outliers):
          print(f"Found {np.sum(outliers)} potential outliers")

High RMSE
~~~~~~~~~

**Cause**: Large prediction errors.

**Solutions**:

1. Check noise level in data
2. Use weighted fitting if noise varies:

   .. code-block:: python

      sigma = estimate_uncertainties(x, y)
      popt, pcov = curve_fit(model, x, y, sigma=sigma, absolute_sigma=True)

Model Mismatch
--------------

Systematic Patterns in Residuals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Model doesn't capture the true relationship.

**Diagnosis**:

.. code-block:: python

   residuals = y - model(x, *popt)

   plt.figure(figsize=(12, 4))

   plt.subplot(1, 3, 1)
   plt.scatter(x, residuals, alpha=0.5)
   plt.axhline(0, color="r", linestyle="--")
   plt.xlabel("x")
   plt.ylabel("Residuals")
   plt.title("Residuals vs x")

   plt.subplot(1, 3, 2)
   plt.scatter(model(x, *popt), residuals, alpha=0.5)
   plt.axhline(0, color="r", linestyle="--")
   plt.xlabel("Predicted y")
   plt.ylabel("Residuals")
   plt.title("Residuals vs Predicted")

   plt.subplot(1, 3, 3)
   plt.hist(residuals, bins=20)
   plt.xlabel("Residual value")
   plt.ylabel("Count")
   plt.title("Residual Distribution")

   plt.tight_layout()
   plt.show()

**Patterns and solutions**:

- **U-shape or curved**: Missing quadratic term
- **Oscillating**: Missing periodic component
- **Increasing spread**: Heteroscedastic data (use weighted fitting)
- **Asymmetric histogram**: Non-normal errors (use robust fitting)

Debugging Checklist
-------------------

.. code-block:: text

   □ Data quality
     □ No NaN or Inf values
     □ Reasonable value ranges
     □ Sufficient data points

   □ Model appropriateness
     □ Matches known physics
     □ Correct number of parameters
     □ All parameters identifiable

   □ Initial guesses
     □ Estimated from data
     □ Within physical bounds
     □ Order of magnitude correct

   □ Bounds
     □ Physically motivated
     □ Not too restrictive
     □ Initial guess within bounds

   □ Fit configuration
     □ Sufficient max iterations
     □ Appropriate tolerance
     □ Correct method (trf for bounds)

Complete Debugging Example
--------------------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit
   import matplotlib.pyplot as plt


   def debug_fit(model, x, y, p0, bounds=None):
       """Comprehensive fit debugging."""

       print("=" * 60)
       print("FIT DEBUGGING REPORT")
       print("=" * 60)

       # 1. Check data
       print("\n1. DATA CHECK")
       print(f"   x: {len(x)} points, range [{x.min():.3g}, {x.max():.3g}]")
       print(f"   y: {len(y)} points, range [{y.min():.3g}, {y.max():.3g}]")
       print(f"   NaN in x: {np.any(np.isnan(x))}")
       print(f"   NaN in y: {np.any(np.isnan(y))}")

       # 2. Try fit
       print("\n2. FITTING")
       try:
           if bounds:
               popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds)
           else:
               popt, pcov = curve_fit(model, x, y, p0=p0)
           print("   Status: SUCCESS")
       except Exception as e:
           print(f"   Status: FAILED - {e}")
           return

       # 3. Parameter analysis
       print("\n3. PARAMETERS")
       perr = np.sqrt(np.diag(pcov))
       for i, (p, e) in enumerate(zip(popt, perr)):
           rel_err = abs(e / p) * 100 if p != 0 else float("inf")
           status = "OK" if rel_err < 50 else "HIGH UNCERTAINTY"
           print(f"   p{i}: {p:10.4g} ± {e:10.4g} ({rel_err:5.1f}%) - {status}")

       # 4. Correlation check
       print("\n4. CORRELATIONS")
       corr = pcov / np.outer(perr, perr)
       high_corr = []
       for i in range(len(popt)):
           for j in range(i + 1, len(popt)):
               if abs(corr[i, j]) > 0.9:
                   high_corr.append((i, j, corr[i, j]))
       if high_corr:
           for i, j, c in high_corr:
               print(f"   WARNING: p{i}-p{j} correlation = {c:.3f}")
       else:
           print("   All correlations < 0.9")

       # 5. Residuals
       print("\n5. FIT QUALITY")
       y_pred = model(x, *popt)
       residuals = y - y_pred
       ss_res = np.sum(residuals**2)
       ss_tot = np.sum((y - np.mean(y)) ** 2)
       r2 = 1 - ss_res / ss_tot
       rmse = np.sqrt(np.mean(residuals**2))

       print(f"   R² = {r2:.4f}")
       print(f"   RMSE = {rmse:.4g}")

       if r2 < 0.9:
           print("   WARNING: R² < 0.9 suggests poor fit")

       return popt, pcov


   # Example usage
   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y = 2.5 * np.exp(-0.5 * x) + 0.3 + 0.1 * np.random.randn(100)

   debug_fit(model, x, y, p0=[2, 0.5, 0.3])

See Also
--------

- :doc:`troubleshooting` - General troubleshooting guide
- :doc:`choose_model` - Model selection
- :doc:`/tutorials/02_understanding_results` - Interpreting results
