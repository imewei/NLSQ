How to Choose a Model Function
==============================

This guide helps you select the right mathematical model for your data.

Overview
--------

Choosing the right model is crucial for successful curve fitting. The model
should:

1. Match the underlying physics/chemistry of your system
2. Have the right number of parameters (not too few, not too many)
3. Be identifiable (parameters can be uniquely determined)

Common Model Types
------------------

Exponential Models
~~~~~~~~~~~~~~~~~~

**Single Exponential Decay**

.. code-block:: python

   def exponential_decay(x, A, k, offset):
       return A * jnp.exp(-k * x) + offset

Use when:
- Radioactive decay
- First-order chemical kinetics
- RC circuit discharge
- Fluorescence lifetime (single species)

**Bi-exponential Decay**

.. code-block:: python

   def biexponential(x, A1, k1, A2, k2, offset):
       return A1 * jnp.exp(-k1 * x) + A2 * jnp.exp(-k2 * x) + offset

Use when:
- Two-component systems
- Energy transfer processes
- Drug elimination kinetics

**Stretched Exponential (Kohlrausch)**

.. code-block:: python

   def stretched_exp(x, A, tau, beta, offset):
       return A * jnp.exp(-jnp.power(x / tau, beta)) + offset

Use when:
- Disordered systems
- Polymer relaxation
- Non-exponential decay

Peak Models
~~~~~~~~~~~

**Gaussian**

.. code-block:: python

   def gaussian(x, A, mu, sigma, offset):
       return A * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset

Use when:
- Spectral peaks
- Chromatography peaks
- Error distributions

**Lorentzian**

.. code-block:: python

   def lorentzian(x, A, x0, gamma, offset):
       return A * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset

Use when:
- Resonance phenomena
- NMR/ESR peaks
- Optical absorption lines

**Voigt (Gaussian + Lorentzian convolution)**

.. code-block:: python

   from scipy.special import voigt_profile


   def voigt(x, A, x0, sigma, gamma, offset):
       return A * voigt_profile(x - x0, sigma, gamma) + offset

Use when:
- Spectral lines with both Gaussian and Lorentzian broadening
- X-ray diffraction peaks
- High-resolution spectroscopy

Polynomial Models
~~~~~~~~~~~~~~~~~

**Linear**

.. code-block:: python

   def linear(x, m, b):
       return m * x + b

Use when:
- Linear relationships
- Calibration curves
- Simple trends

**Quadratic**

.. code-block:: python

   def quadratic(x, a, b, c):
       return a * x**2 + b * x + c

Use when:
- Parabolic trajectories
- Second-order corrections
- Curvature in data

Sigmoidal Models
~~~~~~~~~~~~~~~~

**Logistic**

.. code-block:: python

   def logistic(x, L, k, x0, offset):
       return L / (1 + jnp.exp(-k * (x - x0))) + offset

Use when:
- Dose-response curves
- Growth curves
- Saturation phenomena

**Hill Equation**

.. code-block:: python

   def hill(x, Vmax, K, n, offset):
       return Vmax * x**n / (K**n + x**n) + offset

Use when:
- Enzyme kinetics (cooperativity)
- Ligand binding
- Cooperative processes

Model Selection Criteria
------------------------

1. Physical Justification
~~~~~~~~~~~~~~~~~~~~~~~~~

Choose models that match the underlying mechanism:

- **Know the physics**: Use theory to guide model selection
- **Avoid arbitrary models**: Don't just fit polynomials to everything
- **Consider dimensionality**: Parameters should have physical meaning

2. Goodness of Fit Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare models using:

**R-squared (R²)**

Higher is better, but can be misleading with many parameters.

**Akaike Information Criterion (AIC)**

.. code-block:: python

   AIC = n * log(RSS / n) + 2 * k

Lower is better. Penalizes extra parameters.

**Bayesian Information Criterion (BIC)**

.. code-block:: python

   BIC = n * log(RSS / n) + k * log(n)

Lower is better. Stronger penalty for parameters than AIC.

**Using NLSQ for model comparison:**

.. code-block:: python

   result1 = curve_fit(model1, x, y)
   result2 = curve_fit(model2, x, y)

   print(f"Model 1: AIC={result1.aic:.2f}, BIC={result1.bic:.2f}")
   print(f"Model 2: AIC={result2.aic:.2f}, BIC={result2.bic:.2f}")

3. Residual Analysis
~~~~~~~~~~~~~~~~~~~~

Good models have:

- Random residuals (no pattern)
- Normal distribution of residuals
- Constant variance (homoscedasticity)

.. code-block:: python

   residuals = y - model(x, *popt)

   # Check for patterns
   plt.scatter(x, residuals)
   plt.axhline(0, color="r", linestyle="--")
   plt.xlabel("x")
   plt.ylabel("Residuals")
   plt.title("Residual Plot")

4. Parameter Identifiability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avoid models where:

- Parameters are highly correlated (correlation > 0.95)
- Parameters are at bounds
- Uncertainties are very large

.. code-block:: python

   # Check parameter correlations
   perr = np.sqrt(np.diag(pcov))
   correlation = pcov / np.outer(perr, perr)
   print("Correlation matrix:")
   print(correlation)

Workflow: Choosing a Model
--------------------------

1. **Start simple**: Try the simplest physically-motivated model
2. **Check residuals**: Look for systematic patterns
3. **Add complexity if needed**: Add terms to address residual patterns
4. **Compare with AIC/BIC**: Quantify whether extra parameters are justified
5. **Validate**: Use cross-validation or holdout data

Example: Choosing Between Models
--------------------------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit

   # Generate data (bi-exponential)
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = 3.0 * np.exp(-0.8 * x) + 1.0 * np.exp(-0.1 * x)
   y = y_true + 0.1 * np.random.randn(len(x))


   # Model 1: Single exponential
   def single_exp(x, A, k, offset):
       return A * jnp.exp(-k * x) + offset


   # Model 2: Bi-exponential
   def bi_exp(x, A1, k1, A2, k2, offset):
       return A1 * jnp.exp(-k1 * x) + A2 * jnp.exp(-k2 * x) + offset


   # Fit both
   result1 = curve_fit(single_exp, x, y, p0=[3, 0.5, 0.5])
   result2 = curve_fit(bi_exp, x, y, p0=[2, 1, 1, 0.1, 0.1])

   # Compare
   print("Model Comparison:")
   print("-" * 40)
   print(f"Single exponential: AIC={result1.aic:.2f}, R²={result1.r_squared:.4f}")
   print(f"Bi-exponential:     AIC={result2.aic:.2f}, R²={result2.r_squared:.4f}")

   # Decision: lower AIC wins (if difference > 2)
   delta_aic = result1.aic - result2.aic
   if delta_aic > 2:
       print("\n→ Bi-exponential is significantly better")
   elif delta_aic < -2:
       print("\n→ Single exponential is significantly better")
   else:
       print("\n→ Models are comparable, prefer simpler")

See Also
--------

- :doc:`/tutorials/04_multiple_parameters` - Fitting complex models
- :doc:`/explanation/how_fitting_works` - Understanding curve fitting
- :doc:`debug_bad_fits` - Troubleshooting poor fits
