Tutorial 4: Multi-Parameter Models
===================================

In this tutorial, you'll learn how to fit complex models with many parameters,
handle parameter correlations, and work with multi-component models.

What You'll Learn
-----------------

- Fitting models with 5+ parameters
- Providing good initial guesses
- Understanding parameter correlations
- Multi-peak and multi-component fitting

Prerequisites
-------------

- Completed tutorials 1-3

The Challenge of Multi-Parameter Fitting
----------------------------------------

As the number of parameters increases, fitting becomes harder:

- More local minima (wrong solutions)
- Stronger parameter correlations
- Greater sensitivity to initial guesses

With good practices, NLSQ handles these challenges effectively.

Example: Damped Oscillation
---------------------------

A damped oscillation has 5 parameters:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   def damped_oscillation(t, A, omega, gamma, phi, offset):
       """
       Damped harmonic oscillator.

       Parameters:
           A: Amplitude
           omega: Angular frequency
           gamma: Damping rate
           phi: Phase
           offset: DC offset
       """
       return A * jnp.exp(-gamma * t) * jnp.cos(omega * t + phi) + offset


   # Generate data
   np.random.seed(42)
   t = np.linspace(0, 10, 200)

   # True parameters
   A_true, omega_true, gamma_true, phi_true, offset_true = 3.0, 5.0, 0.3, 0.5, 1.0

   y_true = (
       A_true * np.exp(-gamma_true * t) * np.cos(omega_true * t + phi_true) + offset_true
   )
   y = y_true + 0.2 * np.random.normal(size=len(t))

Estimating Initial Guesses
--------------------------

Good initial guesses are crucial. Estimate from data characteristics:

.. code-block:: python

   # Estimate initial guesses from data
   A_guess = (np.max(y) - np.min(y)) / 2  # Half the range
   offset_guess = np.mean(y)  # Mean value

   # Estimate frequency from zero crossings or FFT
   from scipy.signal import find_peaks

   peaks, _ = find_peaks(y)
   if len(peaks) > 1:
       period = np.mean(np.diff(t[peaks]))
       omega_guess = 2 * np.pi / period
   else:
       omega_guess = 3.0  # Fallback

   gamma_guess = 0.2  # Reasonable starting point
   phi_guess = 0.0  # Start at zero phase

   p0 = [A_guess, omega_guess, gamma_guess, phi_guess, offset_guess]
   print(f"Initial guesses: {p0}")

Fitting with Bounds
-------------------

Add physical constraints:

.. code-block:: python

   # Physical bounds
   bounds = (
       [0, 0, 0, -np.pi, -10],  # Lower bounds
       [10, 20, 2, np.pi, 10],  # Upper bounds
   )

   # Fit
   popt, pcov = curve_fit(damped_oscillation, t, y, p0=p0, bounds=bounds)

   # Results
   param_names = ["Amplitude", "Omega", "Gamma", "Phi", "Offset"]
   perr = np.sqrt(np.diag(pcov))

   print("\nFitted Parameters:")
   print("-" * 50)
   for name, val, err, true_val in zip(
       param_names, popt, perr, [A_true, omega_true, gamma_true, phi_true, offset_true]
   ):
       print(f"{name:12}: {val:8.4f} ± {err:.4f} (true: {true_val})")

Understanding Parameter Correlations
------------------------------------

Correlated parameters are harder to determine independently:

.. code-block:: python

   # Calculate correlation matrix
   perr = np.sqrt(np.diag(pcov))
   correlation = pcov / np.outer(perr, perr)

   print("\nParameter Correlations:")
   print("-" * 50)
   print("           ", "  ".join(f"{n[:6]:>8}" for n in param_names))
   for i, name in enumerate(param_names):
       row = "  ".join(f"{correlation[i,j]:8.3f}" for j in range(len(param_names)))
       print(f"{name[:10]:10} {row}")

Correlations near ±1 indicate parameters that trade off against each other.

Multi-Peak Fitting
------------------

Fitting multiple Gaussian peaks:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   def multi_gaussian(x, *params):
       """
       Sum of Gaussian peaks.

       params: [A1, mu1, sigma1, A2, mu2, sigma2, ..., baseline]
       """
       n_peaks = (len(params) - 1) // 3
       result = params[-1]  # baseline

       for i in range(n_peaks):
           A = params[3 * i]
           mu = params[3 * i + 1]
           sigma = params[3 * i + 2]
           result = result + A * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

       return result


   # Generate data with 3 peaks
   np.random.seed(42)
   x = np.linspace(0, 10, 200)

   # True peaks: (amplitude, center, width)
   peaks_true = [(3.0, 2.0, 0.5), (5.0, 5.0, 0.8), (2.0, 8.0, 0.4)]
   baseline_true = 0.5

   y_true = baseline_true
   for A, mu, sigma in peaks_true:
       y_true = y_true + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
   y = y_true + 0.2 * np.random.normal(size=len(x))

   # Initial guesses for 3 peaks + baseline
   p0 = [
       2.5,
       2.0,
       0.6,  # Peak 1
       4.0,
       5.0,
       0.7,  # Peak 2
       1.5,
       8.0,
       0.5,  # Peak 3
       0.3,
   ]  # Baseline

   # Bounds
   lower = [0, 0, 0.1, 0, 3, 0.1, 0, 6, 0.1, -1]
   upper = [10, 4, 2, 10, 7, 2, 10, 10, 2, 2]
   bounds = (lower, upper)

   # Fit
   popt, pcov = curve_fit(multi_gaussian, x, y, p0=p0, bounds=bounds)

   # Print results
   print("Multi-Peak Fit Results:")
   print("-" * 50)
   for i in range(3):
       A, mu, sigma = popt[3 * i : 3 * i + 3]
       print(f"Peak {i+1}: A={A:.3f}, center={mu:.3f}, width={sigma:.3f}")
   print(f"Baseline: {popt[-1]:.3f}")

Using Presets for Robust Fitting
--------------------------------

For challenging multi-parameter fits, use the ``robust`` or ``global`` preset:

.. code-block:: python

   from nlsq import fit

   # Global optimization with multiple starts
   popt, pcov = fit(
       damped_oscillation,
       t,
       y,
       p0=p0,
       bounds=bounds,
       preset="global",  # Multi-start optimization
   )

   print("Global optimization result:")
   print(f"Parameters: {popt}")

The ``global`` preset runs multiple optimizations from different starting
points and returns the best result.

Tips for Multi-Parameter Fitting
--------------------------------

1. **Estimate Initial Guesses**

   Use data characteristics:

   - Amplitudes from peak heights
   - Positions from peak locations
   - Widths from peak widths at half maximum

2. **Use Appropriate Bounds**

   - Physical constraints (positive amplitudes, etc.)
   - Prevent parameter blow-up

3. **Consider Fixing Some Parameters**

   If you know some values, fix them:

   .. code-block:: python

      # Fix baseline by subtracting it first
      y_corrected = y - known_baseline

      # Fit with fewer parameters
      popt, pcov = curve_fit(model_without_baseline, x, y_corrected, ...)

4. **Use Global Optimization**

   For 6+ parameters or complex landscapes:

   .. code-block:: python

      from nlsq import fit

      popt, pcov = fit(model, x, y, preset="global")

5. **Check the Fit Visually**

   Always plot your results:

   .. code-block:: python

      import matplotlib.pyplot as plt

      plt.figure(figsize=(10, 6))
      plt.scatter(x, y, alpha=0.5, label="Data")
      plt.plot(x, model(x, *popt), "r-", linewidth=2, label="Fit")
      plt.legend()
      plt.show()

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit
   import matplotlib.pyplot as plt


   # Complex model: sum of exponential and oscillation
   def complex_model(t, A_exp, k, A_osc, omega, phi, offset):
       exponential = A_exp * jnp.exp(-k * t)
       oscillation = A_osc * jnp.cos(omega * t + phi)
       return exponential + oscillation + offset


   # Generate data
   np.random.seed(42)
   t = np.linspace(0, 10, 150)
   y_true = 2.0 * np.exp(-0.3 * t) + 1.0 * np.cos(3.0 * t + 0.5) + 0.5
   y = y_true + 0.15 * np.random.normal(size=len(t))

   # Initial guesses
   p0 = [2.0, 0.5, 1.0, 3.0, 0.0, 0.5]

   # Bounds
   bounds = ([0, 0, 0, 0, -np.pi, -2], [10, 5, 5, 10, np.pi, 2])

   # Fit
   popt, pcov = curve_fit(complex_model, t, y, p0=p0, bounds=bounds)
   perr = np.sqrt(np.diag(pcov))

   # Print results
   names = ["A_exp", "k", "A_osc", "omega", "phi", "offset"]
   true_vals = [2.0, 0.3, 1.0, 3.0, 0.5, 0.5]

   print("Complex Model Fit Results:")
   print("=" * 60)
   for name, val, err, true in zip(names, popt, perr, true_vals):
       print(f"{name:8}: {val:8.4f} ± {err:.4f}  (true: {true})")

   # Plot
   plt.figure(figsize=(10, 6))
   plt.scatter(t, y, alpha=0.5, s=20, label="Data")
   plt.plot(t, complex_model(t, *popt), "r-", linewidth=2, label="Fit")
   plt.xlabel("Time")
   plt.ylabel("Signal")
   plt.legend()
   plt.title("Complex Model Fit")
   plt.show()

Key Takeaways
-------------

1. Multi-parameter fits need **good initial guesses**
2. Estimate guesses from **data characteristics**
3. Use **bounds** to constrain the search space
4. Use ``preset='global'`` for **robust optimization**
5. Check for **parameter correlations** in the covariance matrix
6. Always **visualize** your fit results

Next Steps
----------

In :doc:`05_large_datasets`, you'll learn how to:

- Handle datasets with millions of points
- Use streaming optimization
- Manage memory efficiently

.. seealso::

   - :doc:`/howto/choose_model` - Selecting appropriate models
   - :doc:`/explanation/trust_region` - Optimization algorithm details
