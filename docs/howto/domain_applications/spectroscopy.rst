Spectroscopy Applications
=========================

Common curve fitting workflows for spectroscopy data.

Peak Fitting
------------

Gaussian Peaks
~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from nlsq import fit


   def gaussian_peak(x, amplitude, center, sigma, baseline):
       """Single Gaussian peak with baseline."""
       return amplitude * jnp.exp(-0.5 * ((x - center) / sigma) ** 2) + baseline


   # Fit
   popt, pcov = fit(
       gaussian_peak,
       wavelength,
       intensity,
       p0=[1000, 500, 10, 100],
       bounds=([0, 400, 1, 0], [10000, 600, 50, 500]),
   )

Lorentzian Peaks
~~~~~~~~~~~~~~~~

.. code-block:: python

   def lorentzian_peak(x, amplitude, center, gamma, baseline):
       """Single Lorentzian peak with baseline."""
       return amplitude * gamma**2 / ((x - center) ** 2 + gamma**2) + baseline

Voigt Profile
~~~~~~~~~~~~~

.. code-block:: python

   def pseudo_voigt(x, amplitude, center, sigma, gamma, baseline):
       """Pseudo-Voigt approximation (weighted sum)."""
       eta = gamma / (sigma + gamma + 1e-10)
       G = jnp.exp(-0.5 * ((x - center) / sigma) ** 2)
       L = gamma**2 / ((x - center) ** 2 + gamma**2)
       return amplitude * (eta * L + (1 - eta) * G) + baseline

Multi-Peak Fitting
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def multi_gaussian(x, *params):
       """N Gaussian peaks: params = [a1, c1, s1, a2, c2, s2, ..., baseline]"""
       n_peaks = (len(params) - 1) // 3
       result = jnp.zeros_like(x)
       for i in range(n_peaks):
           a, c, s = params[3 * i], params[3 * i + 1], params[3 * i + 2]
           result += a * jnp.exp(-0.5 * ((x - c) / s) ** 2)
       return result + params[-1]


   # Use global optimization for multi-peak (many local minima)
   popt, pcov = fit(
       multi_gaussian,
       x,
       y,
       p0=initial_guesses,
       preset="robust",
   )

Fluorescence Lifetime
---------------------

Single Exponential
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def single_exponential(t, amplitude, tau, offset):
       """Single exponential decay."""
       return amplitude * jnp.exp(-t / tau) + offset

Bi-Exponential
~~~~~~~~~~~~~~

.. code-block:: python

   def biexponential(t, a1, tau1, a2, tau2, offset):
       """Bi-exponential decay (two components)."""
       return a1 * jnp.exp(-t / tau1) + a2 * jnp.exp(-t / tau2) + offset


   # Fluorescence lifetime (high precision)
   popt, pcov = fit(
       biexponential,
       time_ns,
       counts,
       p0=[1000, 2.5, 500, 8.0, 10],
       bounds=([0, 0.1, 0, 0.1, 0], [10000, 100, 10000, 100, 1000]),
       preset="quality",
   )

Stretched Exponential
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def stretched_exponential(t, amplitude, tau, beta, offset):
       """Stretched exponential (Kohlrausch-Williams-Watts)."""
       return amplitude * jnp.exp(-((t / tau) ** beta)) + offset

Tips for Spectroscopy Fitting
-----------------------------

1. **Estimate initial guesses from data**:

   .. code-block:: python

      amplitude_guess = np.max(y) - np.min(y)
      center_guess = x[np.argmax(y)]
      sigma_guess = estimate_fwhm(x, y) / 2.355

2. **Use appropriate bounds**:

   - Amplitudes: positive
   - Widths: positive, physically reasonable
   - Centers: within data range

3. **For overlapping peaks**, use global optimization:

   .. code-block:: python

      popt, pcov = fit(model, x, y, p0=p0, preset="global")

4. **Check residuals** for systematic patterns indicating missing peaks

See Also
--------

- :doc:`/howto/choose_model` - Model selection guide
- :doc:`/tutorials/04_multiple_parameters` - Multi-parameter fitting
