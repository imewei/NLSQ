Imaging Applications
====================

Curve fitting workflows for image analysis and microscopy.

2D Gaussian Fitting
-------------------

Point Spread Function (PSF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import fit


   def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
       """2D Gaussian with rotation."""
       x, y = xy
       a = jnp.cos(theta) ** 2 / (2 * sigma_x**2) + jnp.sin(theta) ** 2 / (2 * sigma_y**2)
       b = -jnp.sin(2 * theta) / (4 * sigma_x**2) + jnp.sin(2 * theta) / (4 * sigma_y**2)
       c = jnp.sin(theta) ** 2 / (2 * sigma_x**2) + jnp.cos(theta) ** 2 / (2 * sigma_y**2)
       return offset + amplitude * jnp.exp(
           -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
       )


   # Prepare data for 2D fitting
   x = np.arange(image.shape[1])
   y = np.arange(image.shape[0])
   X, Y = np.meshgrid(x, y)
   xy_data = (X.ravel(), Y.ravel())
   z_data = image.ravel()

   # Fit
   popt, pcov = fit(
       gaussian_2d,
       xy_data,
       z_data,
       p0=[image.max(), image.shape[1] // 2, image.shape[0] // 2, 5, 5, 0, image.min()],
   )

Symmetric 2D Gaussian
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def gaussian_2d_symmetric(xy, amplitude, x0, y0, sigma, offset):
       """2D Gaussian with equal width in x and y."""
       x, y = xy
       r_sq = (x - x0) ** 2 + (y - y0) ** 2
       return offset + amplitude * jnp.exp(-r_sq / (2 * sigma**2))

Airy Disk Pattern
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scipy.special import j1


   def airy_disk(xy, I0, x0, y0, radius, offset):
       """Airy disk pattern for diffraction-limited spots."""
       x, y = xy
       r = jnp.sqrt((x - x0) ** 2 + (y - y0) ** 2) / radius
       # Handle r=0 case
       r = jnp.where(r < 1e-10, 1e-10, r)
       pattern = (2 * j1(r) / r) ** 2
       return offset + I0 * pattern

FRAP Analysis
-------------

Single Component Recovery
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def frap_single(t, F0, Finf, tau):
       """Single-component FRAP recovery."""
       return Finf - (Finf - F0) * jnp.exp(-t / tau)

Two Component Recovery
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def frap_double(t, F0, Finf, f_fast, tau_fast, tau_slow):
       """Two-component FRAP (fast + slow diffusion)."""
       recovery_fast = f_fast * (1 - jnp.exp(-t / tau_fast))
       recovery_slow = (1 - f_fast) * (1 - jnp.exp(-t / tau_slow))
       return F0 + (Finf - F0) * (recovery_fast + recovery_slow)


   # FRAP fitting
   popt, pcov = fit(
       frap_single,
       time,
       fluorescence,
       p0=[0.2, 1.0, 5.0],
       bounds=([0, 0, 0.1], [1, 2, 1000]),
   )

Large Image Handling
--------------------

For images larger than 1 megapixel:

.. code-block:: python

   from nlsq import curve_fit_large

   if image.size > 1_000_000:
       popt, pcov = curve_fit_large(
           gaussian_2d,
           xy_data,
           z_data,
           p0=p0,
           show_progress=True,
       )
   else:
       popt, pcov = fit(gaussian_2d, xy_data, z_data, p0=p0)

Tips for Image Fitting
----------------------

1. **Pre-process images**:

   - Subtract background
   - Remove hot pixels
   - Apply appropriate smoothing

2. **Estimate initial guesses**:

   .. code-block:: python

      # Find centroid
      x0_guess = np.average(x, weights=image)
      y0_guess = np.average(y, weights=image)

      # Estimate width from second moments
      sigma_guess = np.sqrt(np.average((x - x0_guess) ** 2, weights=image))

3. **Use ROI for faster fitting**: Crop to region of interest

4. **For batch spot fitting**, consider using optimized libraries like
   scikit-image for detection, then NLSQ for precise fitting.

See Also
--------

- :doc:`/tutorials/05_large_datasets` - Large dataset tutorial
- :doc:`/howto/handle_large_data` - Large data guide
