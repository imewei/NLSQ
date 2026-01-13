Getting Started
===============

This chapter covers installation, your first curve fit, and how to interpret results.

.. toctree::
   :maxdepth: 1

   installation
   first_fit
   understanding_results

Chapter Overview
----------------

**Installation** (5 min)
   Install NLSQ and verify your setup, including optional GPU support.

**First Fit** (10 min)
   Fit an exponential decay model to data using the simplest possible approach.

**Understanding Results** (10 min)
   Learn what ``popt`` and ``pcov`` mean and how to calculate parameter uncertainties.

Quick Example
-------------

After completing this chapter, you'll be able to:

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   # Fit the data
   popt, pcov = fit(model, xdata, ydata, p0=[2.0, 0.5])

   # Extract results
   a_fit, b_fit = popt
   a_err, b_err = np.sqrt(np.diag(pcov))

   print(f"a = {a_fit:.3f} +/- {a_err:.3f}")
   print(f"b = {b_fit:.3f} +/- {b_err:.3f}")
