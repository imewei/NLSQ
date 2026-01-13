Model Selection
===============

This chapter covers how to choose and define models for curve fitting.

.. toctree::
   :maxdepth: 1

   built_in_models
   custom_models
   model_validation

Chapter Overview
----------------

**Built-in Models** (10 min)
   Use NLSQ's library of common mathematical models.

**Custom Models** (10 min)
   Write your own model functions with JAX.

**Model Validation** (5 min)
   Check that your model is correct before fitting.

Quick Reference
---------------

.. code-block:: python

   # Built-in exponential decay
   from nlsq.functions import exponential_decay

   popt, pcov = fit(exponential_decay, x, y, p0=[1, 0.5])

   # Custom model
   import jax.numpy as jnp


   def my_model(x, a, b, c):
       return a * jnp.sin(b * x) + c


   popt, pcov = fit(my_model, x, y, p0=[1, 1, 0])

Key Rule
--------

.. important::

   All model functions must use ``jax.numpy`` (not ``numpy``) for mathematical
   operations. This enables automatic differentiation and GPU acceleration.

   .. code-block:: python

      import jax.numpy as jnp  # Use this!
      import numpy as np  # Not for model math


      def model(x, a, b):
          return a * jnp.exp(-b * x)  # jnp, not np
