nlsq.parameter\_estimation module
==================================

.. automodule:: nlsq.precision.parameter_estimation
   :members:
   :noindex:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``parameter_estimation`` module provides utilities for estimating initial parameter values from data.

Key Features
------------

- **Automatic initial guess generation** from data characteristics
- **Problem-specific heuristics** for common model types
- **Bounds inference** based on data range
- **Multi-start optimization** with parameter space exploration

Functions
---------

.. autofunction:: nlsq.parameter_estimation.estimate_initial_parameters
   :noindex:
.. autofunction:: nlsq.parameter_estimation.infer_parameter_bounds
   :noindex:

Example Usage
-------------

.. code-block:: python

   from nlsq.parameter_estimation import estimate_initial_parameters
   import jax.numpy as jnp


   # Define exponential model
   def exponential(x, a, b):
       return a * jnp.exp(-b * x)


   # Generate data
   x = jnp.linspace(0, 10, 100)
   y = 2.5 * jnp.exp(-0.5 * x) + 0.1 * jnp.random.randn(100)

   # Automatically estimate initial parameters
   p0 = estimate_initial_parameters(exponential, x, y, n_params=2)
   print(f"Estimated initial parameters: {p0}")

   # Use in curve fitting
   from nlsq import curve_fit

   popt, pcov = curve_fit(exponential, x, y, p0=p0)

Supported Model Types
---------------------

Automatic estimation for common model types:

- **Exponential decay**: ``y = a * exp(-b * x) + c``
- **Gaussian**: ``y = a * exp(-((x-b)/c)^2)``
- **Linear**: ``y = a * x + b``
- **Polynomial**: ``y = sum(a_i * x^i)``
- **Logistic**: ``y = L / (1 + exp(-k*(x-x0)))``

See Also
--------

- :doc:`nlsq.bound_inference` - Bound inference utilities
- :doc:`nlsq.algorithm_selector` - Algorithm selection
- :doc:`../reference/configuration` - Configuration reference
