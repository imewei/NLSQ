Model Functions Reference
=========================

NLSQ provides built-in model functions for common curve fitting scenarios.

.. module:: nlsq.core.functions

Using Built-in Functions
------------------------

Import from ``nlsq.core.functions``:

.. code-block:: python

   from nlsq.core.functions import gaussian, exponential_decay, lorentzian

All functions are JAX-compatible and JIT-compilable. Each function includes:

- Automatic p0 estimation via ``.estimate_p0(xdata, ydata)``
- Reasonable default bounds via ``.bounds()``

Peak Functions
--------------

gaussian
~~~~~~~~

.. autofunction:: nlsq.core.functions.gaussian
   :no-index:

lorentzian
~~~~~~~~~~

.. autofunction:: nlsq.core.functions.lorentzian
   :no-index:

Exponential Functions
---------------------

exponential_decay
~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.core.functions.exponential_decay
   :no-index:

exponential_growth
~~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.core.functions.exponential_growth
   :no-index:

Sigmoid Functions
-----------------

sigmoid
~~~~~~~

.. autofunction:: nlsq.core.functions.sigmoid
   :no-index:

Power Functions
---------------

power_law
~~~~~~~~~

.. autofunction:: nlsq.core.functions.power_law
   :no-index:

linear
~~~~~~

.. autofunction:: nlsq.core.functions.linear
   :no-index:

polynomial
~~~~~~~~~~

.. autofunction:: nlsq.core.functions.polynomial
   :no-index:

Polynomial function factory for arbitrary degree.

**Example:**

.. code-block:: python

   import jax.numpy as jnp


   # Define directly for curve fitting
   def quadratic(x, a, b, c):
       return a + b * x + c * x**2

Creating Custom Functions
-------------------------

Custom model functions must use JAX operations:

.. code-block:: python

   import jax.numpy as jnp


   def custom_model(x, a, b, c, d):
       """Custom model combining exponential and oscillation."""
       decay = a * jnp.exp(-b * x)
       oscillation = c * jnp.sin(d * x)
       return decay * oscillation


   # Use like any built-in function
   popt, pcov = fit(custom_model, x, y, p0=[1, 0.1, 1, 2 * jnp.pi])

**Rules for custom functions:**

1. Use ``jax.numpy`` instead of ``numpy``
2. Avoid Python control flow on traced values
3. Keep functions pure (no side effects)
4. First parameter must be the independent variable

See Also
--------

- :doc:`/tutorials/routine/getting_started/first_fit` - Basic usage examples
- :doc:`/explanation/jax_autodiff` - Why JAX is required
- :doc:`/howto/choose_model` - Model selection guide
