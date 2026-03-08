Built-in Models
===============

NLSQ provides commonly used mathematical models ready for curve fitting.

Available Models
----------------

Import models from ``nlsq.core.functions``:

.. code-block:: python

   from nlsq.core.functions import (
       exponential_decay,
       gaussian,
       lorentzian,
       polynomial,
       power_law,
       sigmoid,
   )

Exponential Decay
-----------------

.. math::

   f(x) = a \cdot e^{-b \cdot x} + c

.. code-block:: python

   from nlsq import fit
   from nlsq.core.functions import exponential_decay

   # Parameters: amplitude, decay_rate, offset
   popt, pcov = fit(exponential_decay, x, y, p0=[2.0, 0.5, 0.0])
   a, b, c = popt

Gaussian (Normal Distribution)
------------------------------

.. math::

   f(x) = A \cdot e^{-\frac{(x - \mu)^2}{2\sigma^2}}

.. code-block:: python

   from nlsq.core.functions import gaussian

   # Parameters: amplitude, center, width
   popt, pcov = fit(gaussian, x, y, p0=[5.0, 0.0, 1.0])
   amp, mu, sigma = popt

Lorentzian (Cauchy Distribution)
--------------------------------

.. math::

   f(x) = \frac{A}{1 + \left(\frac{x - x_0}{\gamma}\right)^2}

.. code-block:: python

   from nlsq.core.functions import lorentzian

   # Parameters: amplitude, center, half-width
   popt, pcov = fit(lorentzian, x, y, p0=[5.0, 0.0, 1.0])
   amp, x0, gamma = popt

Power Law
---------

.. math::

   f(x) = a \cdot x^b

.. code-block:: python

   from nlsq.core.functions import power_law

   # Parameters: coefficient, exponent
   popt, pcov = fit(power_law, x, y, p0=[1.0, 2.0])
   a, b = popt

Sigmoid (Logistic Function)
----------------------------

.. math::

   f(x) = \frac{L}{1 + e^{-k(x - x_0)}} + b

.. code-block:: python

   from nlsq.core.functions import sigmoid

   # Parameters: max_value, midpoint, steepness, baseline
   popt, pcov = fit(sigmoid, x, y, p0=[1.0, 0.0, 1.0, 0.0])
   L, x0, k, b = popt

Polynomial
----------

Polynomials of any degree:

.. code-block:: python

   import jax.numpy as jnp


   # Define directly for curve fitting
   def quadratic(x, a, b, c):
       return a + b * x + c * x**2


   popt, pcov = fit(quadratic, x, y, p0=[0, 1, 0])

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from nlsq import fit
   from nlsq.core.functions import gaussian

   # Generate data: Gaussian peak with noise
   np.random.seed(42)
   x = np.linspace(-5, 5, 100)
   y_true = 3.0 * np.exp(-0.5 * ((x - 1.0) / 0.8) ** 2)
   y = y_true + 0.2 * np.random.normal(size=len(x))

   # Fit using built-in Gaussian
   popt, pcov = fit(gaussian, x, y, p0=[2.5, 0.5, 1.0])

   print("Fitted parameters:")
   print(f"  Amplitude: {popt[0]:.3f} (true: 3.0)")
   print(f"  Center:    {popt[1]:.3f} (true: 1.0)")
   print(f"  Width:     {popt[2]:.3f} (true: 0.8)")

Choosing the Right Model
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Data Pattern
     - Suggested Model
   * - Decreasing with asymptote
     - ``exponential_decay``
   * - Bell-shaped peak
     - ``gaussian`` or ``lorentzian``
   * - S-shaped curve
     - ``sigmoid``
   * - Power relationship
     - ``power_law``
   * - General trend
     - ``polynomial`` (define inline)

Tips:

- Gaussian peaks are narrower at the base
- Lorentzian peaks have heavier tails
- Use ``exponential_decay`` for radioactive decay, chemical reactions
- Use ``sigmoid`` for growth curves, dose-response

Next Steps
----------

- :doc:`custom_models` - Create your own models
- :doc:`model_validation` - Verify model correctness
