Built-in Models
===============

NLSQ provides commonly used mathematical models ready for curve fitting.

Available Models
----------------

Import models from ``nlsq.functions``:

.. code-block:: python

   from nlsq.functions import (
       exponential_decay,
       gaussian,
       lorentzian,
       polynomial,
       power_law,
       logistic,
   )

Exponential Decay
-----------------

.. math::

   f(x) = A \cdot e^{-k \cdot x} + c

.. code-block:: python

   from nlsq import fit
   from nlsq.functions import exponential_decay

   # Parameters: amplitude, decay_rate, offset
   popt, pcov = fit(exponential_decay, x, y, p0=[2.0, 0.5, 0.0])
   A, k, c = popt

Gaussian (Normal Distribution)
------------------------------

.. math::

   f(x) = A \cdot e^{-\frac{(x - \mu)^2}{2\sigma^2}} + c

.. code-block:: python

   from nlsq.functions import gaussian

   # Parameters: amplitude, center, width, offset
   popt, pcov = fit(gaussian, x, y, p0=[5.0, 0.0, 1.0, 0.0])
   A, mu, sigma, c = popt

Lorentzian (Cauchy Distribution)
--------------------------------

.. math::

   f(x) = \frac{A \cdot \gamma^2}{(x - x_0)^2 + \gamma^2} + c

.. code-block:: python

   from nlsq.functions import lorentzian

   # Parameters: amplitude, center, width, offset
   popt, pcov = fit(lorentzian, x, y, p0=[5.0, 0.0, 1.0, 0.0])
   A, x0, gamma, c = popt

Power Law
---------

.. math::

   f(x) = A \cdot x^n + c

.. code-block:: python

   from nlsq.functions import power_law

   # Parameters: amplitude, exponent, offset
   popt, pcov = fit(power_law, x, y, p0=[1.0, 2.0, 0.0])
   A, n, c = popt

Logistic Function
-----------------

.. math::

   f(x) = \frac{L}{1 + e^{-k(x - x_0)}} + c

.. code-block:: python

   from nlsq.functions import logistic

   # Parameters: max_value, steepness, midpoint, offset
   popt, pcov = fit(logistic, x, y, p0=[1.0, 1.0, 0.0, 0.0])
   L, k, x0, c = popt

Polynomial
----------

Polynomials of any degree:

.. code-block:: python

   from nlsq.functions import polynomial

   # Quadratic: y = a + b*x + c*x^2
   popt, pcov = fit(lambda x, a, b, c: polynomial(x, [a, b, c]), x, y, p0=[0, 1, 0])

   # Or define directly
   import jax.numpy as jnp


   def quadratic(x, a, b, c):
       return a + b * x + c * x**2


   popt, pcov = fit(quadratic, x, y, p0=[0, 1, 0])

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from nlsq import fit
   from nlsq.functions import gaussian

   # Generate data: Gaussian peak with noise
   np.random.seed(42)
   x = np.linspace(-5, 5, 100)
   y_true = 3.0 * np.exp(-0.5 * ((x - 1.0) / 0.8) ** 2) + 0.5
   y = y_true + 0.2 * np.random.normal(size=len(x))

   # Fit using built-in Gaussian
   popt, pcov = fit(gaussian, x, y, p0=[2.5, 0.5, 1.0, 0.0])

   print("Fitted parameters:")
   print(f"  Amplitude: {popt[0]:.3f} (true: 3.0)")
   print(f"  Center:    {popt[1]:.3f} (true: 1.0)")
   print(f"  Width:     {popt[2]:.3f} (true: 0.8)")
   print(f"  Offset:    {popt[3]:.3f} (true: 0.5)")

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
     - ``logistic``
   * - Power relationship
     - ``power_law``
   * - General trend
     - ``polynomial``

Tips:

- Gaussian peaks are narrower at the base
- Lorentzian peaks have heavier tails
- Use ``exponential_decay`` for radioactive decay, chemical reactions
- Use ``logistic`` for growth curves, dose-response

Next Steps
----------

- :doc:`custom_models` - Create your own models
- :doc:`model_validation` - Verify model correctness
