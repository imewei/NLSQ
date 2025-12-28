Model Functions Reference
=========================

NLSQ provides built-in model functions for common curve fitting scenarios.

.. module:: nlsq.functions

Using Built-in Functions
------------------------

Import from ``nlsq.functions`` or directly from ``nlsq``:

.. code-block:: python

   from nlsq.core.functions import gaussian, exponential_decay

   # or
   from nlsq import gaussian, exponential_decay

All functions are JAX-compatible and JIT-compilable.

Peak Functions
--------------

gaussian
~~~~~~~~

.. autofunction:: nlsq.functions.gaussian
   :no-index:

Gaussian peak function.

**Formula:**

.. math::

   f(x) = a \cdot \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)

**Parameters:**

- ``x``: Independent variable
- ``a``: Amplitude (peak height)
- ``mu``: Center position
- ``sigma``: Standard deviation (width)

**Example:**

.. code-block:: python

   from nlsq import fit
   from nlsq.core.functions import gaussian

   popt, pcov = fit(gaussian, x, y, p0=[1.0, 0.0, 1.0])
   amplitude, center, width = popt

lorentzian
~~~~~~~~~~

.. autofunction:: nlsq.functions.lorentzian
   :no-index:

Lorentzian (Cauchy) peak function.

**Formula:**

.. math::

   f(x) = \frac{a \cdot \gamma^2}{(x - x_0)^2 + \gamma^2}

**Parameters:**

- ``x``: Independent variable
- ``a``: Amplitude
- ``x0``: Center position
- ``gamma``: Half-width at half-maximum (HWHM)

voigt
~~~~~

.. autofunction:: nlsq.functions.voigt
   :no-index:

Voigt profile (convolution of Gaussian and Lorentzian).

Exponential Functions
---------------------

exponential_decay
~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.functions.exponential_decay
   :no-index:

Single exponential decay.

**Formula:**

.. math::

   f(t) = a \cdot \exp\left(-\frac{t}{\tau}\right) + c

**Parameters:**

- ``t``: Time (independent variable)
- ``a``: Initial amplitude
- ``tau``: Decay time constant
- ``c``: Offset (baseline)

**Example:**

.. code-block:: python

   from nlsq import fit
   from nlsq.core.functions import exponential_decay

   popt, pcov = fit(
       exponential_decay,
       time,
       signal,
       p0=[1.0, 10.0, 0.0],
       bounds=([0, 0, -0.1], [10, 100, 0.1]),
   )
   amplitude, tau, offset = popt

exponential_growth
~~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.functions.exponential_growth
   :no-index:

Exponential growth function.

**Formula:**

.. math::

   f(t) = a \cdot \exp\left(\frac{t}{\tau}\right) + c

double_exponential
~~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.functions.double_exponential
   :no-index:

Biexponential decay (sum of two exponentials).

**Formula:**

.. math::

   f(t) = a_1 \cdot \exp\left(-\frac{t}{\tau_1}\right) + a_2 \cdot \exp\left(-\frac{t}{\tau_2}\right) + c

Sigmoid Functions
-----------------

sigmoid
~~~~~~~

.. autofunction:: nlsq.functions.sigmoid
   :no-index:

Logistic sigmoid function.

**Formula:**

.. math::

   f(x) = \frac{L}{1 + \exp(-k(x - x_0))}

**Parameters:**

- ``x``: Independent variable
- ``L``: Maximum value (upper asymptote)
- ``k``: Steepness (growth rate)
- ``x0``: Midpoint (x value at half maximum)

**Example:**

.. code-block:: python

   from nlsq import fit
   from nlsq.core.functions import sigmoid

   popt, pcov = fit(sigmoid, x, y, p0=[1.0, 1.0, 0.0])
   maximum, steepness, midpoint = popt

hill
~~~~

.. autofunction:: nlsq.functions.hill
   :no-index:

Hill equation for dose-response curves.

**Formula:**

.. math::

   f(x) = \frac{V_{max} \cdot x^n}{K^n + x^n}

Power Functions
---------------

power_law
~~~~~~~~~

.. autofunction:: nlsq.functions.power_law
   :no-index:

Power law function.

**Formula:**

.. math::

   f(x) = a \cdot x^b

**Parameters:**

- ``x``: Independent variable
- ``a``: Coefficient
- ``b``: Exponent

polynomial
~~~~~~~~~~

.. autofunction:: nlsq.functions.polynomial
   :no-index:

Polynomial function of arbitrary degree.

**Formula:**

.. math::

   f(x) = \sum_{i=0}^{n} a_i x^i

**Example:**

.. code-block:: python

   from nlsq.core.functions import polynomial


   # Quadratic: a + b*x + c*x^2
   def quadratic(x, a, b, c):
       return polynomial(x, a, b, c)

Special Functions
-----------------

sine
~~~~

.. autofunction:: nlsq.functions.sine
   :no-index:

Sinusoidal function.

**Formula:**

.. math::

   f(x) = a \cdot \sin(2\pi f x + \phi) + c

**Parameters:**

- ``x``: Independent variable
- ``a``: Amplitude
- ``f``: Frequency
- ``phi``: Phase
- ``c``: Offset

damped_sine
~~~~~~~~~~~

.. autofunction:: nlsq.functions.damped_sine
   :no-index:

Damped sinusoidal oscillation.

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

- :doc:`/tutorials/01_first_fit` - Basic usage examples
- :doc:`/explanation/jax_autodiff` - Why JAX is required
- :doc:`/howto/choose_model` - Model selection guide
