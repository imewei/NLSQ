Custom Models
=============

Learn how to write your own model functions for curve fitting.

Basic Structure
---------------

A model function must:

1. Take ``x`` as the first argument
2. Take fit parameters as subsequent arguments
3. Return the predicted ``y`` values
4. Use ``jax.numpy`` for mathematical operations

.. code-block:: python

   import jax.numpy as jnp


   def my_model(x, param1, param2, param3):
       """Model description."""
       return param1 * jnp.exp(-param2 * x) + param3

Example: Damped Oscillation
---------------------------

.. code-block:: python

   import jax.numpy as jnp
   from nlsq import fit


   def damped_oscillation(x, amplitude, decay, frequency, phase, offset):
       """Damped sinusoidal oscillation.

       y = A * exp(-gamma * x) * cos(omega * x + phi) + c
       """
       return amplitude * jnp.exp(-decay * x) * jnp.cos(frequency * x + phase) + offset


   # Fit
   p0 = [1.0, 0.1, 2.0, 0.0, 0.0]
   popt, pcov = fit(damped_oscillation, x, y, p0=p0)

Example: Sum of Gaussians
-------------------------

.. code-block:: python

   import jax.numpy as jnp


   def double_gaussian(x, a1, c1, w1, a2, c2, w2, offset):
       """Sum of two Gaussian peaks."""
       g1 = a1 * jnp.exp(-0.5 * ((x - c1) / w1) ** 2)
       g2 = a2 * jnp.exp(-0.5 * ((x - c2) / w2) ** 2)
       return g1 + g2 + offset


   # 7 parameters: 2 peaks (3 each) + offset
   p0 = [1, 2, 0.5, 1, 5, 0.5, 0]
   bounds = ([0, 0, 0.1, 0, 0, 0.1, -1], [10, 10, 3, 10, 10, 3, 1])

   popt, pcov = fit(double_gaussian, x, y, p0=p0, bounds=bounds)

Example: Michaelis-Menten Kinetics
----------------------------------

.. code-block:: python

   import jax.numpy as jnp


   def michaelis_menten(S, Vmax, Km):
       """Enzyme kinetics: v = Vmax * S / (Km + S)"""
       return Vmax * S / (Km + S)


   # Fit enzyme kinetics data
   popt, pcov = fit(michaelis_menten, substrate_conc, reaction_rate, p0=[100.0, 10.0])
   Vmax, Km = popt

Using Constants
---------------

If your model needs fixed constants, use closures:

.. code-block:: python

   import jax.numpy as jnp


   def create_model(wavelength):
       """Create a model with fixed wavelength."""

       def model(x, amplitude, phase):
           k = 2 * jnp.pi / wavelength
           return amplitude * jnp.sin(k * x + phase)

       return model


   # Create model with wavelength = 5.0
   wave_model = create_model(wavelength=5.0)
   popt, pcov = fit(wave_model, x, y, p0=[1.0, 0.0])

Array Parameters
----------------

For models with array-like parameters:

.. code-block:: python

   import jax.numpy as jnp


   def sum_of_exponentials(x, *params):
       """Sum of N exponential terms.

       params = [A1, k1, A2, k2, ..., offset]
       """
       n_terms = (len(params) - 1) // 2
       result = jnp.zeros_like(x)

       for i in range(n_terms):
           A = params[2 * i]
           k = params[2 * i + 1]
           result = result + A * jnp.exp(-k * x)

       return result + params[-1]  # Add offset


   # Two exponential terms + offset = 5 parameters
   popt, pcov = fit(sum_of_exponentials, x, y, p0=[1, 0.1, 0.5, 0.5, 0])

Common Patterns
---------------

**Avoid divisions by zero:**

.. code-block:: python

   def safe_model(x, a, b):
       # Bad: may divide by zero
       # return a / (x - b)

       # Good: add small epsilon
       return a / (x - b + 1e-10)

**Use stable functions:**

.. code-block:: python

   # Numerically stable log-sum-exp
   def log_sum_exp(x, a, b):
       max_val = jnp.maximum(a, b)
       return max_val + jnp.log(jnp.exp(a - max_val) + jnp.exp(b - max_val))

**Vectorized operations:**

.. code-block:: python

   # Good: vectorized
   def good_model(x, a, b):
       return a * jnp.exp(-b * x)


   # Bad: Python loops (slow)
   def bad_model(x, a, b):
       result = []
       for xi in x:
           result.append(a * jnp.exp(-b * xi))
       return jnp.array(result)

Testing Your Model
------------------

Before fitting, verify your model works:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp


   def my_model(x, a, b):
       return a * jnp.exp(-b * x)


   # Test with sample data
   x_test = np.linspace(0, 10, 50)
   y_test = my_model(x_test, 2.0, 0.5)

   print(f"Output shape: {y_test.shape}")
   print(f"Output type: {type(y_test)}")
   print(f"Sample values: {y_test[:5]}")

Next Steps
----------

- :doc:`model_validation` - Validate model correctness
- :doc:`../data_handling/bounds` - Constrain parameters
