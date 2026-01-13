Your First Curve Fit
====================

In this tutorial, you'll fit an exponential decay model to data using NLSQ.

What You'll Learn
-----------------

- How to define a model function
- How to use ``fit()`` to fit data
- How to extract fitted parameters

Step 1: Import NLSQ
-------------------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp
   import numpy as np

.. important::

   Model functions must use ``jax.numpy`` (imported as ``jnp``), not regular
   ``numpy``. This enables automatic differentiation and GPU acceleration.

Step 2: Create Sample Data
--------------------------

Let's create synthetic data with known parameters:

.. code-block:: python

   # True parameters we want to recover
   A_true = 2.5
   k_true = 0.5

   # Generate data with noise
   np.random.seed(42)
   x = np.linspace(0, 10, 50)
   y_true = A_true * np.exp(-k_true * x)
   y = y_true + 0.1 * np.random.normal(size=len(x))

Step 3: Define Your Model
-------------------------

The model function describes the mathematical relationship:

.. code-block:: python

   def exponential_decay(x, A, k):
       """Exponential decay: y = A * exp(-k * x)"""
       return A * jnp.exp(-k * x)

**Model function rules:**

1. First argument is always ``x`` (independent variable)
2. Subsequent arguments are fit parameters
3. Use ``jnp`` for mathematical operations

Step 4: Fit the Model
---------------------

Now fit the model to data:

.. code-block:: python

   # Fit with initial guess
   popt, pcov = fit(exponential_decay, x, y, p0=[1.0, 0.3])

   # Extract fitted parameters
   A_fit, k_fit = popt

   print(f"Fitted parameters:")
   print(f"  A = {A_fit:.4f} (true: {A_true})")
   print(f"  k = {k_fit:.4f} (true: {k_true})")

Expected output:

.. code-block:: text

   Fitted parameters:
     A = 2.4892 (true: 2.5)
     k = 0.4987 (true: 0.5)

Step 5: Visualize Results
-------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6))
   plt.scatter(x, y, label="Data", alpha=0.7)

   x_smooth = np.linspace(0, 10, 200)
   y_fit = exponential_decay(x_smooth, *popt)
   plt.plot(x_smooth, y_fit, "r-", label="Fitted curve", linewidth=2)

   plt.xlabel("x")
   plt.ylabel("y")
   plt.title("Exponential Decay Fit")
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Complete Example
----------------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp
   import numpy as np


   # Define model
   def exponential_decay(x, A, k):
       return A * jnp.exp(-k * x)


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 50)
   y = 2.5 * np.exp(-0.5 * x) + 0.1 * np.random.normal(size=len(x))

   # Fit
   popt, pcov = fit(exponential_decay, x, y, p0=[1.0, 0.3])

   print(f"A = {popt[0]:.4f}")
   print(f"k = {popt[1]:.4f}")

Key Takeaways
-------------

1. **fit()** is the main entry point for curve fitting
2. **Model functions** must use ``jax.numpy`` for math
3. **p0** provides the initial guess for parameters
4. **Returns** ``popt`` (fitted parameters) and ``pcov`` (covariance matrix)

Initial Guess Tips
------------------

A good initial guess helps the optimizer converge:

- **Exponential decay**: Start with reasonable amplitude and rate
- **Gaussian**: Use data range for center, width from peak shape
- **Polynomial**: Start with zeros or small values

If unsure, try ``workflow='auto_global'`` with bounds for robust fitting.

Common Mistakes
---------------

**Using numpy instead of jax.numpy:**

.. code-block:: python

   # Wrong - will not work correctly
   def model(x, a, b):
       return a * np.exp(-b * x)


   # Correct
   def model(x, a, b):
       return a * jnp.exp(-b * x)

**Missing initial guess:**

.. code-block:: python

   # Error: p0 is required
   popt, pcov = fit(model, x, y)

   # Correct
   popt, pcov = fit(model, x, y, p0=[1.0, 0.5])

Next Steps
----------

Continue to :doc:`understanding_results` to learn how to interpret
``pcov`` and calculate parameter uncertainties.
