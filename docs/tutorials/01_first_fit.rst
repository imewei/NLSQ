Tutorial 1: Your First Curve Fit
=================================

In this tutorial, you'll learn the fundamentals of curve fitting with NLSQ.
By the end, you'll be able to fit a mathematical model to data and extract
parameter values.

What You'll Learn
-----------------

- What curve fitting does
- How to define a model function
- How to use ``curve_fit()``
- How to interpret the results

Prerequisites
-------------

- Python 3.12+ installed
- NLSQ installed (``pip install nlsq``)
- Basic Python knowledge

Step 1: Install NLSQ
--------------------

If you haven't already, install NLSQ:

.. code-block:: bash

   pip install nlsq

Verify the installation:

.. code-block:: python

   import nlsq

   print(f"NLSQ version: {nlsq.__version__}")

Step 2: Understand the Problem
------------------------------

Curve fitting answers this question: *Given some data points, what parameters
make a mathematical model best match the data?*

For example, if you have data that looks like exponential decay:

.. code-block:: text

   y = A * exp(-k * x)

You want to find the values of ``A`` (amplitude) and ``k`` (decay rate)
that make the curve pass as close as possible to your data points.

Step 3: Generate Sample Data
----------------------------

Let's create some synthetic data that follows an exponential decay with
added noise:

.. code-block:: python

   import numpy as np

   # True parameters we want to recover
   A_true = 2.5
   k_true = 0.5

   # Generate x values
   x = np.linspace(0, 10, 50)

   # Generate y values with noise
   np.random.seed(42)  # For reproducibility
   y_true = A_true * np.exp(-k_true * x)
   y = y_true + 0.1 * np.random.normal(size=len(x))

   print(f"Data shape: x={x.shape}, y={y.shape}")
   print(f"True parameters: A={A_true}, k={k_true}")

Step 4: Define Your Model Function
----------------------------------

The model function describes the mathematical relationship between x and y.
It must:

1. Take ``x`` as the first argument
2. Take the fit parameters as subsequent arguments
3. Use ``jax.numpy`` for mathematical operations (for GPU acceleration)

.. code-block:: python

   import jax.numpy as jnp


   def exponential_decay(x, A, k):
       """Exponential decay model: y = A * exp(-k * x)"""
       return A * jnp.exp(-k * x)

.. important::

   Always use ``jax.numpy`` (imported as ``jnp``) inside model functions,
   not regular ``numpy``. This enables automatic differentiation and GPU
   acceleration.

Step 5: Fit the Model
---------------------

Now we can fit the model to our data:

.. code-block:: python

   from nlsq import curve_fit

   # Fit the model
   popt, pcov = curve_fit(exponential_decay, x, y)

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

The fitted values are close to the true values, despite the noise in the data!

Step 6: Visualize the Results
-----------------------------

Let's plot the data and the fitted curve:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Create the plot
   plt.figure(figsize=(10, 6))

   # Plot data points
   plt.scatter(x, y, label="Data", alpha=0.7)

   # Plot fitted curve
   x_smooth = np.linspace(0, 10, 200)
   y_fit = exponential_decay(x_smooth, *popt)
   plt.plot(x_smooth, y_fit, "r-", label="Fitted curve", linewidth=2)

   # Plot true curve for comparison
   y_true_smooth = A_true * np.exp(-k_true * x_smooth)
   plt.plot(x_smooth, y_true_smooth, "g--", label="True curve", linewidth=2)

   plt.xlabel("x")
   plt.ylabel("y")
   plt.title("Exponential Decay Fit")
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Complete Example
----------------

Here's the complete code in one block:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit


   # 1. Define model function
   def exponential_decay(x, A, k):
       return A * jnp.exp(-k * x)


   # 2. Generate sample data
   np.random.seed(42)
   x = np.linspace(0, 10, 50)
   y = 2.5 * np.exp(-0.5 * x) + 0.1 * np.random.normal(size=len(x))

   # 3. Fit the model
   popt, pcov = curve_fit(exponential_decay, x, y)

   # 4. Print results
   print(f"Fitted A = {popt[0]:.4f}")
   print(f"Fitted k = {popt[1]:.4f}")

Key Takeaways
-------------

1. **curve_fit** takes a model function, x data, and y data
2. **Model functions** must use ``jax.numpy`` for math operations
3. **Returns** ``popt`` (optimal parameters) and ``pcov`` (covariance matrix)
4. The fit finds parameters that minimize the sum of squared residuals

Common Mistakes to Avoid
------------------------

**Using numpy instead of jax.numpy:**

.. code-block:: python

   # Wrong - will cause errors
   def model(x, a, b):
       return a * np.exp(-b * x)


   # Correct
   def model(x, a, b):
       return a * jnp.exp(-b * x)

**Forgetting to import jax.numpy:**

.. code-block:: python

   # Add this at the top of your script
   import jax.numpy as jnp

Next Steps
----------

In :doc:`02_understanding_results`, you'll learn how to:

- Interpret the covariance matrix ``pcov``
- Calculate parameter uncertainties
- Assess the quality of your fit

.. seealso::

   - :doc:`/howto/migrate_from_scipy` - If you're coming from SciPy
   - :doc:`/explanation/how_fitting_works` - Understand the algorithm
