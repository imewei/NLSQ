Basic Data
==========

This tutorial covers how to load and prepare data for curve fitting.

Data Requirements
-----------------

NLSQ requires:

- **x**: Independent variable (1D array)
- **y**: Dependent variable (1D array, same length as x)
- **p0**: Initial parameter guess (list or array)

.. code-block:: python

   import numpy as np
   from nlsq import fit

   x = np.array([0, 1, 2, 3, 4])
   y = np.array([1.0, 0.6, 0.4, 0.2, 0.1])
   p0 = [1.0, 0.5]

   popt, pcov = fit(model, x, y, p0=p0)

Loading from Files
------------------

**CSV files:**

.. code-block:: python

   import numpy as np

   # Simple CSV with columns: x, y
   data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
   x, y = data[:, 0], data[:, 1]

   # With pandas
   import pandas as pd

   df = pd.read_csv("data.csv")
   x, y = df["x"].values, df["y"].values

**NumPy files:**

.. code-block:: python

   data = np.load("data.npz")
   x, y = data["x"], data["y"]

**HDF5 files:**

.. code-block:: python

   import h5py

   with h5py.File("data.h5", "r") as f:
       x = f["x"][:]
       y = f["y"][:]

Data Types
----------

NLSQ accepts various array types:

.. code-block:: python

   # Python lists (converted internally)
   popt, pcov = fit(model, [0, 1, 2], [1.0, 0.6, 0.4], p0=[1, 0.5])

   # NumPy arrays (recommended)
   x = np.array([0, 1, 2])
   y = np.array([1.0, 0.6, 0.4])

   # JAX arrays
   import jax.numpy as jnp

   x = jnp.array([0, 1, 2])
   y = jnp.array([1.0, 0.6, 0.4])

Float64 is used internally for numerical precision.

Handling Missing Data
---------------------

Remove NaN values before fitting:

.. code-block:: python

   # Method 1: Boolean indexing
   mask = ~(np.isnan(x) | np.isnan(y))
   x_clean = x[mask]
   y_clean = y[mask]

   # Method 2: Use nan_policy parameter
   popt, pcov = fit(model, x, y, p0=[...], nan_policy="omit")

Data Scaling
------------

For best numerical stability, scale data if values are very large or small:

.. code-block:: python

   # Scale x to [0, 1] range
   x_min, x_max = x.min(), x.max()
   x_scaled = (x - x_min) / (x_max - x_min)

   # Scale y to reasonable range
   y_mean = y.mean()
   y_scaled = y / y_mean

   # Fit scaled data
   popt, pcov = fit(model, x_scaled, y_scaled, p0=[...])

   # Adjust parameters back (depends on model)

NLSQ can also automatically rescale data:

.. code-block:: python

   popt, pcov = fit(model, x, y, p0=[...], rescale_data=True)

Multi-dimensional X
-------------------

For models with multiple independent variables:

.. code-block:: python

   import jax.numpy as jnp


   def surface(xy, a, b, c):
       """2D surface: z = a*x + b*y + c"""
       x, y = xy
       return a * x + b * y + c


   # Pack x, y into tuple
   xdata = (x_array, y_array)
   popt, pcov = fit(surface, xdata, z_array, p0=[1, 1, 0])

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import fit

   # Load data
   df = pd.read_csv("experiment.csv")
   x = df["time"].values
   y = df["signal"].values

   # Remove any invalid data
   mask = np.isfinite(x) & np.isfinite(y)
   x, y = x[mask], y[mask]


   # Define model
   def exponential(x, A, k, c):
       return A * jnp.exp(-k * x) + c


   # Initial guess based on data inspection
   p0 = [
       y.max() - y.min(),  # Amplitude
       1.0 / (x.max() / 3),  # Rough decay rate
       y.min(),
   ]  # Offset

   # Fit
   popt, pcov = fit(exponential, x, y, p0=p0)

   A, k, c = popt
   print(f"Amplitude: {A:.3f}")
   print(f"Decay rate: {k:.3f}")
   print(f"Offset: {c:.3f}")

Next Steps
----------

- :doc:`uncertainties` - Add measurement errors
- :doc:`bounds` - Constrain parameters
