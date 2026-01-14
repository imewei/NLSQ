DataPreprocessor
================

.. versionadded:: 0.6.4

The ``DataPreprocessor`` handles input validation, type conversion, and
data cleaning before optimization.

Basic Usage
-----------

.. code-block:: python

   from nlsq.core.orchestration import DataPreprocessor
   import jax.numpy as jnp


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   preprocessor = DataPreprocessor()
   preprocessed = preprocessor.preprocess(
       f=model, xdata=x, ydata=y, sigma=None, check_finite=True
   )

   # Access preprocessed data
   x_clean = preprocessed.xdata
   y_clean = preprocessed.ydata
   n_points = preprocessed.n_points

PreprocessedData
----------------

The ``preprocess()`` method returns a ``PreprocessedData`` object:

.. code-block:: python

   @dataclass
   class PreprocessedData:
       xdata: jnp.ndarray  # Preprocessed x data
       ydata: jnp.ndarray  # Preprocessed y data
       sigma: jnp.ndarray | None  # Uncertainties
       n_points: int  # Number of data points
       is_padded: bool  # Was data padded?
       has_nans_removed: bool  # Were NaNs removed?
       original_shape: tuple  # Original data shape

preprocess() Parameters
-----------------------

.. code-block:: python

   preprocessed = preprocessor.preprocess(
       f,  # Model function
       xdata,  # Independent variable
       ydata,  # Dependent variable
       sigma=None,  # Measurement uncertainties
       absolute_sigma=False,  # Interpret sigma as absolute
       check_finite=True,  # Check for inf/NaN
       nan_policy="raise",  # 'raise', 'omit', 'propagate'
       stability_check=False,  # Run stability checks
       **kwargs
   )

NaN Handling
------------

**nan_policy='raise' (default):**

.. code-block:: python

   # Raises ValueError if NaN found
   preprocessor.preprocess(f, x_with_nan, y, nan_policy="raise")

**nan_policy='omit':**

.. code-block:: python

   # Removes NaN values
   preprocessed = preprocessor.preprocess(f, x_with_nan, y, nan_policy="omit")
   print(f"Points after NaN removal: {preprocessed.n_points}")
   print(f"NaNs removed: {preprocessed.has_nans_removed}")

**nan_policy='propagate':**

.. code-block:: python

   # Passes NaN through (may cause fitting issues)
   preprocessed = preprocessor.preprocess(f, x_with_nan, y, nan_policy="propagate")

Sigma Validation
----------------

.. code-block:: python

   # Validate sigma has correct shape
   preprocessor.validate_sigma(sigma, ydata.shape)

   # With absolute_sigma
   preprocessed = preprocessor.preprocess(
       f, x, y, sigma=measurement_errors, absolute_sigma=True
   )

Type Conversion
---------------

Input types are converted to JAX arrays:

.. code-block:: python

   import numpy as np

   # NumPy arrays → JAX arrays
   x_np = np.array([1, 2, 3])
   preprocessed = preprocessor.preprocess(f, x_np, y_np)
   assert isinstance(preprocessed.xdata, jnp.ndarray)

   # Python lists → JAX arrays
   preprocessed = preprocessor.preprocess(f, [1, 2, 3], [1.0, 0.5, 0.3])

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq.core.orchestration import DataPreprocessor


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Create data with some NaN values
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y = 2.5 * np.exp(-0.5 * x) + 0.3 + 0.1 * np.random.randn(100)
   y[5] = np.nan  # Add NaN
   y[50] = np.nan

   sigma = 0.1 * np.ones(100)

   # Preprocess
   preprocessor = DataPreprocessor()
   preprocessed = preprocessor.preprocess(
       f=model, xdata=x, ydata=y, sigma=sigma, nan_policy="omit", check_finite=True
   )

   print(f"Original points: 100")
   print(f"After preprocessing: {preprocessed.n_points}")
   print(f"NaNs removed: {preprocessed.has_nans_removed}")
   print(f"Data type: {type(preprocessed.xdata)}")

Next Steps
----------

- :doc:`optimization_selector` - Optimization configuration
- :doc:`../factories_di/dependency_injection` - Custom preprocessing
