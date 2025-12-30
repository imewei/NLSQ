nlsq.functions module
======================

.. currentmodule:: nlsq.core.functions

.. automodule:: nlsq.core.functions
   :noindex:

Overview
--------

The ``nlsq.functions`` module provides a library of commonly used fit functions with
automatic parameter estimation. These pre-built functions eliminate the need to write
custom models for common curve fitting tasks.

**New in version 0.1.1**: Complete function library with 10+ pre-built models and automatic
parameter estimation.

Key Features
------------

- **10+ pre-built models** for common curve fitting tasks
- **Automatic initial parameter estimation** from data
- **JAX-optimized implementations** for GPU/TPU acceleration
- **Comprehensive parameter bounds** for robust fitting
- **Detailed documentation** for each function

Available Functions
-------------------

.. autosummary::
   :toctree: generated/

   gaussian
   exponential_decay
   exponential_growth
   sigmoid
   power_law
   polynomial

Usage Examples
--------------

Gaussian Function
~~~~~~~~~~~~~~~~~

Fit a Gaussian (normal distribution) to data:

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.core.functions import gaussian
    import numpy as np

    # Generate synthetic data
    x = np.linspace(-5, 5, 100)
    y_true = gaussian(x, amplitude=10, mean=0, std=1.5)
    y = y_true + np.random.normal(0, 0.5, len(x))

    # Fit with automatic parameter estimation
    popt, pcov = curve_fit(gaussian, x, y)

    print(f"Amplitude: {popt[0]:.2f}")
    print(f"Mean: {popt[1]:.2f}")
    print(f"Std Dev: {popt[2]:.2f}")

Exponential Decay
~~~~~~~~~~~~~~~~~

Fit an exponential decay curve:

.. code-block:: python

    from nlsq.core.functions import exponential_decay

    # Generate decay data
    x = np.linspace(0, 10, 100)
    y_true = exponential_decay(x, amplitude=5, rate=0.5, offset=1)
    y = y_true + np.random.normal(0, 0.2, len(x))

    # Fit with automatic initial parameters
    popt, pcov = curve_fit(exponential_decay, x, y)

    print(f"Amplitude: {popt[0]:.2f}")
    print(f"Decay rate: {popt[1]:.2f}")
    print(f"Offset: {popt[2]:.2f}")

Sigmoid Function
~~~~~~~~~~~~~~~~

Fit a sigmoid (logistic) curve:

.. code-block:: python

    from nlsq.core.functions import sigmoid

    # Generate sigmoid data
    x = np.linspace(-10, 10, 100)
    y_true = sigmoid(x, L=10, k=1, x0=0)
    y = y_true + np.random.normal(0, 0.5, len(x))

    # Fit sigmoid
    popt, pcov = curve_fit(sigmoid, x, y)

    print(f"Maximum value: {popt[0]:.2f}")
    print(f"Growth rate: {popt[1]:.2f}")
    print(f"Midpoint: {popt[2]:.2f}")

Power Law
~~~~~~~~~

Fit a power law relationship:

.. code-block:: python

    from nlsq.core.functions import power_law

    # Generate power law data
    x = np.linspace(1, 100, 50)
    y_true = power_law(x, scale=2, exponent=0.5)
    y = y_true + np.random.normal(0, 0.1, len(x))

    # Fit power law
    popt, pcov = curve_fit(power_law, x, y)

    print(f"Scale: {popt[0]:.2f}")
    print(f"Exponent: {popt[1]:.2f}")

Sinusoidal Function
~~~~~~~~~~~~~~~~~~~

Fit a sinusoidal (periodic) function:

.. code-block:: python

    from nlsq.core.functions import sinusoidal

    # Generate periodic data
    x = np.linspace(0, 4 * np.pi, 100)
    y_true = sinusoidal(x, amplitude=3, frequency=2, phase=0, offset=1)
    y = y_true + np.random.normal(0, 0.2, len(x))

    # Fit sinusoid
    popt, pcov = curve_fit(sinusoidal, x, y)

    print(f"Amplitude: {popt[0]:.2f}")
    print(f"Frequency: {popt[1]:.2f}")
    print(f"Phase: {popt[2]:.2f}")
    print(f"Offset: {popt[3]:.2f}")

Automatic Parameter Estimation
-------------------------------

All functions in this module include intelligent parameter estimation:

.. code-block:: python

    from nlsq.core.functions import gaussian

    # Fit without providing initial parameters
    # The function automatically estimates reasonable starting values
    popt, pcov = curve_fit(gaussian, x, y)

    # Or provide custom initial parameters if needed
    popt, pcov = curve_fit(gaussian, x, y, p0=[10, 0, 1])

Function Parameters
-------------------

Each function has well-defined parameters with physical meaning:

**gaussian(x, amplitude, mean, std)**
    - ``amplitude``: Height of the peak
    - ``mean``: Center of the distribution
    - ``std``: Standard deviation (width)

**exponential_decay(x, amplitude, rate, offset)**
    - ``amplitude``: Initial value
    - ``rate``: Decay rate (positive)
    - ``offset``: Asymptotic value

**sigmoid(x, L, k, x0)**
    - ``L``: Maximum value (carrying capacity)
    - ``k``: Growth rate
    - ``x0``: Midpoint (inflection point)

Interactive Notebooks
---------------------

- `Function Library Demo <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/05_feature_demos/function_library_demo.ipynb>`_ (20 min) - Pre-built models and automatic parameter estimation

See Also
--------

- :doc:`../tutorials/01_first_fit` : Getting started tutorial
- :doc:`nlsq.minpack` : Main curve fitting API
- :doc:`nlsq.bound_inference` : Automatic bounds detection
