nlsq.bound\_inference module
==============================

.. currentmodule:: nlsq.precision.bound_inference

.. automodule:: nlsq.precision.bound_inference
   :noindex:

Overview
--------

The ``nlsq.bound_inference`` module provides intelligent automatic inference of parameter
bounds from data. Instead of manually specifying bounds, the module analyzes your data
and model to determine reasonable constraints automatically.

**New in version 0.1.1**: Complete bounds inference system with smart defaults.

Key Features
------------

- **Automatic bounds detection** from the data ranges and the initial guess ``p0``
- **Positivity enforcement** when ``p0`` is positive
- **Merge with user bounds**, where user-specified bounds take precedence
- **Multi-start bounds** with a wider default safety factor
- **Quality analysis** of bounds relative to ``p0``

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BoundsInference

Functions
---------

.. autosummary::
   :toctree: generated/

   infer_bounds
   infer_bounds_for_multistart
   merge_bounds
   analyze_bounds_quality

Usage Examples
--------------

Basic Automatic Bounds
~~~~~~~~~~~~~~~~~~~~~~~

Let NLSQ infer parameter bounds automatically:

.. code-block:: python

    import jax.numpy as jnp
    import numpy as np

    from nlsq import curve_fit


    def exponential(x, a, b):
        return a * jnp.exp(-b * x)


    x = np.linspace(0, 10, 200)
    y = 2.5 * np.exp(-0.5 * x)

    # Infer bounds from the data and p0, then fit inside them
    popt, pcov = curve_fit(
        exponential,
        x,
        y,
        p0=[1.0, 0.5],
        auto_bounds=True,
        bounds_safety_factor=10.0,  # larger = more conservative bounds
    )

Explicit Bounds Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~

Manually infer and inspect bounds before fitting. ``infer_bounds`` returns a
``(lower, upper)`` tuple of arrays:

.. code-block:: python

    from nlsq.precision.bound_inference import infer_bounds

    lower, upper = infer_bounds(x, y, p0=[1.0, 0.5])

    print(f"Lower bounds: {lower}")
    print(f"Upper bounds: {upper}")

    # Use inferred bounds
    popt, pcov = curve_fit(exponential, x, y, p0=[1.0, 0.5], bounds=(lower, upper))

Merging with User Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

Combine automatic inference with user-specified constraints:

.. code-block:: python

    from nlsq.precision.bound_inference import infer_bounds, merge_bounds

    # Infer bounds
    auto_bounds = infer_bounds(x, y, p0=[1.0, 0.5])

    # User knows that parameter 'a' must be between 0 and 10
    user_bounds = ([0, -np.inf], [10, np.inf])

    # Merge: user bounds take precedence, inferred bounds fill the rest
    final_bounds = merge_bounds(auto_bounds, user_bounds)

    popt, pcov = curve_fit(exponential, x, y, p0=[1.0, 0.5], bounds=final_bounds)

Controlling the Inference
~~~~~~~~~~~~~~~~~~~~~~~~~

``BoundsInference`` is the class behind ``infer_bounds``. It exposes the data
statistics it used:

.. code-block:: python

    from nlsq.precision.bound_inference import BoundsInference

    inference = BoundsInference(
        x,
        y,
        p0=[1.0, 0.5],
        safety_factor=20.0,  # wider bounds around p0 and the data
        enforce_positivity=True,  # lower bounds >= 0 when p0 > 0
    )
    lower, upper = inference.infer()

    print(inference.y_min, inference.y_max, inference.x_range)

Multi-Start Bounds
~~~~~~~~~~~~~~~~~~

``infer_bounds_for_multistart`` uses a larger default safety factor (20) so that
global exploration can cover a broader region, and accepts user bounds directly:

.. code-block:: python

    from nlsq.precision.bound_inference import infer_bounds_for_multistart

    lower, upper = infer_bounds_for_multistart(
        x, y, p0=[1.0, 0.5], user_bounds=([0, 0], [10, np.inf])
    )

Bounds Quality
~~~~~~~~~~~~~~

Check how bounds relate to the initial guess:

.. code-block:: python

    from nlsq.precision.bound_inference import analyze_bounds_quality

    analysis = analyze_bounds_quality((lower, upper), p0=[1.0, 0.5])

    print(analysis["is_feasible"])  # is p0 inside the bounds?
    print(analysis["avg_bound_ratio"])
    print(analysis["tight_parameters"], analysis["loose_parameters"])

Inference Strategies
--------------------

Each parameter's bounds are the union of two heuristics:

Parameter Magnitude Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the initial guess (``safety_factor`` defaults to 10):

- **Range**: between ``p0 / safety_factor`` and ``p0 * safety_factor``
- **Zero p0**: symmetric around zero, scaled by the range of ``y``

Data Range Strategy
~~~~~~~~~~~~~~~~~~~

Based on position in the parameter vector:

- **First parameter** (amplitude-like): ``[min(y) - range(y), max(y) + range(y)]``
- **Second parameter** (rate-like): ``[0.1 / range(x), 10 / range(x)]``
- **Other parameters** (offsets): ``[min(y) - range(y), max(y) + range(y)]``

Positivity
~~~~~~~~~~

With ``enforce_positivity=True`` and a positive ``p0`` entry, the lower bound is
clipped at 0 (and at ``1e-10`` for the second, rate-like parameter).

See Also
--------

- :doc:`../reference/configuration` : Configuration reference
- :doc:`nlsq.minpack` : Main curve fitting API
- :doc:`nlsq.validators` : Input validation
