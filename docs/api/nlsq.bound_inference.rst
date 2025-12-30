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

- **Automatic bounds detection** from data characteristics
- **Model-aware inference** for common function types
- **Physical constraints** enforcement (positivity, ranges)
- **Data-driven estimation** using statistics
- **Merge with user bounds** for hybrid constraints
- **Validation** of inferred bounds

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
   merge_bounds

Usage Examples
--------------

Basic Automatic Bounds
~~~~~~~~~~~~~~~~~~~~~~~

Let NLSQ infer parameter bounds automatically:

.. code-block:: python

    from nlsq import curve_fit
    import jax.numpy as jnp


    def exponential(x, a, b):
        return a * jnp.exp(-b * x)


    # Enable automatic bounds inference
    result = curve_fit(
        exponential, x, y, p0=[1.0, 0.5], auto_bounds=True  # Infer bounds automatically
    )

    # Access inferred bounds
    print(f"Inferred bounds: {result.bounds_used}")

Explicit Bounds Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~

Manually infer and inspect bounds before fitting:

.. code-block:: python

    from nlsq.bound_inference import infer_bounds

    # Infer bounds from data
    bounds = infer_bounds(exponential, x, y, p0=[1.0, 0.5])

    print(f"Lower bounds: {bounds.lower}")
    print(f"Upper bounds: {bounds.upper}")
    print(f"Confidence: {bounds.confidence}")

    # Use inferred bounds
    result = curve_fit(
        exponential, x, y, p0=[1.0, 0.5], bounds=(bounds.lower, bounds.upper)
    )

Merging with User Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

Combine automatic inference with user-specified constraints:

.. code-block:: python

    from nlsq.bound_inference import infer_bounds, merge_bounds

    # Infer bounds
    auto_bounds = infer_bounds(exponential, x, y, p0=[1.0, 0.5])

    # User knows that parameter 'a' must be between 0 and 10
    user_bounds = ([0, -np.inf], [10, np.inf])

    # Merge: use user bounds where specified, auto-inferred elsewhere
    final_bounds = merge_bounds(auto_bounds, user_bounds)

    result = curve_fit(exponential, x, y, p0=[1.0, 0.5], bounds=final_bounds)

Model-Specific Inference
~~~~~~~~~~~~~~~~~~~~~~~~

For known model types, inference uses domain knowledge:

.. code-block:: python

    from nlsq.core.functions import gaussian
    from nlsq.bound_inference import infer_bounds

    # For Gaussian, inference knows:
    # - Amplitude should be ~ max(y)
    # - Mean should be in range of x
    # - Std should be positive and ~ range(x)
    bounds = infer_bounds(gaussian, x, y, p0=None, model_type="gaussian")

    print(f"Gaussian-specific bounds: {bounds}")

Physical Constraints
~~~~~~~~~~~~~~~~~~~~

Enforce physical constraints (positivity, ranges):

.. code-block:: python

    from nlsq.bound_inference import BoundsInference

    inference = BoundsInference(
        enforce_positivity=["amplitude", "rate"],  # These must be > 0
        enforce_ranges={"temperature": (0, 1000)},  # Physical limits
    )

    bounds = inference.infer(
        model, x, y, p0, param_names=["amplitude", "rate", "temperature"]
    )

Data-Driven Estimation
~~~~~~~~~~~~~~~~~~~~~~

Use data statistics for bounds estimation:

.. code-block:: python

    from nlsq.bound_inference import infer_bounds

    # Uses data statistics (min, max, mean, std)
    bounds = infer_bounds(
        model,
        x,
        y,
        p0,
        use_data_range=True,  # Use min/max of data
        margin_factor=1.5,  # Add 50% margin
    )

    print(f"Data-driven bounds: {bounds}")

Bounds Validation
~~~~~~~~~~~~~~~~~

Validate inferred or user-provided bounds:

.. code-block:: python

    from nlsq.bound_inference import validate_bounds

    # Check if bounds are reasonable
    validation_result = validate_bounds(
        bounds,
        p0,
        check_coverage=True,  # Ensure p0 is within bounds
        check_feasibility=True,  # Ensure bounds make sense
    )

    if not validation_result.valid:
        print(f"Warning: {validation_result.warnings}")

Inference Strategies
--------------------

The module uses multiple strategies for bounds inference:

Data Range Strategy
~~~~~~~~~~~~~~~~~~~

Estimates bounds from data characteristics:

- **Lower**: `min(y) - margin * range(y)`
- **Upper**: `max(y) + margin * range(y)`
- **Margin**: Typically 10-20% of data range

Model Type Strategy
~~~~~~~~~~~~~~~~~~~

Uses domain knowledge for common models:

- **Gaussian**: amplitude ~ max(y), mean ~ median(x), std ~ range(x)/6
- **Exponential**: amplitude ~ max(y), rate > 0, offset ~ min(y)
- **Sigmoid**: L ~ max(y), x0 ~ median(x), k > 0

Parameter Name Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Infers from parameter names:

- **"amplitude", "height"**: 0 to 2*max(y)
- **"rate", "decay"**: Positive only
- **"temperature"**: Physical limits (0 to reasonable max)
- **"concentration"**: Non-negative

Initialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Based on initial parameter guess:

- **Lower**: `p0 / factor` (default factor = 10)
- **Upper**: `p0 * factor`
- **Sign preservation**: If p0 > 0, lower = 0

Configuration
-------------

Configure bounds inference behavior:

.. code-block:: python

    from nlsq.bound_inference import BoundsInference

    inference = BoundsInference(
        default_margin=0.2,  # 20% margin around data range
        enforce_positivity=[],  # Parameter indices for positivity
        use_model_heuristics=True,  # Use model-specific knowledge
        confidence_threshold=0.8,  # Minimum confidence for inference
        fallback_to_unbounded=True,  # Use inf bounds if uncertain
    )

See Also
--------

- :doc:`../reference/configuration` : Configuration reference
- :doc:`nlsq.minpack` : Main curve fitting API
- :doc:`nlsq.validators` : Input validation
