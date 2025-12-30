nlsq.parameter\_normalizer module
==================================

.. currentmodule:: nlsq.precision.parameter_normalizer

.. automodule:: nlsq.precision.parameter_normalizer
   :noindex:
   :exclude-members: normalization_jacobian

Overview
--------

The ``nlsq.parameter_normalizer`` module provides automatic parameter scaling to
address gradient signal weakness caused by parameter scale imbalance. Parameters
spanning many orders of magnitude can cause slow convergence and numerical
instability in optimization.

**New in version 0.3.0**: Parameter normalization for adaptive hybrid streaming.

Key Features
------------

- **Bounds-based normalization**: Normalize to [0, 1] using parameter bounds
- **p0-based normalization**: Scale by initial parameter magnitudes
- **Identity transform**: No normalization option
- **JAX JIT compatibility**: All operations are JIT-compilable
- **Automatic Jacobian computation**: Analytical denormalization Jacobian for covariance transform
- **Transparent model wrapping**: User model operates in original parameter space

Classes
-------

.. autoclass:: ParameterNormalizer
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: normalization_jacobian

.. autoclass:: NormalizedModelWrapper
   :members:
   :undoc-members:
   :show-inheritance:

Normalization Strategies
------------------------

Bounds-Based Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Normalizes parameters to [0, 1] using provided bounds. Best when you have
meaningful parameter bounds:

.. code-block:: python

    import jax.numpy as jnp
    from nlsq.parameter_normalizer import ParameterNormalizer

    # Parameters: amplitude in [10, 100], decay in [0, 1]
    p0 = jnp.array([50.0, 0.5])
    bounds = (jnp.array([10.0, 0.0]), jnp.array([100.0, 1.0]))

    normalizer = ParameterNormalizer(p0, bounds, strategy="bounds")

    # Normalize: (50-10)/(100-10) = 0.444, (0.5-0)/(1-0) = 0.5
    normalized = normalizer.normalize(p0)
    print(normalized)  # [0.444... 0.5]

    # Denormalize back to original
    denormalized = normalizer.denormalize(normalized)
    print(jnp.allclose(denormalized, p0))  # True

p0-Based Normalization
~~~~~~~~~~~~~~~~~~~~~~

Scales parameters by their initial magnitudes. Best when parameters have
vastly different scales but no clear bounds:

.. code-block:: python

    # Parameters: large, medium, small
    p0 = jnp.array([1000.0, 1.0, 0.001])

    normalizer = ParameterNormalizer(p0, bounds=None, strategy="p0")

    # All parameters normalized to ~1.0
    normalized = normalizer.normalize(p0)
    print(normalized)  # [1. 1. 1.]

    # Works with different values
    params = jnp.array([500.0, 2.0, 0.002])
    normalized = normalizer.normalize(params)
    print(normalized)  # [0.5 2. 2.]

No Normalization
~~~~~~~~~~~~~~~~

Identity transform when normalization is not needed:

.. code-block:: python

    p0 = jnp.array([5.0, 15.0])
    normalizer = ParameterNormalizer(p0, bounds=None, strategy="none")

    normalized = normalizer.normalize(p0)
    print(jnp.allclose(normalized, p0))  # True

Auto Strategy
~~~~~~~~~~~~~

Automatically selects the best strategy:

.. code-block:: python

    # With bounds: uses bounds-based
    normalizer = ParameterNormalizer(p0, bounds=bounds, strategy="auto")
    print(normalizer.strategy)  # 'bounds'

    # Without bounds: uses p0-based
    normalizer = ParameterNormalizer(p0, bounds=None, strategy="auto")
    print(normalizer.strategy)  # 'p0'

Usage Examples
--------------

Model Wrapping for Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``NormalizedModelWrapper`` to transparently work in normalized space:

.. code-block:: python

    import jax.numpy as jnp
    from nlsq.parameter_normalizer import ParameterNormalizer, NormalizedModelWrapper


    # Define model in original parameter space
    def model(x, amplitude, decay):
        return amplitude * jnp.exp(-decay * x)


    # Setup normalization
    p0 = jnp.array([100.0, 0.1])  # Very different scales
    bounds = (jnp.array([10.0, 0.01]), jnp.array([200.0, 1.0]))
    normalizer = ParameterNormalizer(p0, bounds, strategy="bounds")

    # Wrap model
    wrapped_model = NormalizedModelWrapper(model, normalizer)

    # Use wrapped model with normalized parameters
    x = jnp.linspace(0, 10, 100)
    normalized_p0 = normalizer.normalize(p0)

    # Wrapped model internally denormalizes before calling original model
    predictions = wrapped_model(x, *normalized_p0)
    print(predictions.shape)  # (100,)

JIT Compilation
~~~~~~~~~~~~~~~

All operations are JAX JIT-compatible:

.. code-block:: python

    import jax
    from nlsq.parameter_normalizer import ParameterNormalizer, NormalizedModelWrapper


    def model(x, a, b):
        return a * x + b


    p0 = jnp.array([5.0, 10.0])
    normalizer = ParameterNormalizer(p0, bounds=None, strategy="p0")
    wrapped = NormalizedModelWrapper(model, normalizer)


    @jax.jit
    def compute_predictions(x, a_norm, b_norm):
        return wrapped(x, a_norm, b_norm)


    x = jnp.array([1.0, 2.0, 3.0])
    normalized = normalizer.normalize(p0)
    result = compute_predictions(x, *normalized)

Bounds Transformation
~~~~~~~~~~~~~~~~~~~~~

Transform bounds to normalized space:

.. code-block:: python

    p0 = jnp.array([50.0])
    bounds = (jnp.array([10.0]), jnp.array([100.0]))
    normalizer = ParameterNormalizer(p0, bounds, strategy="bounds")

    lb_norm, ub_norm = normalizer.transform_bounds()
    print(lb_norm, ub_norm)  # [0.] [1.]

Covariance Transform
~~~~~~~~~~~~~~~~~~~~

Transform covariance from normalized to original space using the Jacobian:

.. code-block:: python

    # Get denormalization Jacobian
    J = normalizer.normalization_jacobian
    print(J.shape)  # (n_params, n_params)

    # Transform covariance: Cov_orig = J @ Cov_norm @ J.T
    cov_normalized = jnp.eye(2) * 0.01  # Example normalized covariance
    cov_original = J @ cov_normalized @ J.T

Mathematical Details
--------------------

Normalization Transform
~~~~~~~~~~~~~~~~~~~~~~~

For bounds-based normalization with bounds :math:`[l_i, u_i]`:

.. math::

    \\theta_{\\text{norm},i} = \\frac{\\theta_i - l_i}{u_i - l_i}

For p0-based normalization with initial values :math:`\\theta_0`:

.. math::

    \\theta_{\\text{norm},i} = \\frac{\\theta_i}{|\\theta_{0,i}|}

Denormalization Jacobian
~~~~~~~~~~~~~~~~~~~~~~~~

The denormalization Jacobian is diagonal:

.. math::

    J_{ii} = \\text{scale}_i

For bounds-based: :math:`J_{ii} = u_i - l_i`

For p0-based: :math:`J_{ii} = |\\theta_{0,i}|`

Covariance Transform
~~~~~~~~~~~~~~~~~~~~

To transform covariance from normalized to original space:

.. math::

    \\Sigma_{\\text{orig}} = J \\, \\Sigma_{\\text{norm}} \\, J^T

Since :math:`J` is diagonal, this simplifies to:

.. math::

    \\Sigma_{\\text{orig},ij} = J_{ii} \\, \\Sigma_{\\text{norm},ij} \\, J_{jj}

See Also
--------

- :doc:`nlsq.adaptive_hybrid_streaming` : Uses this for Phase 0 normalization
- :doc:`nlsq.hybrid_streaming_config` : Configuration with normalization_strategy
- :doc:`../howto/advanced_api` : Advanced optimization features
