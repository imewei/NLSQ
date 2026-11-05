nlsq.global_optimization.sampling
==================================

Sampling strategies for multi-start global optimization.

This module provides various sampling methods for generating initial parameter
guesses in multi-start optimization scenarios.

.. automodule:: nlsq.global_optimization.sampling
   :members:
   :undoc-members:
   :show-inheritance:

Sampling Methods
----------------

The module supports the following sampling strategies:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Description
   * - ``lhs``
     - Latin Hypercube Sampling - space-filling with good coverage
   * - ``sobol``
     - Sobol quasi-random sequences - low discrepancy sampling
   * - ``halton``
     - Halton sequences - deterministic quasi-random sampling

Usage Example
-------------

All samplers generate points in the unit hypercube ``[0, 1]^n_dims``; use
``scale_samples_to_bounds`` to map them to parameter bounds.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from nlsq.global_optimization.sampling import get_sampler, scale_samples_to_bounds

   # Get a Latin Hypercube sampler by name ('lhs', 'sobol' or 'halton')
   sampler = get_sampler("lhs")

   # Generate 20 points in the unit cube (reproducible via rng_key)
   unit_points = sampler(20, 3, rng_key=jax.random.key(0))

   # Scale to the parameter bounds
   lb = jnp.array([0.0, 0.0, 0.0])
   ub = jnp.array([10.0, 5.0, 1.0])
   points = scale_samples_to_bounds(unit_points, lb, ub)
   print(points.shape)  # (20, 3)
