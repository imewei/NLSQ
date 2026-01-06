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
   * - ``random``
     - Uniform random sampling

Usage Example
-------------

.. code-block:: python

   from nlsq.global_optimization.sampling import create_sampler

   # Create a Latin Hypercube sampler
   sampler = create_sampler("lhs", n_dims=3, bounds=[(0, 10), (0, 5), (0, 1)])

   # Generate 20 starting points
   points = sampler.sample(n_points=20)
