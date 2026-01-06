nlsq.global_optimization.config
================================

Configuration classes for global optimization strategies.

.. automodule:: nlsq.global_optimization.config
   :members:
   :undoc-members:
   :show-inheritance:

GlobalOptimizationConfig
------------------------

.. autoclass:: nlsq.global_optimization.config.GlobalOptimizationConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Usage Example
-------------

.. code-block:: python

   from nlsq.global_optimization import GlobalOptimizationConfig

   # Create config with Latin Hypercube Sampling
   config = GlobalOptimizationConfig(
       n_starts=20,
       sampler="lhs",
       parallel=True,
   )

   # Or use a preset
   config = GlobalOptimizationConfig.from_preset("multimodal")
