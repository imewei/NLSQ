Extending NLSQ
==============

This chapter covers extending NLSQ with custom protocols, plugins, and
testing strategies.

.. toctree::
   :maxdepth: 1

   custom_protocols
   plugin_development
   testing_strategies

Chapter Overview
----------------

**Custom Protocols** (10 min)
   Implementing new protocols for custom components.

**Plugin Development** (10 min)
   Creating NLSQ extensions and plugins.

**Testing Strategies** (10 min)
   Testing custom components and extensions.

Extension Points
----------------

NLSQ can be extended at multiple levels:

1. **Custom Optimizers**: Implement OptimizerProtocol
2. **Custom Preprocessors**: Implement DataPreprocessorProtocol
3. **Custom Covariance**: Implement CovarianceComputerProtocol
4. **Custom Caches**: Implement CacheProtocol

.. code-block:: python

   from nlsq.interfaces import OptimizerProtocol


   class MyOptimizer:
       def optimize(self, fun, x0, **kwargs):
           # Your optimization logic
           pass


   # Use with NLSQ infrastructure
   optimizer: OptimizerProtocol = MyOptimizer()
