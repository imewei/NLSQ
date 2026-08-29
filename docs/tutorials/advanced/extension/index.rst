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

NLSQ's optimizer classes are duck-typed rather than defined against a formal
protocol/ABC hierarchy: :class:`~nlsq.core.trf.TrustRegionReflective` is a
plain class exposing ``.trf()``, ``.calculate_cost()``, and
``.default_loss_func()``, and anything providing the same interface can be
used in its place.

.. code-block:: python

   class MyOptimizer:
       def optimize(self, fun, x0, **kwargs):
           # Your optimization logic
           pass


   # Use with NLSQ infrastructure
   optimizer = MyOptimizer()
