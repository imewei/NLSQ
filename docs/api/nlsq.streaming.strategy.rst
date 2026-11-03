nlsq.streaming.strategy module
==============================

Internal builder for the two large-dataset engines: ``build_large_strategy(kind,
config, **common)`` constructs ``AdaptiveHybridStreamingOptimizer`` (``kind="hybrid"``)
or ``LargeDatasetFitter`` (``kind="chunked"``) for a path the caller has already
chosen; it does not route. ``Strategy.fit`` owns the input kwargs translation (the
hybrid engine has no ``gtol``/``ftol``/``xtol``; ``gtol`` maps onto
``gauss_newton_tol`` at build time) and returns the engine's native result
unchanged.

.. automodule:: nlsq.streaming.strategy
   :members:
   :undoc-members:
   :show-inheritance:
