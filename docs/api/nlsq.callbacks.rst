nlsq.callbacks module
======================

.. currentmodule:: nlsq.callbacks

.. automodule:: nlsq.callbacks
   :noindex:

Overview
--------

The ``nlsq.callbacks`` module provides callback functions for monitoring and controlling
optimization progress. Callbacks can display progress bars, log iterations, implement early
stopping, and save intermediate results.

**New in version 0.1.1**: Complete callback system with progress bars and early stopping.

Key Features
------------

- **Real-time progress monitoring** with rich progress bars
- **Early stopping** based on convergence criteria
- **Iteration logging** for debugging and analysis
- **Result saving** for checkpointing
- **Custom callbacks** for specialized monitoring

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ProgressBar
   EarlyStopping
   IterationLogger

Usage Examples
--------------

Progress Bar
~~~~~~~~~~~~

Display a real-time progress bar during optimization:

.. code-block:: python

    from nlsq import curve_fit
    from nlsq.callbacks import ProgressBar
    import jax.numpy as jnp


    def model(x, a, b):
        return a * jnp.exp(-b * x)


    result = curve_fit(model, x, y, p0=[1.0, 0.5], callback=ProgressBar())

Early Stopping
~~~~~~~~~~~~~~

Stop optimization when parameters converge:

.. code-block:: python

    from nlsq.callbacks import EarlyStopping

    callback = EarlyStopping(patience=10, min_delta=1e-6, monitor="cost")

    result = curve_fit(model, x, y, p0=[1.0, 0.5], callback=callback)

    print(f"Stopped at iteration {callback.stopped_iteration}")

Iteration Logging
~~~~~~~~~~~~~~~~~

Log detailed information at each iteration:

.. code-block:: python

    from nlsq.callbacks import IterationLogger

    logger = IterationLogger(log_every=5, log_cost=True, log_gradient_norm=True)

    result = curve_fit(model, x, y, p0=[1.0, 0.5], callback=logger)

    # Access logged history
    print(logger.history)

Result Saving
~~~~~~~~~~~~~

Save intermediate results during optimization:

.. code-block:: python

    from nlsq.callbacks import ResultSaver

    saver = ResultSaver(save_every=10, save_path="checkpoints/")

    result = curve_fit(model, x, y, p0=[1.0, 0.5], callback=saver)

Combining Callbacks
~~~~~~~~~~~~~~~~~~~

Use multiple callbacks simultaneously:

.. code-block:: python

    from nlsq.callbacks import CallbackComposer, ProgressBar, EarlyStopping

    callbacks = CallbackComposer(
        [ProgressBar(), EarlyStopping(patience=10), IterationLogger(log_every=5)]
    )

    result = curve_fit(model, x, y, p0=[1.0, 0.5], callback=callbacks)

Custom Callbacks
~~~~~~~~~~~~~~~~

Create your own custom callback by subclassing ``OptimizationCallback``:

.. code-block:: python

    from nlsq.callbacks import OptimizationCallback


    class CustomCallback(OptimizationCallback):
        def on_iteration_end(self, iteration, params, cost, gradient_norm):
            if cost < 1e-10:
                print("Excellent fit achieved!")
                self.stop_training = True


    result = curve_fit(model, x, y, p0=[1.0, 0.5], callback=CustomCallback())

Callback Methods
----------------

All callbacks inherit from ``OptimizationCallback`` and can override these methods:

- ``on_optimization_begin()``: Called before optimization starts
- ``on_iteration_begin(iteration)``: Called before each iteration
- ``on_iteration_end(iteration, params, cost, gradient_norm)``: Called after each iteration
- ``on_optimization_end(result)``: Called after optimization completes

Interactive Notebooks
---------------------

- `Callbacks Demo <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/05_feature_demos/callbacks_demo.ipynb>`_ (15 min) - Progress monitoring, early stopping, and custom callbacks

See Also
--------

- :doc:`../reference/configuration` : Configuration reference
- :doc:`nlsq.minpack` : Main curve fitting API
- :doc:`nlsq.diagnostics` : Diagnostics and monitoring
