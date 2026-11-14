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

Complete callback system with progress bars and early stopping.

Key Features
------------

- **Real-time progress monitoring** with tqdm progress bars
- **Early stopping** when the cost stops improving
- **Iteration logging** to stdout or a file
- **Callback chaining** with ``CallbackChain``
- **Custom callbacks** by subclassing ``CallbackBase``

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   CallbackBase
   CallbackChain
   StopOptimization
   ProgressBar
   EarlyStopping
   IterationLogger

Usage Examples
--------------

Progress Bar
~~~~~~~~~~~~

Display a real-time progress bar during optimization (requires ``tqdm``):

.. code-block:: python

    import jax.numpy as jnp
    import numpy as np

    from nlsq import curve_fit
    from nlsq.callbacks import ProgressBar


    def model(x, a, b):
        return a * jnp.exp(-b * x)


    x = np.linspace(0, 10, 200)
    y = 2.5 * np.exp(-0.5 * x)

    popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5], callback=ProgressBar())

Early Stopping
~~~~~~~~~~~~~~

Stop optimization when the cost has not improved by at least ``min_delta`` for
``patience`` consecutive iterations:

.. code-block:: python

    from nlsq.callbacks import EarlyStopping

    callback = EarlyStopping(patience=10, min_delta=1e-6, verbose=True)

    popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5], callback=callback)

    print(f"Best cost: {callback.best_cost}")

Iteration Logging
~~~~~~~~~~~~~~~~~

Log cost, gradient norm, function evaluations and elapsed time at each iteration,
to stdout or to a file:

.. code-block:: python

    from nlsq.callbacks import IterationLogger

    logger = IterationLogger(filename="fit.log", log_params=True)

    popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5], callback=logger)
    logger.close()  # flush and close the log file

Combining Callbacks
~~~~~~~~~~~~~~~~~~~

Use multiple callbacks simultaneously with ``CallbackChain``:

.. code-block:: python

    from nlsq.callbacks import CallbackChain, EarlyStopping, IterationLogger

    callbacks = CallbackChain(
        EarlyStopping(patience=10),
        IterationLogger(filename="fit.log"),
    )

    popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5], callback=callbacks)
    callbacks.close()

Custom Callbacks
~~~~~~~~~~~~~~~~

Create your own callback by subclassing ``CallbackBase`` and raising
``StopOptimization`` to request early termination:

.. code-block:: python

    from nlsq.callbacks import CallbackBase, StopOptimization


    class CostThreshold(CallbackBase):
        def __call__(self, iteration, cost, params, info):
            if cost < 1e-10:
                print("Excellent fit achieved!")
                raise StopOptimization("cost below threshold")


    popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5], callback=CostThreshold())

Callback Interface
------------------

A callback is any callable (``CallbackBase`` subclasses are the convention) with
the signature ``callback(iteration, cost, params, info)``:

- ``iteration``: 0-indexed iteration number
- ``cost``: current cost value
- ``params``: current parameter values (``numpy.ndarray``)
- ``info``: dict with additional information such as ``gradient_norm`` and ``nfev``

``CallbackBase.close()`` can be overridden to release resources (files, network
connections); ``IterationLogger`` uses it to close its log file.

Interactive Notebooks
---------------------

- `Callbacks Demo <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/05_feature_demos/callbacks_demo.ipynb>`_ (15 min) - Progress monitoring, early stopping, and custom callbacks

See Also
--------

- :doc:`../reference/configuration` : Configuration reference
- :doc:`nlsq.minpack` : Main curve fitting API
- :doc:`nlsq.diagnostics` : Diagnostics and monitoring
