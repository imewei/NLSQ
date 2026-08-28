How to Use Streaming Checkpoints
=================================

For very long fits on large datasets, ``AdaptiveHybridStreamingOptimizer``
can periodically save its optimization state to disk. This lets you inspect
progress on a fit that is still running, or recover the last-known state if
the process is killed.

.. note::

   As of this writing, checkpoint **saving** is implemented, but automatic
   **resuming** from a saved checkpoint is not. Setting
   ``resume_from_checkpoint`` on the config raises ``NotImplementedError``
   when you call ``fit()``. See `Resuming from Checkpoint`_ below.

When to Use Checkpoints
------------------------

Consider enabling checkpoints when:

- Fit may take **> 1 hour**
- Running on **unreliable infrastructure** (cloud spot instances)
- Processing **very large datasets** (> 10 million points)
- You want visibility into optimizer state during a long run

Basic Checkpoint Usage
------------------------

Enable checkpointing through ``HybridStreamingConfig``:

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   config = HybridStreamingConfig(
       checkpoint_dir="./checkpoints",  # Where to save (None disables saving)
       checkpoint_frequency=100,  # Save every 100 iterations (default)
   )

   optimizer = AdaptiveHybridStreamingOptimizer(config)

   # data_source is currently a (x_data, y_data) tuple
   result = optimizer.fit((x_data, y_data), model, p0=[2.0, 0.5, 0.3])

``checkpoint_frequency`` counts optimizer **iterations**, not seconds.
Checkpoints are only written if ``checkpoint_dir`` is set and
``enable_checkpoints`` is ``True`` (the default).

Checkpoint File Structure
---------------------------

Checkpoints are saved as HDF5 files, one per checkpoint, named by phase and
iteration number:

.. code-block:: text

   ./checkpoints/
   ├── checkpoint_phase1_iter100.h5
   ├── checkpoint_phase1_iter200.h5
   ├── checkpoint_phase2_iter50.h5
   └── checkpoint_phase2_iter100.h5

Each checkpoint stores (format version ``3.0``):

- Current phase and normalized parameters
- Phase 1 L-BFGS optimizer state (if applicable)
- Phase 2 accumulated J^T J / J^T r matrices (if applicable)
- Best parameters and cost found so far
- Phase history and, if multi-start is enabled, tournament state

Checkpoints are not pruned automatically; old files accumulate in
``checkpoint_dir`` until you remove them yourself.

Resuming from Checkpoint
--------------------------

``HybridStreamingConfig`` has a ``resume_from_checkpoint`` field, but
``fit()`` does not currently act on it — passing it raises
``NotImplementedError`` with guidance to either unset it or load state
manually. There is no supported ``resume=True`` argument to ``fit()``.

If you need the saved state for inspection or a custom recovery path, load
it directly with ``CheckpointManager``:

.. code-block:: python

   from nlsq.streaming.phases import CheckpointManager

   manager = CheckpointManager(config)
   state = manager.load("./checkpoints/checkpoint_phase2_iter200.h5")

   print(state.current_phase, state.best_cost_global)
   print(state.best_params_global)

Wiring a loaded ``CheckpointState`` back into a fresh optimizer run is a
manual, low-level operation — there is no public API that restarts ``fit()``
from it today.

Best Practices
----------------

1. **Use Absolute Paths**

   .. code-block:: python

      import os

      checkpoint_dir = os.path.abspath("./checkpoints")

2. **Clean Up Checkpoints After a Successful Run**

   Since checkpoints are never pruned automatically, remove them once a fit
   completes if you don't need the intermediate history:

   .. code-block:: python

      import shutil

      shutil.rmtree(checkpoint_dir, ignore_errors=True)

3. **Log Checkpoint Events**

   .. code-block:: python

      import logging

      logging.basicConfig(level=logging.INFO)

Complete Example
------------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig
   import os


   def model(x, a, b, c, d):
       return a * jnp.exp(-b * x) * jnp.sin(c * x) + d


   # Generate large dataset
   np.random.seed(42)
   n = 5_000_000
   x = np.linspace(0, 100, n)
   y = 2.0 * np.exp(-0.02 * x) * np.sin(0.5 * x) + 1.0
   y += 0.1 * np.random.randn(n)

   checkpoint_dir = os.path.abspath("./my_fit_checkpoints")

   config = HybridStreamingConfig(
       checkpoint_dir=checkpoint_dir,
       checkpoint_frequency=100,
   )
   optimizer = AdaptiveHybridStreamingOptimizer(config)

   result = optimizer.fit((x, y), model, p0=[2.0, 0.02, 0.5, 1.0], verbose=1)

   print(f"Parameters: {result['x']}")

   # Cleanup checkpoints after successful completion
   import shutil

   shutil.rmtree(checkpoint_dir, ignore_errors=True)

See Also
----------

- :doc:`handle_large_data` - Large dataset handling
- :doc:`/tutorials/routine/data_handling/large_datasets` - Large dataset tutorial
- :doc:`/explanation/streaming` - How streaming works
