How to Use Streaming Checkpoints
=================================

For very long fits on large datasets, checkpointing allows you to save
progress and resume if the process is interrupted.

When to Use Checkpoints
-----------------------

Consider checkpointing when:

- Fit may take **> 1 hour**
- Running on **unreliable infrastructure** (cloud spot instances)
- Processing **very large datasets** (> 10 million points)
- Doing **exploratory optimization** (may want to stop and restart)

Basic Checkpoint Usage
----------------------

Enable checkpointing with the adaptive hybrid streaming optimizer:

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Create optimizer with checkpointing
   optimizer = AdaptiveHybridStreamingOptimizer(
       model,
       n_params=3,
       checkpoint_dir="./checkpoints",  # Where to save
       checkpoint_interval=60,  # Save every 60 seconds
   )

   # Start fit
   result = optimizer.fit(x_data, y_data, p0=[2.0, 0.5, 0.3])

Resuming from Checkpoint
------------------------

If the fit is interrupted, resume from the last checkpoint:

.. code-block:: python

   # Resume from existing checkpoint
   optimizer = AdaptiveHybridStreamingOptimizer(
       model,
       n_params=3,
       checkpoint_dir="./checkpoints",
   )

   # This will automatically detect and resume from checkpoint
   result = optimizer.fit(x_data, y_data, p0=[2.0, 0.5, 0.3], resume=True)

Checkpoint File Structure
-------------------------

Checkpoints are saved as JSON files:

.. code-block:: text

   ./checkpoints/
   ├── checkpoint_20240101_120000.json
   ├── checkpoint_20240101_121000.json
   └── checkpoint_latest.json

Each checkpoint contains:

- Current parameter values
- Iteration count
- Accumulated gradients and state
- Data chunk progress
- Timing information

Managing Checkpoints
--------------------

Listing Checkpoints
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   import json

   checkpoint_dir = "./checkpoints"
   checkpoints = sorted(
       [
           f
           for f in os.listdir(checkpoint_dir)
           if f.startswith("checkpoint_") and f.endswith(".json")
       ]
   )

   for cp in checkpoints:
       with open(os.path.join(checkpoint_dir, cp)) as f:
           data = json.load(f)
       print(f"{cp}: iteration {data['iteration']}, params {data['params']}")

Cleaning Old Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Keep only the 5 most recent checkpoints
   max_checkpoints = 5
   checkpoints = sorted(
       [
           f
           for f in os.listdir(checkpoint_dir)
           if f.startswith("checkpoint_") and f != "checkpoint_latest.json"
       ]
   )

   for old_cp in checkpoints[:-max_checkpoints]:
       os.remove(os.path.join(checkpoint_dir, old_cp))
       print(f"Removed {old_cp}")

Configuration Options
---------------------

.. code-block:: python

   optimizer = AdaptiveHybridStreamingOptimizer(
       model,
       n_params=3,
       # Checkpoint settings
       checkpoint_dir="./checkpoints",  # Directory for checkpoints
       checkpoint_interval=120,  # Seconds between saves (default: 60)
       max_checkpoints=10,  # Max files to keep (default: 5)
       checkpoint_compression=True,  # Compress checkpoint files
   )

Best Practices
--------------

1. **Use Absolute Paths**

   .. code-block:: python

      import os

      checkpoint_dir = os.path.abspath("./checkpoints")

2. **Test Resume Before Long Runs**

   .. code-block:: python

      # Start a short fit
      result = optimizer.fit(x[:1000], y[:1000], p0=p0, max_iter=10)

      # Verify checkpoint exists
      assert os.path.exists("./checkpoints/checkpoint_latest.json")

      # Test resume
      result = optimizer.fit(x[:1000], y[:1000], p0=p0, resume=True)

3. **Log Checkpoint Events**

   .. code-block:: python

      import logging

      logging.basicConfig(level=logging.INFO)

      # Now you'll see checkpoint saves in logs

4. **Use Fast Storage**

   Checkpoint files are small (~1KB), but frequent writes benefit from
   fast storage (SSD preferred over network drives).

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import AdaptiveHybridStreamingOptimizer
   import os
   import time


   def model(x, a, b, c, d):
       return a * jnp.exp(-b * x) * jnp.sin(c * x) + d


   # Generate large dataset
   np.random.seed(42)
   n = 5_000_000
   x = np.linspace(0, 100, n)
   y = 2.0 * np.exp(-0.02 * x) * np.sin(0.5 * x) + 1.0
   y += 0.1 * np.random.randn(n)

   # Setup checkpointing
   checkpoint_dir = os.path.abspath("./my_fit_checkpoints")
   os.makedirs(checkpoint_dir, exist_ok=True)

   print(f"Fitting {n:,} points with checkpointing")
   print(f"Checkpoints will be saved to: {checkpoint_dir}")

   # Create optimizer
   optimizer = AdaptiveHybridStreamingOptimizer(
       model,
       n_params=4,
       checkpoint_dir=checkpoint_dir,
       checkpoint_interval=30,  # Save every 30 seconds
   )

   # Fit with optional resume
   resume = os.path.exists(os.path.join(checkpoint_dir, "checkpoint_latest.json"))
   if resume:
       print("Resuming from checkpoint...")

   start = time.time()
   result = optimizer.fit(
       x, y, p0=[2.0, 0.02, 0.5, 1.0], show_progress=True, resume=resume
   )
   elapsed = time.time() - start

   print(f"\nCompleted in {elapsed:.1f}s")
   print(f"Parameters: {result.popt}")

   # Cleanup checkpoints after successful completion
   for f in os.listdir(checkpoint_dir):
       os.remove(os.path.join(checkpoint_dir, f))
   os.rmdir(checkpoint_dir)
   print("Cleaned up checkpoints")

Troubleshooting
---------------

"Checkpoint not found" error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check if checkpoint exists
   import os

   cp_file = "./checkpoints/checkpoint_latest.json"
   if not os.path.exists(cp_file):
       print("No checkpoint found, starting fresh")
       resume = False
   else:
       resume = True

"Incompatible checkpoint" error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checkpoints are tied to:

- Model function (same signature)
- Number of parameters
- Optimizer settings

If these change, delete old checkpoints and start fresh:

.. code-block:: python

   import shutil

   shutil.rmtree("./checkpoints", ignore_errors=True)

See Also
--------

- :doc:`handle_large_data` - Large dataset handling
- :doc:`/tutorials/05_large_datasets` - Large dataset tutorial
- :doc:`/explanation/streaming` - How streaming works
