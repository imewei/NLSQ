Multi-GPU
=========

With several GPUs visible to JAX, NLSQ's multi-start global optimization
evaluates starting points in parallel worker threads, one worker per visible
device. NLSQ does not pin a start to a particular GPU and does not shard a
single fit's data across GPUs: where each start's computation runs is left to
JAX's default device placement.

Detecting Multiple GPUs
-----------------------

.. code-block:: python

   import jax

   devices = jax.devices()
   print(f"Available devices: {len(devices)}")
   for i, device in enumerate(devices):
       print(f"  {i}: {device}")

Example output:

.. code-block:: text

   Available devices: 4
     0: CudaDevice(id=0)
     1: CudaDevice(id=1)
     2: CudaDevice(id=2)
     3: CudaDevice(id=3)

How Multi-Start Uses Several GPUs
---------------------------------

With ``workflow='auto_global'`` or ``workflow='hpc'`` on the multi-start route
(standard memory strategy), starting points are evaluated in a thread pool whose
size is ``min(n_starts, number of visible JAX devices)`` when more than one
device is visible (``_select_worker_count`` in
``nlsq/global_optimization/multi_start.py``). With one device the pool is capped
at 4 workers, which also covers a CPU-only machine (JAX then reports one CPU
device). The CPU-count cap (at most 16) applies only if JAX reports no devices.

The thread count follows the device count, but the code does not assign each
start to its own GPU. Watch ``nvidia-smi`` to see how the work is actually
placed on your machine. The CMA-ES route has no per-GPU worker logic.

.. code-block:: python

   from nlsq import fit

   # With 4 visible GPUs, up to 4 starts run concurrently
   popt, pcov = fit(
       model, x, y, p0=[...], workflow="auto_global", bounds=bounds, n_starts=100
   )

Controlling GPU Selection
-------------------------

Select specific GPUs:

.. code-block:: bash

   # Use only GPUs 0 and 1
   export CUDA_VISIBLE_DEVICES=0,1

   python my_fit.py

Or in Python:

.. code-block:: python

   import os

   os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

   # Must set before importing JAX
   import jax
   from nlsq import fit

Single GPU Selection
--------------------

Force use of a specific GPU:

.. code-block:: python

   import os

   os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use only GPU 2

   from nlsq import fit

HPC Cluster Configuration
-------------------------

On HPC clusters with job schedulers:

**PBS:**

.. code-block:: bash

   #PBS -l nodes=1:ppn=8:gpus=4

   cd $PBS_O_WORKDIR
   python my_fit.py

**SLURM:**

.. code-block:: bash

   #SBATCH --gres=gpu:4
   #SBATCH --ntasks-per-node=1

   python my_fit.py

NLSQ does not read the scheduler's GPU allocation to configure the fit. The
GPUs it counts are the ones JAX can see inside the job (see "Detecting Multiple
GPUs" above). ``workflow="hpc"`` additionally logs a PBS or local-GPU summary,
and recognises no SLURM variables (see :doc:`../three_workflows/hpc_workflow`).

Memory Considerations
---------------------

With multiple GPUs, each GPU uses memory:

.. code-block:: python

   import os

   # Limit memory per GPU
   os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

   # Ensure each GPU doesn't preallocate all memory
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

Performance Tips
----------------

**1. Match n_starts to GPU count:**

.. code-block:: python

   import jax

   n_gpus = len(jax.devices())
   n_starts = n_gpus * 10  # a multiple of the worker count keeps every worker busy

   popt, pcov = fit(
       model, x, y, p0=[...], workflow="auto_global", bounds=bounds, n_starts=n_starts
   )

**2. Large data benefits most:**

Multi-GPU is most beneficial when:

- Dataset is large (100K+ points)
- Many starting points needed (global optimization)
- Model is computationally intensive

**3. Watch for memory:**

Large datasets + many GPUs can exceed memory. Monitor with:

.. code-block:: bash

   watch nvidia-smi

Complete Example
----------------

.. code-block:: python

   import os
   import numpy as np
   import jax
   import jax.numpy as jnp
   from nlsq import fit

   # Check GPU configuration
   print(f"Available GPUs: {len(jax.devices())}")
   for dev in jax.devices():
       print(f"  {dev}")


   # Model with multiple local minima
   def complex_model(x, a, b, c, d, e):
       return a * jnp.exp(-b * x) * jnp.sin(c * x + d) + e


   # Generate data
   np.random.seed(42)
   n_points = 500_000
   x = np.linspace(0, 20, n_points)
   y_true = 2 * np.exp(-0.1 * x) * np.sin(2 * x + 0.5) + 0.3
   y = y_true + 0.2 * np.random.randn(n_points)

   # Bounds for global search
   bounds = ([0, 0, 0, -np.pi, -1], [10, 1, 10, np.pi, 1])

   # Global optimization; the multi-start worker pool is sized by visible GPUs
   n_gpus = len(jax.devices())
   n_starts = n_gpus * 20  # Scale starts with the worker count

   import time

   start = time.time()

   popt, pcov = fit(
       complex_model,
       x,
       y,
       p0=[2, 0.1, 2, 0.5, 0.3],
       workflow="auto_global",
       bounds=bounds,
       n_starts=n_starts,
   )

   print(f"\nFit time: {time.time() - start:.1f}s")
   print(f"Results: {popt}")

Troubleshooting
---------------

**Not all GPUs used:**

.. code-block:: bash

   # Check visible devices
   echo $CUDA_VISIBLE_DEVICES

   # Check JAX sees all GPUs
   python -c "import jax; print(jax.devices())"

**Memory errors with multiple GPUs:**

- Reduce ``memory_limit_gb``
- Use ``XLA_PYTHON_CLIENT_MEM_FRACTION``
- Reduce batch size

**Uneven GPU utilization:**

- Normal for small workloads
- Increase ``n_starts`` for better distribution

Next Steps
----------

- :doc:`../three_workflows/hpc_workflow` - HPC configuration
- :doc:`../troubleshooting/common_issues` - Troubleshooting guide
