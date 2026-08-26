workflow="hpc" - HPC Cluster Optimization
=========================================

The ``hpc`` workflow is designed for long-running optimization jobs on High
Performance Computing (HPC) clusters. It wraps ``auto_global`` with automatic
cluster detection (PBS/SLURM).

.. warning::

   **Checkpointing is not yet implemented.** ``hpc`` currently accepts
   ``checkpoint_dir``/``checkpoint_interval`` for forward API compatibility
   and emits a ``UserWarning`` that they are ignored -- no checkpoint file is
   ever written, and there is no crash recovery. Everything in the
   "Checkpointing" section below describes planned, not current, behavior.
   Today, ``hpc`` behaves identically to ``auto_global`` plus the cluster
   detection described further down this page.

When to Use
-----------

Use ``hpc`` workflow when:

- Running on HPC clusters (PBS, SLURM, etc.) and want automatic cluster
  detection logged for diagnostics
- Running memory-aware global optimization (same requirements as
  ``auto_global``)

.. important::

   ``hpc`` **requires bounds** (same as ``auto_global``).

Basic Usage
-----------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # HPC workflow (bounds required, same as auto_global)
   popt, pcov = fit(
       model,
       xdata,
       ydata,
       p0=[1.0, 0.5, 0.0],
       workflow="hpc",
       bounds=([0, 0, -1], [10, 5, 1]),
   )

Checkpointing (planned, not yet implemented)
---------------------------------------------

.. warning::

   Nothing in this section works today -- see the warning at the top of this
   page. ``checkpoint_dir``/``checkpoint_interval`` are accepted but ignored.

The eventual design is for checkpoints to be saved periodically during
optimization:

.. code-block:: python

   popt, pcov = fit(
       model,
       x,
       y,
       p0=[...],
       workflow="hpc",
       bounds=bounds,
       checkpoint_dir="/scratch/checkpoints",  # currently ignored
       checkpoint_interval=10,  # currently ignored
   )

**Planned checkpoint contents:**

- Current best parameters
- Optimization state
- Iteration number
- All explored starting points

**Planned automatic recovery:**

If a job crashes and restarts, NLSQ would automatically detect existing
checkpoints and resume from the last saved state. This does not happen yet.

Cluster Detection
-----------------

NLSQ automatically detects HPC environments:

**PBS/Torque:**

.. code-block:: bash

   # Detected via $PBS_NODEFILE
   export PBS_NODEFILE=/var/spool/pbs/aux/12345.node1

**SLURM:**

.. code-block:: bash

   # Detected via SLURM environment variables
   export SLURM_JOB_ID=12345
   export SLURM_NNODES=4

**Multi-GPU:**

.. code-block:: bash

   # Detected via JAX device count
   python -c "import jax; print(jax.device_count())"

HPC Job Script Example
----------------------

**PBS script:**

.. code-block:: bash

   #!/bin/bash
   #PBS -N nlsq_fit
   #PBS -l nodes=1:ppn=8:gpus=2
   #PBS -l walltime=24:00:00
   #PBS -q gpu

   cd $PBS_O_WORKDIR
   source activate nlsq_env

   python fit_job.py

**SLURM script:**

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=nlsq_fit
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --gres=gpu:2
   #SBATCH --time=24:00:00

   module load cuda
   source activate nlsq_env

   python fit_job.py

**fit_job.py:**

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp
   import numpy as np


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Load your data
   data = np.load("/data/experiment.npz")
   x, y = data["x"], data["y"]

   # Run HPC optimization
   popt, pcov = fit(
       model,
       x,
       y,
       p0=[1, 0.5, 0],
       workflow="hpc",
       bounds=([0, 0, -1], [10, 5, 1]),
       n_starts=50,
   )

   # Save results
   np.savez("/results/fit_result.npz", popt=popt, pcov=pcov)
   print(f"Fitted: {popt}")

Multi-GPU Configuration
-----------------------

For jobs with multiple GPUs:

.. code-block:: python

   popt, pcov = fit(
       model,
       x,
       y,
       p0=[...],
       workflow="hpc",
       bounds=bounds,
       n_starts=100,  # More starts for multi-GPU
   )

NLSQ automatically distributes starting points across available GPUs.

Best Practices for HPC
----------------------

**1. Request appropriate walltime:**

Estimate based on:
- Dataset size
- Number of starts
- Complexity of model

**2. Handle preemption:**

There is currently no checkpoint/resume support (see the warning at the top
of this page) -- on preemptible queues, a preempted job restarts the fit from
scratch. Prefer non-preemptible queues for long ``hpc`` jobs until
checkpointing lands, or reduce ``n_starts``/walltime to fit inside a single
preemption window.

Complete HPC Example
--------------------

.. code-block:: python

   #!/usr/bin/env python
   """HPC curve fitting job."""

   import os
   import numpy as np
   import jax.numpy as jnp
   from nlsq import fit


   # Model definition
   def complex_model(x, a, b, c, d, e):
       return a * jnp.exp(-b * x) * jnp.cos(c * x + d) + e


   def main():
       job_id = os.environ.get("SLURM_JOB_ID", os.environ.get("PBS_JOBID", "local"))

       # Load data
       data = np.load("experiment_data.npz")
       x, y, sigma = data["x"], data["y"], data["sigma"]

       # Define bounds
       bounds = (
           [0, 0, 0, -np.pi, -10],  # Lower bounds
           [100, 10, 20, np.pi, 10],  # Upper bounds
       )

       # Run HPC fit
       print(f"Starting HPC fit with job ID: {job_id}")
       popt, pcov = fit(
           complex_model,
           x,
           y,
           p0=[10, 1, 5, 0, 0],
           sigma=sigma,
           workflow="hpc",
           bounds=bounds,
           n_starts=100,
       )

       # Save results
       perr = np.sqrt(np.diag(pcov))
       np.savez("fit_results.npz", popt=popt, pcov=pcov, perr=perr)

       # Print summary
       names = ["a", "b", "c", "d", "e"]
       print("\nFit Results:")
       for name, val, err in zip(names, popt, perr):
           print(f"  {name} = {val:.4f} +/- {err:.4f}")


   if __name__ == "__main__":
       main()

Comparison: auto_global vs hpc
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - ``auto_global``
     - ``hpc``
   * - Checkpointing
     - No
     - No (planned, not yet implemented)
   * - Crash recovery
     - No
     - No (planned, not yet implemented)
   * - Cluster detection
     - No
     - Yes
   * - Overhead
     - Lower
     - Slightly higher
   * - Best for
     - Interactive use
     - Batch jobs

Troubleshooting HPC
-------------------

**Job times out before completion:**

- Increase walltime
- Reduce ``n_starts``
- (Resume-from-checkpoint is not available yet -- see the warning at the top
  of this page)

**Multi-GPU not detected:**

.. code-block:: python

   import jax

   print(f"Devices: {jax.devices()}")
   print(f"Device count: {jax.device_count()}")

**Memory errors on GPU:**

- Reduce batch size via ``memory_limit_gb``
- Use streaming for very large datasets

Next Steps
----------

- :doc:`../gpu_acceleration/multi_gpu` - Multi-GPU configuration
- :doc:`../troubleshooting/common_issues` - General troubleshooting
- :doc:`/reference/configuration` - Configuration reference
