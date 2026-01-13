workflow="hpc" - HPC Cluster Optimization
=========================================

The ``hpc`` workflow is designed for long-running optimization jobs on High
Performance Computing (HPC) clusters. It wraps ``auto_global`` with automatic
checkpointing for fault tolerance and crash recovery.

When to Use
-----------

Use ``hpc`` workflow when:

- Running on HPC clusters (PBS, SLURM, etc.)
- Jobs may take hours or days to complete
- You need crash recovery via checkpoints
- Running on shared/preemptible resources

.. important::

   ``hpc`` **requires bounds** (same as ``auto_global``).

Basic Usage
-----------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # HPC workflow with checkpointing
   popt, pcov = fit(
       model,
       xdata,
       ydata,
       p0=[1.0, 0.5, 0.0],
       workflow="hpc",
       bounds=([0, 0, -1], [10, 5, 1]),
       checkpoint_dir="/scratch/my_job/checkpoints",
   )

Checkpointing
-------------

Checkpoints are saved periodically during optimization:

.. code-block:: python

   popt, pcov = fit(
       model,
       x,
       y,
       p0=[...],
       workflow="hpc",
       bounds=bounds,
       checkpoint_dir="/scratch/checkpoints",
       checkpoint_interval=10,
   )  # Save every 10 iterations

**Checkpoint contents:**

- Current best parameters
- Optimization state
- Iteration number
- All explored starting points

**Automatic recovery:**

If a job crashes and restarts, NLSQ automatically detects existing checkpoints
and resumes from the last saved state.

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
       checkpoint_dir="/scratch/$SLURM_JOB_ID/checkpoints",
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
       checkpoint_dir="/scratch/ckpts",
   )

NLSQ automatically distributes starting points across available GPUs.

Best Practices for HPC
----------------------

**1. Use scratch storage for checkpoints:**

.. code-block:: python

   # Good: fast local storage
   checkpoint_dir = "/scratch/user/job_123/ckpts"

   # Bad: network filesystem
   checkpoint_dir = "/home/user/checkpoints"

**2. Request appropriate walltime:**

Estimate based on:
- Dataset size
- Number of starts
- Complexity of model

**3. Handle preemption:**

For preemptible queues, use frequent checkpoints:

.. code-block:: python

   popt, pcov = fit(
       model, x, y, p0=[...], workflow="hpc", bounds=bounds, checkpoint_interval=5
   )  # More frequent saves

**4. Clean up checkpoints:**

After successful completion:

.. code-block:: python

   import shutil

   if fit_succeeded:
       shutil.rmtree(checkpoint_dir)

Complete HPC Example
--------------------

.. code-block:: python

   #!/usr/bin/env python
   """HPC curve fitting job with checkpointing."""

   import os
   import numpy as np
   import jax.numpy as jnp
   from nlsq import fit


   # Model definition
   def complex_model(x, a, b, c, d, e):
       return a * jnp.exp(-b * x) * jnp.cos(c * x + d) + e


   def main():
       # Setup paths
       job_id = os.environ.get("SLURM_JOB_ID", os.environ.get("PBS_JOBID", "local"))
       checkpoint_dir = f"/scratch/{job_id}/checkpoints"
       os.makedirs(checkpoint_dir, exist_ok=True)

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
           checkpoint_dir=checkpoint_dir,
           checkpoint_interval=10,
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
     - Yes
   * - Crash recovery
     - No
     - Yes
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
- Enable checkpointing for resume

**Checkpoint corruption:**

- Use atomic writes (NLSQ does this automatically)
- Check disk space on scratch

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
