workflow="hpc" - HPC Cluster Optimization
=========================================

The ``hpc`` workflow is designed for long-running optimization jobs on High
Performance Computing (HPC) clusters. It wraps ``auto_global`` with automatic
cluster detection (PBS/SLURM).

.. note::

   **Checkpoint/resume is implemented for the CMA-ES route only**
   (``method="cmaes"`` with ``CMAESConfig(restart_strategy="none")``). BIPOP
   restarts (``restart_strategy="bipop"``, the default) and the
   multistart/chunked/streaming ``workflow='hpc'`` routes do not support
   checkpointing yet -- passing ``checkpoint_dir`` on those routes emits a
   ``UserWarning`` and is silently ignored. See the "Checkpointing" section
   below for the working CMA-ES example.

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

Checkpointing (CMA-ES route only)
----------------------------------

Checkpoint/resume works for the CMA-ES route with
``CMAESConfig(restart_strategy="none")``. To use it, request the CMA-ES
method explicitly, pass a fixed ``seed``, and provide ``checkpoint_dir``
along with two stable identifiers, ``run_id`` and ``model_id``:

.. code-block:: python

   from nlsq import fit
   from nlsq.global_optimization import CMAESConfig

   popt, pcov = fit(
       model,
       x,
       y,
       p0=[...],
       workflow="hpc",
       bounds=bounds,
       method="cmaes",
       cmaes_config=CMAESConfig(restart_strategy="none"),
       checkpoint_dir="/scratch/checkpoints",
       checkpoint_interval=10,  # generations between checkpoint saves
       run_id="experiment-42",  # stable string identifying this run
       model_id="exp_decay_v1",  # stable string identifying the model function
       seed=42,  # required: checkpoint/resume needs a fixed seed
   )

``checkpoint_dir``, ``run_id``, ``model_id``, and ``seed`` are all required
together -- omitting any one of them raises a ``ValueError`` when
``checkpoint_dir`` is set. Combining ``checkpoint_dir`` with the default
``restart_strategy="bipop"`` raises ``NotImplementedError``; use
``restart_strategy="none"`` as shown above.

**Checkpoint contents:**

- Current best parameters and their fitness
- CMA-ES optimizer state (mean, covariance, step size, evolution paths)
- Generation number
- A fingerprint of the run (data shape, bounds, seed, ``model_id``,
  ``run_id``) used to validate a resume matches the original run

**Automatic recovery:**

To resume, call ``fit`` again with the *same* ``checkpoint_dir``, ``run_id``,
``seed``, ``model_id``, data, and bounds as the original run. NLSQ
auto-detects the existing checkpoint file, verifies the fingerprint matches,
and continues from the last saved generation. If the fingerprint doesn't
match (e.g. different data or bounds reused the same ``run_id``), ``fit``
raises a ``ValueError`` rather than silently starting over or resuming with
mismatched state.

**Handling preemption (SIGTERM/SIGUSR1):**

On a preemptible HPC queue, the scheduler typically sends a warning signal
before killing the job. When checkpointing is enabled, NLSQ catches
``SIGTERM``/``SIGUSR1``, saves a checkpoint at the next safe point, and then
raises ``CMAESPreempted`` -- a ``SystemExit`` subclass carrying exit code 75
-- out of ``fit()``. A wrapping shell/SLURM script can check for that exit
code to distinguish a clean checkpointed stop from a crash and resubmit:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=nlsq_fit
   #SBATCH --time=04:00:00
   #SBATCH --signal=B:TERM@60  # send SIGTERM 60s before the time limit

   python fit_job.py
   status=$?

   if [ "$status" -eq 75 ]; then
       echo "Preempted after checkpoint save -- resubmitting"
       sbatch "$0"
   fi

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

For the CMA-ES route (``method="cmaes"``, ``restart_strategy="none"``), pass
``checkpoint_dir`` as shown in "Checkpointing" above so a preempted job
resumes from the last saved generation instead of restarting from scratch --
see that section for the ``CMAESPreempted``/exit-code-75 resubmission
pattern. The multistart/chunked/streaming ``hpc`` routes don't support
checkpointing yet: prefer non-preemptible queues for long jobs on those
routes, or reduce ``n_starts``/walltime to fit inside a single preemption
window.

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
     - CMA-ES route only (``method="cmaes"``, ``restart_strategy="none"``)
   * - Crash recovery
     - No
     - CMA-ES route only (see above)
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
- On the CMA-ES route, pass ``checkpoint_dir`` (see "Checkpointing" above) so
  a resubmitted job resumes instead of restarting from scratch. The
  multistart/chunked/streaming routes don't support resume yet.

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
