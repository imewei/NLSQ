The 3-Workflow System
=====================

NLSQ's 3-workflow system is the core feature that distinguishes it from other
curve fitting libraries. Instead of manually configuring memory settings,
optimization strategies, and solver options, you simply choose one of three
workflows and NLSQ handles the rest.

.. versionchanged:: 0.6.3
   The workflow system was simplified from 9 presets to 3 smart workflows.

.. toctree::
   :maxdepth: 1

   auto_workflow
   auto_global_workflow
   hpc_workflow

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 15 35 20 30

   * - Workflow
     - Description
     - Bounds
     - Best For
   * - ``auto``
     - Memory-aware local optimization
     - Optional
     - Standard fitting tasks (default)
   * - ``auto_global``
     - Memory-aware global optimization
     - **Required**
     - Multiple minima, unknown initial guess
   * - ``hpc``
     - Global optimization + checkpointing
     - **Required**
     - Long HPC jobs, crash recovery

Choosing the Right Workflow
---------------------------

**Use auto (default) when:**

- You have a reasonable initial guess
- Your model has a single clear minimum
- You want the fastest possible fit

**Use auto_global when:**

- You don't know the initial parameters
- Your model may have multiple local minima
- You need robust fitting with exploration

**Use hpc when:**

- Running on HPC clusters (PBS, SLURM)
- Jobs may take hours or days
- You need crash recovery via checkpoints

Quick Reference
---------------

.. code-block:: python

   from nlsq import fit

   # Default: local optimization
   popt, pcov = fit(model, x, y, p0=[1, 0.5])

   # Global optimization (requires bounds)
   popt, pcov = fit(
       model, x, y, p0=[1, 0.5], workflow="auto_global", bounds=([0, 0], [10, 5])
   )

   # HPC with checkpointing (requires bounds)
   popt, pcov = fit(
       model,
       x,
       y,
       p0=[1, 0.5],
       workflow="hpc",
       bounds=([0, 0], [10, 5]),
       checkpoint_dir="/scratch/checkpoints",
   )

How Memory Selection Works
--------------------------

Both ``auto`` and ``auto_global`` automatically analyze your data size and
available memory to choose the best strategy:

.. code-block:: text

   ┌─────────────────────────────────────────────┐
   │           Memory Budget Analysis            │
   ├─────────────────────────────────────────────┤
   │ data_gb = n_points × 2 × 8 bytes           │
   │ jacobian_gb = n_points × n_params × 8      │
   │ peak_gb = data_gb + 1.3 × jacobian_gb      │
   └─────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  data_gb > available × 75%? │
        └─────────────────────────────┘
               YES │        │ NO
                   ▼        ▼
          ┌──────────┐  ┌─────────────────────┐
          │ STREAMING│  │ peak_gb > threshold?│
          └──────────┘  └─────────────────────┘
                              YES │      │ NO
                                  ▼      ▼
                          ┌─────────┐ ┌──────────┐
                          │ CHUNKED │ │ STANDARD │
                          └─────────┘ └──────────┘

**STANDARD**: All data fits in memory - fastest option

**CHUNKED**: Data fits but Jacobian doesn't - processes in chunks

**STREAMING**: Data exceeds memory - adaptive batch streaming

You don't need to understand these details - NLSQ handles it automatically.
This diagram is provided for users who want to understand what happens
under the hood.

Decision Flowchart
------------------

Use this flowchart to choose the right workflow:

.. code-block:: text

   Do you have parameter bounds?
            │
       ┌────┴────┐
       │ NO      │ YES
       ▼         ▼
   ┌────────┐   Do you need global search?
   │ auto   │         │
   └────────┘    ┌────┴────┐
                 │ NO      │ YES
                 ▼         ▼
            ┌────────┐   Running on HPC cluster?
            │ auto   │         │
            │ with   │    ┌────┴────┐
            │ bounds │    │ NO      │ YES
            └────────┘    ▼         ▼
                     ┌───────────┐ ┌─────┐
                     │auto_global│ │ hpc │
                     └───────────┘ └─────┘

Next Steps
----------

Learn about each workflow in detail:

- :doc:`auto_workflow` - Default local optimization
- :doc:`auto_global_workflow` - Global search with multi-start or CMA-ES
- :doc:`hpc_workflow` - Checkpointed HPC jobs
