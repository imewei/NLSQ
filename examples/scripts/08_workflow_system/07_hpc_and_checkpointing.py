"""
Converted from 07_hpc_and_checkpointing.ipynb

This script was automatically generated from a Jupyter notebook.

Features demonstrated:
- ClusterDetector and ClusterInfo for PBS Pro detection
- WorkflowTier.STREAMING_CHECKPOINT for fault tolerance
- Checkpointing with enable_checkpoints=True and checkpoint_dir
- create_checkpoint_directory() for timestamp-based directories
- Checkpoint resume workflow
- PBS Pro job script generation

Run this example:
    python examples/scripts/08_workflow_system/07_hpc_and_checkpointing.py
"""

import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

from nlsq.core.workflow import (
    WORKFLOW_PRESETS,
    ClusterDetector,
    ClusterInfo,
    OptimizationGoal,
    WorkflowConfig,
    WorkflowTier,
    create_checkpoint_directory,
    create_distributed_config,
    get_multi_gpu_config,
)

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
if QUICK:
    print("Quick mode: reduced iterations for HPC and checkpointing demo.")


def save_checkpoint(checkpoint_dir, iteration, params, loss, metadata=None):
    """Save optimization checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save checkpoint
    iteration : int
        Current iteration number
    params : np.ndarray
        Current parameter values
    loss : float
        Current loss value
    metadata : dict, optional
        Additional metadata to save
    """
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{iteration:06d}.pkl"

    checkpoint_data = {
        "iteration": iteration,
        "params": np.array(params),
        "loss": float(loss),
        "metadata": metadata or {},
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"  Saved checkpoint: {checkpoint_path.name}")
    return checkpoint_path


def load_latest_checkpoint(checkpoint_dir):
    """Load the most recent checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoints

    Returns
    -------
    dict or None
        Checkpoint data if found, None otherwise
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pkl"))
    if not checkpoints:
        return None

    # Sort by name (which includes iteration number)
    latest = sorted(checkpoints)[-1]

    with open(latest, "rb") as f:
        checkpoint_data = pickle.load(f)

    print(f"  Loaded checkpoint: {latest.name}")
    return checkpoint_data


def optimization_with_checkpoints(
    checkpoint_dir, max_iterations=100, checkpoint_interval=10
):
    """Example optimization loop with checkpoint support."""

    # Try to resume from checkpoint
    checkpoint = load_latest_checkpoint(checkpoint_dir)

    if checkpoint:
        start_iteration = checkpoint["iteration"] + 1
        params = checkpoint["params"]
        print(f"  Resuming from iteration {start_iteration}")
    else:
        start_iteration = 0
        params = np.array([1.0, 1.0, 0.0])  # Initial guess
        print("  Starting fresh optimization")

    # Optimization loop
    for iteration in range(start_iteration, max_iterations):
        # Simulate optimization step
        params = params + 0.001 * np.random.randn(3)
        loss = np.sum(params**2)  # Dummy loss

        # Checkpoint at intervals
        if iteration > 0 and iteration % checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir, iteration, params, loss)

    # Final checkpoint
    save_checkpoint(checkpoint_dir, max_iterations - 1, params, loss)

    return params


def main():
    print("=" * 70)
    print("HPC Integration and Checkpointing")
    print("=" * 70)
    print()

    np.random.seed(42)

    # =========================================================================
    # 1. ClusterDetector and ClusterInfo
    # =========================================================================
    print("1. ClusterDetector and ClusterInfo:")
    print("-" * 50)

    detector = ClusterDetector()

    print(f"  PBS environment detected: {detector.is_pbs_environment()}")

    cluster_info = detector.detect()

    if cluster_info:
        print("\n  Cluster detected:")
        print(f"    Scheduler: {cluster_info.scheduler}")
        print(f"    Node count: {cluster_info.node_count}")
        print(f"    GPUs per node: {cluster_info.gpus_per_node}")
        print(f"    Total GPUs: {cluster_info.total_gpus}")
    else:
        print("\n  No cluster environment detected (running locally)")

    # Simulated PBS cluster for demonstration
    simulated_cluster = ClusterInfo(
        node_count=4,
        gpus_per_node=8,
        total_gpus=32,
        node_list=["node01", "node02", "node03", "node04"],
        scheduler="pbs",
        job_id="12345.pbs_server",
        interconnect="infiniband",
    )

    print("\n  Simulated PBS cluster:")
    print(f"    Nodes: {simulated_cluster.node_count}")
    print(f"    GPUs: {simulated_cluster.total_gpus}")
    print(f"    Job ID: {simulated_cluster.job_id}")
    print(f"    Interconnect: {simulated_cluster.interconnect}")

    # =========================================================================
    # 2. WorkflowTier.STREAMING_CHECKPOINT
    # =========================================================================
    print()
    print("2. WorkflowTier.STREAMING_CHECKPOINT:")
    print("-" * 50)

    print("  Available WorkflowTiers:")
    for tier in WorkflowTier:
        print(f"    - {tier.name}")

    hpc_config = WorkflowConfig(
        tier=WorkflowTier.STREAMING_CHECKPOINT,
        goal=OptimizationGoal.ROBUST,
        gtol=1e-7,
        ftol=1e-7,
        xtol=1e-7,
        enable_checkpoints=True,
        checkpoint_dir="./nlsq_checkpoints",
        enable_multistart=True,
        n_starts=10,
    )

    print("\n  HPC Configuration:")
    print(f"    tier: {hpc_config.tier}")
    print(f"    goal: {hpc_config.goal}")
    print(f"    enable_checkpoints: {hpc_config.enable_checkpoints}")
    print(f"    checkpoint_dir: {hpc_config.checkpoint_dir}")
    print(f"    enable_multistart: {hpc_config.enable_multistart}")

    if QUICK:
        print()
        print("=" * 70)
        print("Summary (Quick Mode)")
        print("=" * 70)
        print()
        print("HPC Integration:")
        print("  - ClusterDetector.detect() for PBS Pro detection")
        print("  - ClusterInfo for cluster metadata (nodes, GPUs, job ID)")
        print()
        print("Checkpointing:")
        print("  - WorkflowTier.STREAMING_CHECKPOINT for fault tolerance")
        print("  - enable_checkpoints=True, checkpoint_dir='./checkpoints'")
        print()
        print("Defense Layers for Checkpoint Resume (v0.3.6+):")
        print("  - Use HybridStreamingConfig.defense_strict() for resume protection")
        print("  - 4-layer defense prevents Adam warmup from diverging")
        return

    # =========================================================================
    # 3. Checkpointing Configuration
    # =========================================================================
    print()
    print("3. Checkpointing Configuration:")
    print("-" * 50)

    checkpoint_dir = create_checkpoint_directory()
    print(f"  Created checkpoint directory: {checkpoint_dir}")

    custom_checkpoint_dir = create_checkpoint_directory(
        base_dir="./my_project_checkpoints"
    )
    print(f"  Custom checkpoint directory: {custom_checkpoint_dir}")

    # =========================================================================
    # 4. Checkpoint Resume Workflow
    # =========================================================================
    print()
    print("4. Checkpoint Resume Workflow:")
    print("-" * 50)

    demo_dir = create_checkpoint_directory(base_dir="./demo_checkpoints")

    # Simulate saving checkpoints
    print("\n  Simulating optimization with checkpoints...")
    for i in range(0, 30, 10):
        params = np.array([2.0 + 0.01 * i, 1.0 - 0.005 * i, 0.5])
        loss = 0.1 / (1 + i * 0.1)
        save_checkpoint(demo_dir, i, params, loss, metadata={"epoch": i // 10})

    # Load latest checkpoint
    print("\n  Loading latest checkpoint for resume...")
    latest = load_latest_checkpoint(demo_dir)

    if latest:
        print(f"    Iteration: {latest['iteration']}")
        print(f"    Parameters: {latest['params']}")
        print(f"    Loss: {latest['loss']:.6f}")

    # Run optimization with checkpoints
    print("\n  Running optimization loop with checkpoints...")
    final_params = optimization_with_checkpoints(demo_dir, max_iterations=50)
    print(f"  Final parameters: {final_params}")

    # =========================================================================
    # 5. HPC Distributed Configuration
    # =========================================================================
    print()
    print("5. HPC Distributed Configuration:")
    print("-" * 50)

    dist_config = create_distributed_config(simulated_cluster)

    print("  Distributed config from cluster:")
    for key, value in list(dist_config.items())[:8]:
        print(f"    {key}: {value}")

    gpu_config = get_multi_gpu_config(simulated_cluster)

    if gpu_config:
        print("\n  Multi-GPU configuration:")
        print(f"    n_devices: {gpu_config.n_devices}")
        print(f"    per_device_batch_size: {gpu_config.per_device_batch_size}")
        print(f"    total_batch_size: {gpu_config.total_batch_size}")

    # =========================================================================
    # 6. PBS Pro Job Script
    # =========================================================================
    print()
    print("6. PBS Pro Job Script:")
    print("-" * 50)

    pbs_script = """#!/bin/bash
#PBS -N nlsq_fit
#PBS -l select=4:ncpus=32:ngpus=8:mem=256gb
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -j oe
#PBS -o nlsq_fit.log

# NLSQ Curve Fitting Job Script for PBS Pro
# ===========================================

cd $PBS_O_WORKDIR

# Load required modules
module load python/3.12
module load cuda/12.0
module load cudnn/8.9

# Activate virtual environment
source ./venv/bin/activate

# Set NLSQ environment variables
export NLSQ_WORKFLOW_GOAL=robust
export NLSQ_MEMORY_LIMIT_GB=200
export NLSQ_CHECKPOINT_DIR=$PBS_O_WORKDIR/checkpoints

# Create checkpoint directory
mkdir -p $NLSQ_CHECKPOINT_DIR

# Display job information
echo "========================================"
echo "NLSQ Fitting Job Started"
echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Node list:"
cat $PBS_NODEFILE
echo "========================================"

# Run NLSQ fitting script
python fit_large_dataset.py \\
    --data-file ./data/large_dataset.h5 \\
    --output-dir ./results \\
    --checkpoint-dir $NLSQ_CHECKPOINT_DIR \\
    --enable-checkpoints \\
    --checkpoint-interval 50

echo "========================================"
echo "Job Completed: $(date)"
echo "========================================"
"""

    pbs_script_path = Path("nlsq_fit.pbs")
    pbs_script_path.write_text(pbs_script)

    print("  Created PBS job script: nlsq_fit.pbs")
    print("  Key directives:")
    print("    #PBS -l select=4:ncpus=32:ngpus=8:mem=256gb")
    print("    #PBS -l walltime=24:00:00")
    print("    #PBS -q gpu")

    # =========================================================================
    # 7. HPC Distributed Preset
    # =========================================================================
    print()
    print("7. HPC Distributed Preset:")
    print("-" * 50)

    hpc_preset = WORKFLOW_PRESETS["hpc_distributed"]

    for key, value in list(hpc_preset.items())[:8]:
        print(f"  {key}: {value}")

    # =========================================================================
    # 8. Defense Layers for Checkpoint Resume (v0.3.6+)
    # =========================================================================
    print()
    print("8. Defense Layers for Checkpoint Resume (v0.3.6+):")
    print("-" * 70)
    print()
    print("When resuming from checkpoints, your initial parameters are near-optimal.")
    print("This is a classic warm-start scenario where defense layers are critical.")
    print()
    print("Without defense layers, Adam warmup can DIVERGE from your checkpoint:")
    print("  - Momentum builds up from large initial gradients")
    print("  - Parameters overshoot and loss increases")
    print("  - All progress from previous run is lost")
    print()
    print("With 4-layer defense, checkpoint resume is protected:")
    print("  Layer 1: Detects you're starting near-optimal -> may skip warmup")
    print("  Layer 2: Scales learning rate based on initial fit quality")
    print("  Layer 3: Aborts warmup if loss increases > 5%")
    print("  Layer 4: Clips step magnitudes to prevent overshooting")
    print()
    print("Recommended configuration for checkpoint resume:")
    print()
    print("  from nlsq import HybridStreamingConfig")
    print()
    print("  # Use defense_strict for checkpoint resume scenarios")
    print("  config = HybridStreamingConfig.defense_strict()")
    print("  config = config.with_overrides(")
    print("      enable_checkpoints=True,")
    print("      checkpoint_dir='./checkpoints',")
    print("  )")
    print()
    print("Defense presets comparison:")
    print("  defense_strict()     - Best for checkpoint resume (LR: 1e-6 to 1e-4)")
    print("  defense_relaxed()    - For fresh starts (LR: 1e-4 to 0.01)")
    print("  scientific_default() - Balanced for production")
    print("  defense_disabled()   - Pre-0.3.6 behavior (no protection)")

    # =========================================================================
    # Cleanup
    # =========================================================================
    print()
    print("Cleaning up...")

    for path_str in [
        "nlsq_checkpoints",
        "demo_checkpoints",
        "my_project_checkpoints",
    ]:
        path = Path(path_str)
        if path.exists():
            shutil.rmtree(path)
            print(f"  Removed: {path_str}")

    if pbs_script_path.exists():
        pbs_script_path.unlink()
        print("  Removed: nlsq_fit.pbs")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("HPC Integration:")
    print("  - ClusterDetector.detect() for PBS Pro detection")
    print("  - ClusterInfo for cluster metadata (nodes, GPUs, job ID)")
    print("  - create_distributed_config() for HPC-optimized settings")
    print()
    print("Checkpointing:")
    print("  - WorkflowTier.STREAMING_CHECKPOINT for fault tolerance")
    print("  - enable_checkpoints=True, checkpoint_dir='./checkpoints'")
    print("  - create_checkpoint_directory() for timestamped directories")
    print()
    print("PBS Pro Job Scripts:")
    print("  - #PBS -l select=N:ncpus=C:ngpus=G:mem=Mgb")
    print("  - Environment variables: NLSQ_WORKFLOW_GOAL, NLSQ_MEMORY_LIMIT_GB")
    print("  - Checkpoint directory: NLSQ_CHECKPOINT_DIR")
    print()
    print("Defense Layers for Checkpoint Resume (v0.3.6+):")
    print("  - Checkpoint resume = warm-start scenario (parameters near optimal)")
    print("  - Use HybridStreamingConfig.defense_strict() for resume protection")
    print("  - 4-layer defense prevents Adam warmup from diverging")
    print("  - See docs/guides/defense_layers.rst for full configuration")


if __name__ == "__main__":
    main()
