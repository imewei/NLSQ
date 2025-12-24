# NLSQ Examples

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)
[![NLSQ Version](https://img.shields.io/badge/nlsq-0.4.1+-orange.svg)](https://github.com/imewei/NLSQ)

> **v0.4.1**: CLI interface, 2D surface fitting, gallery examples. See [CHANGELOG.md](../CHANGELOG.md) for details.

Welcome to the NLSQ examples repository! This collection provides comprehensive, interactive tutorials for learning and mastering GPU-accelerated nonlinear least squares curve fitting with JAX.

---

## 📁 Directory Structure

This directory contains **55 notebooks and 55 scripts** organized for optimal learning progression:

```
examples/
├── notebooks/          # Interactive Jupyter notebooks (55 files)
│   ├── 00_learning_map.ipynb        # Navigation guide
│   ├── 01_getting_started/          # 2 beginner tutorials
│   ├── 02_core_tutorials/           # 4 intermediate tutorials
│   ├── 03_advanced/                 # 7 advanced topics
│   ├── 04_gallery/                  # 11 domain-specific examples
│   ├── 05_feature_demos/            # 5 feature demonstrations
│   ├── 06_streaming/                # 5 streaming examples
│   ├── 07_global_optimization/      # 5 multi-start tutorials (NEW v0.3.3)
│   ├── 08_workflow_system/          # 7 unified fit() tutorials (NEW v0.3.4)
│   └── 09_gallery_advanced/         # 11 advanced gallery with fit() API (NEW v0.3.4)
├── scripts/            # Python scripts (mirrors notebooks/)
│   └── [same structure as notebooks/]
├── _templates/         # Notebook templates for contributors
└── README.md           # This file
```

**Total**: 114 files (57 notebooks + 57 scripts)

---

## 🚀 Quick Start

**New to NLSQ?** Start here:

```bash
# 1. Install NLSQ
pip install nlsq

# 2. Navigate to examples
cd examples/notebooks

# 3. Start with the learning map
jupyter notebook 00_learning_map.ipynb

# 4. Then your first tutorial
jupyter notebook 01_getting_started/nlsq_quickstart.ipynb
```

**Already familiar with NLSQ?** Jump to:
- [Core Tutorials](#02-core-tutorials-intermediate) for essential features
- [Global Optimization](#07-global-optimization-v033) for multi-start optimization
- [Workflow System](#08-workflow-system-v034) for the unified `fit()` API
- [Advanced Gallery](#09-advanced-gallery-v034) for domain examples with `fit()`
- [Performance Guide](#-performance-optimization-guide) for optimization

---

## 📚 Learning Paths

### Path 1: Complete Beginner (45 min)
**Best for**: First-time curve fitting users

```
START → Quickstart (15 min) → Interactive Tutorial (30 min) → Domain Example
```

1. [NLSQ Quickstart](notebooks/01_getting_started/nlsq_quickstart.ipynb)
2. [Interactive Tutorial](notebooks/01_getting_started/nlsq_interactive_tutorial.ipynb)
3. Choose from [Gallery](#04-gallery-domain-specific-examples)

### Path 2: SciPy Migrator (90 min)
**Best for**: Experienced with SciPy, need GPU acceleration

```
START → Quickstart (15 min) → fit() Quickstart (15 min) → Performance (20 min)
```

1. [NLSQ Quickstart](notebooks/01_getting_started/nlsq_quickstart.ipynb)
2. [fit() Quickstart](notebooks/08_workflow_system/01_fit_quickstart.ipynb) ⭐ NEW
3. [Performance Optimization](notebooks/02_core_tutorials/performance_optimization_demo.ipynb)

### Path 3: Global Optimization (120 min)
**Best for**: Problems with local minima, multi-modal fitting

```
START → Quickstart → Multi-Start Basics → Sampling Strategies → Integration
```

1. [NLSQ Quickstart](notebooks/01_getting_started/nlsq_quickstart.ipynb)
2. [Multi-Start Basics](notebooks/07_global_optimization/01_multistart_basics.ipynb) ⭐ NEW
3. [Sampling Strategies](notebooks/07_global_optimization/02_sampling_strategies.ipynb) ⭐ NEW
4. [Multi-Start Integration](notebooks/07_global_optimization/05_multistart_integration.ipynb) ⭐ NEW

### Path 4: Workflow System (90 min)
**Best for**: Production workflows, automatic optimization selection

```
START → fit() Quickstart → Workflow Tiers → Optimization Goals → Presets
```

1. [fit() Quickstart](notebooks/08_workflow_system/01_fit_quickstart.ipynb) ⭐ NEW
2. [Workflow Tiers](notebooks/08_workflow_system/02_workflow_tiers.ipynb) ⭐ NEW
3. [Optimization Goals](notebooks/08_workflow_system/03_optimization_goals.ipynb) ⭐ NEW
4. [Workflow Presets](notebooks/08_workflow_system/04_workflow_presets.ipynb) ⭐ NEW

### Path 5: Domain Expert (60 min)
**Best for**: Scientists with specific applications

```
START → Quickstart (15 min) → Your Domain (20 min) → Advanced Gallery (25 min)
```

1. [NLSQ Quickstart](notebooks/01_getting_started/nlsq_quickstart.ipynb)
2. Choose your field in [Gallery](#04-gallery-domain-specific-examples)
3. Same field in [Advanced Gallery](#09-advanced-gallery-v034) with `fit()` API

### Path 6: HPC & Production (180 min)
**Best for**: Large-scale computing, fault-tolerant fitting

```
START → fit() Quickstart → Workflow Tiers → YAML Config → HPC & Checkpointing
```

1. [fit() Quickstart](notebooks/08_workflow_system/01_fit_quickstart.ipynb) ⭐ NEW
2. [Workflow Tiers](notebooks/08_workflow_system/02_workflow_tiers.ipynb) ⭐ NEW
3. [YAML Configuration](notebooks/08_workflow_system/05_yaml_configuration.ipynb) ⭐ NEW
4. [HPC & Checkpointing](notebooks/08_workflow_system/07_hpc_and_checkpointing.ipynb) ⭐ NEW

---

## 📖 Tutorial Categories

### 00. Learning Map
**Location**: `notebooks/00_learning_map.ipynb`

Your complete navigation guide:
- ✓ Find the right starting point
- ✓ Understand tutorial structure
- ✓ Navigate efficiently
- ✓ Plan your learning journey

**Time**: 5-10 minutes

---

### 01. Getting Started (Beginner)
**Location**: `notebooks/01_getting_started/` | `scripts/01_getting_started/`

**Perfect for first-time users:**

1. **NLSQ Quickstart** (`nlsq_quickstart.ipynb`)
   - Basic `curve_fit()` usage
   - Memory management
   - Performance comparisons
   - **Time**: 15-20 min | **Level**: ●○○ Beginner

2. **Interactive Tutorial** (`nlsq_interactive_tutorial.ipynb`)
   - Hands-on practice with exercises
   - Common fitting patterns
   - Parameter bounds and uncertainties
   - **Time**: 30 min | **Level**: ●○○ Beginner

---

### 02. Core Tutorials (Intermediate)
**Location**: `notebooks/02_core_tutorials/` | `scripts/02_core_tutorials/`

**Master essential NLSQ features:**

1. **Large Dataset Demo** (`large_dataset_demo.ipynb`)
   - Scaling to 100M+ data points
   - Automatic chunking
   - Memory estimation
   - **Time**: 25-35 min | **Level**: ●●○ Intermediate

2. **2D Gaussian Fitting** (`nlsq_2d_gaussian_demo.ipynb`)
   - Multi-dimensional fitting
   - Image data processing
   - **Time**: 20-30 min | **Level**: ●●○ Intermediate

3. **Advanced Features** (`advanced_features_demo.ipynb`)
   - Diagnostics and monitoring
   - Error recovery
   - Algorithm selection
   - **Time**: 30-40 min | **Level**: ●●○ Intermediate

4. **Performance Optimization** (`performance_optimization_demo.ipynb`)
   - MemoryPool (2-5x speedup)
   - SparseJacobian (10-100x memory reduction)
   - StreamingOptimizer (unlimited data)
   - **Time**: 40-50 min | **Level**: ●●● Advanced

---

### 03. Advanced Topics (Advanced)
**Location**: `notebooks/03_advanced/` | `scripts/03_advanced/`

**Deep dives for expert users:**

1. `custom_algorithms_advanced` - Custom optimization algorithms
2. `gpu_optimization_deep_dive` - GPU performance tuning
3. `ml_integration_tutorial` - Machine learning workflows
4. `nlsq_challenges` - Complex real-world problems
5. `research_workflow_case_study` - Research applications
6. `time_series_analysis` - Time series fitting
7. `troubleshooting_guide` - Debugging and optimization

**Time**: 4-6 hours total | **Level**: ●●● Advanced

---

### 04. Gallery (Domain-Specific Examples)
**Location**: `notebooks/04_gallery/` | `scripts/04_gallery/`

**Real-world applications using `curve_fit()` API:**

#### 🧬 Biology (3 notebooks)
- **Dose-Response Curves** - IC50 calculation, Hill slopes
- **Enzyme Kinetics** - Michaelis-Menten, Km/Vmax estimation
- **Growth Curves** - Logistic models, bacterial growth

#### ⚗️ Chemistry (2 notebooks)
- **Reaction Kinetics** - Rate laws, rate constants
- **Titration Curves** - pH fitting, pKa determination

#### ⚛️ Physics (3 notebooks)
- **Damped Oscillation** - Harmonic oscillator, pendulums
- **Radioactive Decay** - Exponential decay, half-life
- **Spectroscopy Peaks** - Peak fitting, Lorentzian shapes

#### 🔧 Engineering (3 notebooks)
- **Sensor Calibration** - Calibration curves, polynomial regression
- **Materials Characterization** - Stress-strain, Young's modulus
- **System Identification** - Transfer functions, control systems

**Time**: Browse as needed | **Level**: ●●○ Intermediate

---

### 05. Feature Demonstrations
**Location**: `notebooks/05_feature_demos/` | `scripts/05_feature_demos/`

**Focused demonstrations of specific NLSQ features:**

1. **Callbacks** - Progress monitoring, early stopping
2. **Enhanced Error Messages** - Actionable diagnostics
3. **Function Library** - Pre-built models, auto p0 estimation
4. **Result Enhancements** - `.plot()`, `.summary()`, statistics
5. **Defense Layers** ⭐ NEW v0.3.6 - 4-layer defense strategy for warmup divergence prevention

**Time**: 1-2 hours | **Level**: ●●○ Intermediate

---

### 06. Streaming Examples
**Location**: `notebooks/06_streaming/` | `scripts/06_streaming/`

**Advanced streaming optimization for unlimited datasets:**

1. **Basic Fault Tolerance** - NaN/Inf detection, adaptive retry
2. **Checkpoint & Resume** - Save and resume optimization state
3. **Custom Retry Settings** - Custom retry strategies
4. **Interpreting Diagnostics** - Performance metrics analysis
5. **Hybrid Streaming API** - 4-phase optimizer with defense layers (v0.3.6+)

**Time**: 2-3 hours | **Level**: ●●● Advanced

---

### 07. Global Optimization (v0.3.3)
**Location**: `notebooks/07_global_optimization/` | `scripts/07_global_optimization/`

**Multi-start optimization for escaping local minima:** ⭐ NEW

1. **Multi-Start Basics** (`01_multistart_basics.ipynb`)
   - Why local optimization fails (local minima trap)
   - `GlobalOptimizationConfig` with `n_starts`, `sampling_method`
   - Interpret multi-start results
   - **Time**: 20 min | **Level**: ●●○ Intermediate

2. **Sampling Strategies** (`02_sampling_strategies.ipynb`)
   - Compare LHS, Sobol, Halton, Random sampling
   - Space-filling properties visualization
   - Sampler selection guidance
   - **Time**: 25 min | **Level**: ●●○ Intermediate

3. **Presets and Config** (`03_presets_and_config.ipynb`)
   - All `GlobalOptimizationConfig` parameters
   - Built-in presets: 'fast', 'robust', 'global', 'thorough'
   - Custom configuration patterns
   - **Time**: 20 min | **Level**: ●●○ Intermediate

4. **Tournament Selection** (`04_tournament_selection.ipynb`)
   - `TournamentSelector` for streaming scenarios
   - Memory-efficient global optimization
   - Tournament parameters configuration
   - **Time**: 30 min | **Level**: ●●● Advanced

5. **Multi-Start Integration** (`05_multistart_integration.ipynb`)
   - Integration with `curve_fit()` workflows
   - Bounds handling with multi-start
   - Combine with `curve_fit_large()`
   - **Time**: 25 min | **Level**: ●●○ Intermediate

**Key APIs:**
```python
from nlsq import curve_fit, GlobalOptimizationConfig

config = GlobalOptimizationConfig(n_starts=10, sampler="lhs")
popt, pcov = curve_fit(model, x, y, p0=p0, multistart=True, n_starts=10)
```

---

### 08. Workflow System (v0.3.4)
**Location**: `notebooks/08_workflow_system/` | `scripts/08_workflow_system/`

**Unified `fit()` API with automatic workflow selection:** ⭐ NEW

1. **fit() Quickstart** (`01_fit_quickstart.ipynb`)
   - Unified `fit()` function for any dataset size
   - Presets: 'fast', 'robust', 'global', 'quality'
   - Comparison with `curve_fit()` and `curve_fit_large()`
   - **Time**: 15 min | **Level**: ●○○ Beginner

2. **Workflow Tiers** (`02_workflow_tiers.ipynb`)
   - STANDARD, CHUNKED, STREAMING, STREAMING_CHECKPOINT
   - Automatic tier selection thresholds
   - Manual tier override
   - **Time**: 20 min | **Level**: ●●○ Intermediate

3. **Optimization Goals** (`03_optimization_goals.ipynb`)
   - FAST, ROBUST, GLOBAL, MEMORY_EFFICIENT, QUALITY
   - Internal settings for each goal
   - Combining goals with tiers
   - **Time**: 20 min | **Level**: ●●○ Intermediate

4. **Workflow Presets** (`04_workflow_presets.ipynb`)
   - `WORKFLOW_PRESETS` dictionary
   - All 7 named configurations
   - Customizing presets
   - **Time**: 15 min | **Level**: ●○○ Beginner

5. **YAML Configuration** (`05_yaml_configuration.ipynb`)
   - `nlsq.yaml` file structure
   - Environment variable overrides
   - Production deployment patterns
   - **Time**: 20 min | **Level**: ●●○ Intermediate

6. **Auto Selection** (`06_auto_selection.ipynb`)
   - `WorkflowSelector` decision logic
   - `DatasetSizeTier` and `MemoryTier` classification
   - Custom selection scenarios
   - **Time**: 25 min | **Level**: ●●● Advanced

7. **HPC & Checkpointing** (`07_hpc_and_checkpointing.ipynb`)
   - PBS Pro cluster detection
   - Checkpointing for fault tolerance
   - Resume from checkpoints
   - **Time**: 30 min | **Level**: ●●● Advanced

**Key APIs:**
```python
from nlsq import fit, WorkflowConfig, OptimizationGoal

# Auto-select workflow
popt, pcov = fit(model, x, y, p0=p0)

# With preset
popt, pcov = fit(model, x, y, p0=p0, preset="robust")

# With custom config
config = WorkflowConfig(goal=OptimizationGoal.QUALITY)
popt, pcov = fit(model, x, y, p0=p0, config=config)
```

---

### 09. Advanced Gallery (v0.3.4)
**Location**: `notebooks/09_gallery_advanced/` | `scripts/09_gallery_advanced/`

**Domain examples using `fit()` API with global optimization:** ⭐ NEW

Same 11 domain examples as Section 04, updated to demonstrate:
- Unified `fit()` API with presets
- `GlobalOptimizationConfig` for multi-start optimization
- Best practices for production workflows

#### 🧬 Biology (3 notebooks)
- **Dose-Response** - Hill equation with global optimization
- **Enzyme Kinetics** - Michaelis-Menten with `preset="robust"`
- **Growth Curves** - Logistic models with multi-start

#### ⚗️ Chemistry (2 notebooks)
- **Reaction Kinetics** - Rate constants with global search
- **Titration Curves** - pKa with `preset="global"`

#### ⚛️ Physics (3 notebooks)
- **Damped Oscillation** - Multi-start for oscillation parameters
- **Radioactive Decay** - Robust fitting for decay constants
- **Spectroscopy Peaks** - Global optimization for peak fitting (critical for local minima)

#### 🔧 Engineering (3 notebooks)
- **Sensor Calibration** - Robust calibration curves
- **Materials Characterization** - Multi-start for material parameters
- **System Identification** - Global search for transfer functions

**Key Pattern:**
```python
from nlsq import fit, GlobalOptimizationConfig

# Method 1: Preset
popt, pcov = fit(model, x, y, p0=p0, preset="robust")

# Method 2: Global preset (20 multi-starts)
popt, pcov = fit(model, x, y, p0=p0, preset="global")

# Method 3: Custom multi-start
popt, pcov = fit(model, x, y, p0=p0, multistart=True, n_starts=15, sampler="lhs")
```

**Time**: Browse as needed | **Level**: ●●○ Intermediate

---

## 🧭 Which Tutorial Should I Use?

### By Data Size

| Data Points | Recommended Tutorial | Key Features |
|-------------|---------------------|--------------|
| < 1,000 | Quickstart | Basic usage, GPU acceleration |
| 1K - 10K | fit() Quickstart | Automatic workflow selection |
| 10K - 100K | Large Dataset Demo | Memory management |
| 100K - 1M | Workflow Tiers | CHUNKED tier, diagnostics |
| 1M - 10M | Workflow Tiers + Streaming | STREAMING tier |
| > 10M | HPC & Checkpointing | STREAMING_CHECKPOINT tier |

### By Problem Type

| Problem | Recommended Tutorial | Key Features |
|---------|---------------------|--------------|
| Simple fitting | fit() Quickstart | `fit()` with auto-selection |
| Local minima | Multi-Start Basics | `GlobalOptimizationConfig` |
| Noisy data | Optimization Goals | `OptimizationGoal.ROBUST` |
| Production | YAML Configuration | File-based config |
| HPC cluster | HPC & Checkpointing | PBS Pro, checkpoints |

### By Experience Level

| Level | Start Here | Then... |
|-------|-----------|---------|
| **Beginner** | Quickstart → fit() Quickstart | → Workflow Presets → Gallery |
| **Intermediate** | fit() Quickstart → Multi-Start Basics | → Workflow Tiers → Advanced Gallery |
| **Advanced** | Multi-Start Basics → Tournament Selection | → Auto Selection → HPC |

---

## ⚡ Performance Optimization Guide

NLSQ provides advanced features for performance-critical applications:

| Feature | Purpose | Typical Speedup | Memory Reduction |
|---------|---------|-----------------|------------------|
| **MemoryPool** | Reuse pre-allocated buffers | 2-5x | 90-99% allocations |
| **SparseJacobian** | Exploit sparsity patterns | 1-3x | 10-100x memory |
| **StreamingOptimizer** | Process unlimited data | N/A | Unlimited |
| **Multi-Start** | Escape local minima | N/A | Finds global optimum |
| **Workflow System** | Auto-select strategy | Optimal | Optimal |

### When to Optimize

✅ **Use `fit()` when you want:**
- Automatic workflow selection
- Production-ready defaults
- Simple API for any dataset size

✅ **Use Global Optimization when you have:**
- Multi-modal loss landscapes
- Risk of local minima
- Uncertain initial parameters

❌ **Don't optimize prematurely:**
- Profile first to identify bottlenecks
- Standard `fit()` handles most cases well
- Global optimization adds compute time

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.12+** (required)
- **JAX** (automatically installed with NLSQ)
- **NumPy, SciPy** (standard scientific Python)
- **Matplotlib** (for visualizations)
- **Jupyter** (for notebooks) or **Google Colab** (cloud)

### Local Installation

```bash
# Create virtual environment (recommended)
python3.12 -m venv nlsq-env
source nlsq-env/bin/activate  # On Windows: nlsq-env\Scripts\activate

# Install NLSQ with all dependencies
pip install nlsq

# Install Jupyter
pip install jupyter

# Clone repository
git clone https://github.com/imewei/NLSQ.git
cd NLSQ/examples/notebooks

# Launch Jupyter
jupyter notebook
```

### GPU Setup (Optional)

NLSQ automatically detects and uses GPUs when available.

```python
import jax

print(f"JAX devices: {jax.devices()}")
```

---

## 🔬 Format Options

### Notebooks vs Scripts

**Notebooks** (`notebooks/`) - Interactive exploration:
```bash
jupyter notebook notebooks/08_workflow_system/01_fit_quickstart.ipynb
```

**Scripts** (`scripts/`) - Automation/CLI:
```bash
python scripts/08_workflow_system/01_fit_quickstart.py
```

Both formats contain identical examples - choose based on your workflow!

---

## 🆘 Common Issues and Solutions

### JAX precision warning
```
UserWarning: JAX is not using 64-bit precision
```
**Solution**: Import NLSQ before JAX (NLSQ auto-configures precision)

### GPU out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use `preset="memory_efficient"` or streaming tier
```python
popt, pcov = fit(model, x, y, p0=p0, preset="memory_efficient")
```

### Stuck in local minimum
**Solution**: Use global optimization
```python
popt, pcov = fit(model, x, y, p0=p0, preset="global")
```

---

## 📚 Additional Resources

### Documentation
- **Main Docs**: [https://nlsq.readthedocs.io](https://nlsq.readthedocs.io)
- **GitHub**: [https://github.com/imewei/NLSQ](https://github.com/imewei/NLSQ)
- **API Reference**: [https://nlsq.readthedocs.io/en/latest/api.html](https://nlsq.readthedocs.io/en/latest/api.html)
- **PyPI**: [https://pypi.org/project/nlsq/](https://pypi.org/project/nlsq/)

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/imewei/NLSQ/issues)
- **Discussions**: [GitHub Discussions](https://github.com/imewei/NLSQ/discussions)

---

## 🤝 Contributing

Found an issue or want to improve the examples?

1. **Report bugs**: [GitHub Issues](https://github.com/imewei/NLSQ/issues)
2. **Suggest examples**: [GitHub Discussions](https://github.com/imewei/NLSQ/discussions)
3. **Submit PRs**: Fork, improve, submit!

**Template System**: Use templates in `_templates/` for consistent notebook structure.

---

## 📜 License

NLSQ is released under the MIT License. See [LICENSE](../LICENSE) for details.

---

## 🎓 Citation

If you use NLSQ in your research, please cite:

```bibtex
@software{nlsq2024,
  title={NLSQ: GPU-Accelerated Nonlinear Least Squares Curve Fitting},
  author={Chen, Wei},
  year={2024},
  url={https://github.com/imewei/NLSQ},
  note={Argonne National Laboratory}
}
```

---

## 🌟 Summary

✨ **55 comprehensive notebooks** covering basics to advanced optimization
✨ **Production-ready** unified `fit()` API with automatic workflow selection
✨ **Global optimization** with multi-start for escaping local minima
✨ **GPU-accelerated** with 150-270x speedup over SciPy
✨ **Memory-efficient** with chunking, streaming, and checkpointing
✨ **Well-documented** with clear learning paths
✨ **Domain-specific** examples for biology, chemistry, physics, engineering
✨ **HPC-ready** with PBS Pro support and fault tolerance

**Ready to get started?** Open [00_learning_map.ipynb](notebooks/00_learning_map.ipynb) to plan your journey! 🚀

---

<p align="center">
<i>Last updated: 2025-12-24 | NLSQ v0.4.1</i>
</p>
