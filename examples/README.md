# NLSQ Examples

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)
[![NLSQ Version](https://img.shields.io/badge/nlsq-0.3.2+-orange.svg)](https://github.com/imewei/NLSQ)

> **⚠️ v0.2.0 Update**: Notebooks updated for NLSQ v0.2.0 with streaming optimization replacing subsampling for zero accuracy loss. See [MIGRATION_V0.2.0.md](../MIGRATION_V0.2.0.md) for details.

Welcome to the NLSQ examples repository! This collection provides comprehensive, interactive tutorials for learning and mastering GPU-accelerated nonlinear least squares curve fitting with JAX.

---

## 📁 Directory Structure

This directory contains **32+ notebooks and scripts** organized for optimal learning progression:

```
examples/
├── notebooks/          # Interactive Jupyter notebooks (32 files)
│   ├── 00_learning_map.ipynb        # Navigation guide
│   ├── 01_getting_started/          # 2 beginner tutorials
│   ├── 02_core_tutorials/           # 4 intermediate tutorials
│   ├── 03_advanced/                 # 7 advanced topics
│   ├── 04_gallery/                  # 11 domain-specific examples
│   ├── 05_feature_demos/            # 4 feature demonstrations
│   └── 06_streaming/                # 4 streaming examples
├── scripts/            # Python scripts (mirrors notebooks/)
│   └── [same structure as notebooks/]
├── _templates/         # Notebook templates for contributors
└── README.md           # This file
```

**Total**: 64 files (32 notebooks + 32 scripts)

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
- [Advanced Topics](#03-advanced-topics-advanced) for deep dives
- [Gallery](#04-gallery-domain-specific-examples) for domain examples
- [Performance Guide](#⚡-performance-optimization-guide) for optimization

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
START → Quickstart (15 min) → Performance Guide (20 min) → Large Datasets (25 min)
```

1. [NLSQ Quickstart](notebooks/01_getting_started/nlsq_quickstart.ipynb)
2. [Performance Optimization](notebooks/02_core_tutorials/performance_optimization_demo.ipynb)
3. [Large Dataset Handling](notebooks/02_core_tutorials/large_dataset_demo.ipynb)

### Path 3: Domain Expert (60 min)
**Best for**: Scientists with specific applications

```
START → Quickstart (15 min) → Your Domain (20 min) → Feature Demos (25 min)
```

1. [NLSQ Quickstart](notebooks/01_getting_started/nlsq_quickstart.ipynb)
2. Choose your field in [Gallery](#04-gallery-domain-specific-examples)
3. Explore [Feature Demos](#05-feature-demonstrations)

### Path 4: Performance Optimizer (150 min)
**Best for**: Large datasets, performance-critical applications

```
START → Interactive (30 min) → Large Data (30 min) → GPU Dive (40 min) → Streaming (50 min)
```

1. [Interactive Tutorial](notebooks/01_getting_started/nlsq_interactive_tutorial.ipynb)
2. [Large Dataset Demo](notebooks/02_core_tutorials/large_dataset_demo.ipynb)
3. [GPU Optimization Deep Dive](notebooks/03_advanced/gpu_optimization_deep_dive.ipynb)
4. [Streaming Examples](notebooks/06_streaming/)

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

4. **Performance Optimization** (`performance_optimization_demo.ipynb`) ⭐ NEW
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

**Real-world applications by scientific domain:**

#### 🧬 Biology (3 notebooks)
- **Dose-Response Curves** (`biology/dose_response.ipynb`)
  - IC50 calculation, Hill slopes, pharmacology
- **Enzyme Kinetics** (`biology/enzyme_kinetics.ipynb`)
  - Michaelis-Menten kinetics, Km/Vmax estimation
- **Growth Curves** (`biology/growth_curves.ipynb`)
  - Bacterial/cellular growth modeling, logistic models

#### ⚗️ Chemistry (2 notebooks)
- **Reaction Kinetics** (`chemistry/reaction_kinetics.ipynb`)
  - Chemical reaction rate analysis, rate laws
- **Titration Curves** (`chemistry/titration_curves.ipynb`)
  - pH titration curve fitting, pKa determination

#### ⚛️ Physics (3 notebooks)
- **Damped Oscillation** (`physics/damped_oscillation.ipynb`)
  - Damped harmonic oscillator, pendulums, resonance
- **Radioactive Decay** (`physics/radioactive_decay.ipynb`)
  - Exponential decay processes, half-life determination
- **Spectroscopy Peaks** (`physics/spectroscopy_peaks.ipynb`)
  - Peak fitting, Lorentzian line shapes

#### 🔧 Engineering (3 notebooks)
- **Sensor Calibration** (`engineering/sensor_calibration.ipynb`)
  - Sensor calibration curves, polynomial regression
- **Materials Characterization** (`engineering/materials_characterization.ipynb`)
  - Stress-strain analysis, Young's modulus
- **System Identification** (`engineering/system_identification.ipynb`)
  - Control system parameter estimation, transfer functions

**Time**: Browse as needed | **Level**: ●●○ Intermediate

---

### 05. Feature Demonstrations
**Location**: `notebooks/05_feature_demos/` | `scripts/05_feature_demos/`

**Focused demonstrations of specific NLSQ v0.1.1+ features:**

1. **Callbacks** (`callbacks_demo.ipynb`)
   - Progress monitoring with `ProgressBar`
   - Early stopping with `EarlyStopping`
   - Iteration logging with `IterationLogger`

2. **Enhanced Error Messages** (`enhanced_error_messages_demo.ipynb`)
   - Actionable diagnostics
   - Clear recommendations

3. **Function Library** (`function_library_demo.ipynb`)
   - Pre-built models (exponential, gaussian, sigmoid)
   - Automatic p0 estimation

4. **Result Enhancements** (`result_enhancements_demo.ipynb`)
   - Enhanced `.plot()`, `.summary()`, `.confidence_intervals()`
   - Statistical metrics (R², RMSE, AIC, BIC)

**Time**: 1-2 hours | **Level**: ●●○ Intermediate

---

### 06. Streaming Examples
**Location**: `notebooks/06_streaming/` | `scripts/06_streaming/`

**Advanced streaming optimization for unlimited datasets:**

1. **Basic Fault Tolerance** (`01_basic_fault_tolerance.ipynb`)
   - Fault tolerance basics
   - NaN/Inf detection
   - Adaptive retry strategies

2. **Checkpoint & Resume** (`02_checkpoint_resume.ipynb`)
   - Save and resume optimization state
   - Automatic checkpoint detection

3. **Custom Retry Settings** (`03_custom_retry_settings.ipynb`)
   - Custom retry strategies
   - Success rate thresholds

4. **Interpreting Diagnostics** (`04_interpreting_diagnostics.ipynb`)
   - Comprehensive diagnostic analysis
   - Performance metrics interpretation

**Time**: 2-3 hours | **Level**: ●●● Advanced

---

## 🧭 Which Tutorial Should I Use?

### By Data Size

| Data Points | Recommended Tutorial | Key Features |
|-------------|---------------------|--------------|
| < 1,000 | Quickstart | Basic usage, GPU acceleration |
| 1K - 10K | Quickstart or 2D Gaussian | Standard optimization |
| 10K - 100K | Large Dataset Demo | Memory management |
| 100K - 1M | Large Dataset + Advanced | Chunking, diagnostics |
| 1M - 10M | Large Dataset + Performance | Sparse Jacobian, pooling |
| > 10M or disk | Performance Optimization | Streaming, HDF5 |

### By Experience Level

| Level | Start Here | Then... |
|-------|-----------|---------|
| **Beginner** | Quickstart → Interactive Tutorial | → Gallery (your domain) |
| **Intermediate** | Quickstart → Advanced Features | → Large Dataset → Performance |
| **Advanced** | Quickstart → Performance Optimization | → Custom Algorithms |

---

## ⚡ Performance Optimization Guide

NLSQ provides three advanced features for performance-critical applications:

| Feature | Purpose | Typical Speedup | Memory Reduction |
|---------|---------|-----------------|------------------|
| **MemoryPool** | Reuse pre-allocated buffers | 2-5x | 90-99% allocations |
| **SparseJacobian** | Exploit sparsity patterns | 1-3x | 10-100x memory |
| **StreamingOptimizer** | Process unlimited data | N/A | Unlimited |

### When to Optimize

✅ **Optimize when you have:**
- Very large datasets (>100K points)
- Memory constraints
- Repeated fitting operations
- Real-time/low-latency requirements
- Sparse problem structure

❌ **Don't optimize prematurely:**
- Profile first to identify bottlenecks
- Standard `CurveFit` handles most cases well
- Optimization adds complexity

**See**: [performance_optimization_demo.ipynb](notebooks/02_core_tutorials/performance_optimization_demo.ipynb) for detailed examples and benchmarks.

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

### Google Colab (Cloud)

No installation needed! Many notebooks include "Open in Colab" badges.

### GPU Setup (Optional)

NLSQ automatically detects and uses GPUs when available.

**Check GPU availability:**
```python
import jax

print(f"JAX devices: {jax.devices()}")
```

---

## 🔬 Format Options

### Notebooks vs Scripts

**Notebooks** (`notebooks/`) - Interactive exploration:
```bash
jupyter notebook notebooks/01_getting_started/nlsq_quickstart.ipynb
```

**Scripts** (`scripts/`) - Automation/CLI:
```bash
python scripts/01_getting_started/nlsq_quickstart.py
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
**Solution**: Use CPU or streaming optimization
```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"
```

### Slow first fit
**Expected**: First fit includes JIT compilation (1-2 seconds)
**Solution**: Reuse `CurveFit` objects for subsequent fits

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

✨ **32 comprehensive notebooks** covering basics to advanced optimization
✨ **Production-ready** with tested examples
✨ **GPU-accelerated** with 150-270x speedup over SciPy
✨ **Memory-efficient** with chunking and streaming support
✨ **Well-documented** with clear learning paths
✨ **Domain-specific** examples for biology, chemistry, physics, engineering
✨ **Beginner-friendly** with progressive tutorials

**Ready to get started?** Open [00_learning_map.ipynb](notebooks/00_learning_map.ipynb) to plan your journey! 🚀

---

<p align="center">
<i>Last updated: 2025-12-18 | NLSQ v0.3.2</i>
</p>
