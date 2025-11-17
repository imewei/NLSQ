# NLSQ Examples Directory Structure

This directory contains comprehensive examples and tutorials for NLSQ, organized for optimal learning progression.

## Directory Organization

```
examples/
├── notebooks/          # Interactive Jupyter notebooks
│   ├── 01_getting_started/
│   ├── 02_core_tutorials/
│   ├── 03_advanced/
│   ├── 04_gallery/
│   ├── 05_feature_demos/
│   └── 06_streaming/
├── scripts/            # Python scripts (mirrors notebooks/)
│   ├── 01_getting_started/
│   ├── 02_core_tutorials/
│   ├── 03_advanced/
│   ├── 04_gallery/
│   ├── 05_feature_demos/
│   └── 06_streaming/
└── README.md
```

## Learning Path

### 01. Getting Started (Beginner)
**Location**: `notebooks/01_getting_started/` | `scripts/01_getting_started/`

Start here if you're new to NLSQ:
- `nlsq_quickstart` - Basic usage, GPU acceleration, memory config
- `nlsq_interactive_tutorial` - Step-by-step interactive guide

**Time**: 30-45 minutes

### 02. Core Tutorials (Intermediate)
**Location**: `notebooks/02_core_tutorials/` | `scripts/02_core_tutorials/`

Master essential NLSQ features:
- `advanced_features_demo` - Diagnostics, error recovery, algorithm selection
- `large_dataset_demo` - Scaling to 100M+ data points
- `nlsq_2d_gaussian_demo` - Multi-dimensional fitting
- `performance_optimization_demo` - MemoryPool, SparseJacobian, Streaming

**Time**: 2-3 hours

### 03. Advanced Topics (Advanced)
**Location**: `notebooks/03_advanced/` | `scripts/03_advanced/`

Deep dives into specialized topics:
- `custom_algorithms_advanced` - Implementing custom optimization algorithms
- `gpu_optimization_deep_dive` - GPU performance tuning
- `ml_integration_tutorial` - Machine learning workflows
- `nlsq_challenges` - Complex real-world problems
- `research_workflow_case_study` - Research applications
- `time_series_analysis` - Time series fitting
- `troubleshooting_guide` - Debugging and optimization

**Time**: 4-6 hours

### 04. Gallery (Domain-Specific Examples)
**Location**: `notebooks/04_gallery/` | `scripts/04_gallery/`

Real-world applications by scientific domain:

**Biology**:
- `dose_response` - IC50 calculation and dose-response curves
- `enzyme_kinetics` - Michaelis-Menten kinetics
- `growth_curves` - Bacterial/cellular growth modeling

**Chemistry**:
- `reaction_kinetics` - Chemical reaction rate analysis
- `titration_curves` - pH titration curve fitting

**Engineering**:
- `materials_characterization` - Materials property analysis
- `sensor_calibration` - Sensor calibration curves
- `system_identification` - Control system parameter estimation

**Physics**:
- `damped_oscillation` - Damped harmonic oscillator
- `radioactive_decay` - Exponential decay processes
- `spectroscopy_peaks` - Peak fitting in spectroscopy

**Time**: Browse as needed

### 05. Feature Demonstrations
**Location**: `notebooks/05_feature_demos/` | `scripts/05_feature_demos/`

Focused demonstrations of specific NLSQ features:
- `callbacks_demo` - Progress monitoring and early stopping
- `enhanced_error_messages_demo` - Error diagnostics
- `function_library_demo` - Pre-built model functions
- `result_enhancements_demo` - Result analysis tools

**Time**: 1-2 hours

### 06. Streaming Examples
**Location**: `notebooks/06_streaming/` | `scripts/06_streaming/`

Advanced streaming optimization for unlimited datasets:
- `01_basic_fault_tolerance` - Fault tolerance basics
- `02_checkpoint_resume` - Checkpoint and resume
- `03_custom_retry_settings` - Custom retry strategies
- `04_interpreting_diagnostics` - Diagnostic interpretation

**Time**: 2-3 hours

## Quick Navigation

### By Format

**Notebooks** (Interactive exploration):
```bash
jupyter notebook examples/notebooks/01_getting_started/nlsq_quickstart.ipynb
```

**Scripts** (Automation/CLI):
```bash
python examples/scripts/01_getting_started/nlsq_quickstart.py
```

### By Level

| Level | Directory | Time | Focus |
|-------|-----------|------|-------|
| Beginner | `01_getting_started/` | 45 min | Getting started |
| Intermediate | `02_core_tutorials/` | 3 hrs | Core features |
| Advanced | `03_advanced/` | 6 hrs | Deep topics |
| Domain | `04_gallery/` | As needed | Real-world examples |
| Feature | `05_feature_demos/` | 2 hrs | Specific features |
| Streaming | `06_streaming/` | 3 hrs | Unlimited data |

## File Count

- **Total examples**: 64 files (32 notebooks + 32 scripts)
- **Getting Started**: 4 files (2 notebooks + 2 scripts)
- **Core Tutorials**: 8 files (4 notebooks + 4 scripts)
- **Advanced**: 14 files (7 notebooks + 7 scripts)
- **Gallery**: 22 files (11 notebooks + 11 scripts)
- **Feature Demos**: 8 files (4 notebooks + 4 scripts)
- **Streaming**: 8 files (4 notebooks + 4 scripts)

## Benefits of This Structure

1. **Clear Learning Progression**: Numbered directories guide from beginner to advanced
2. **Format Separation**: Easy to find notebooks vs scripts
3. **Mirrored Structure**: Same organization for both formats
4. **Domain Grouping**: Find domain-specific examples quickly
5. **Consistent Naming**: Predictable file locations

## Migration Notes

This structure was created by reorganizing the original flat structure:
- Root-level tutorials moved to numbered categories
- Old `demos/` → `notebooks/05_feature_demos/` + `scripts/05_feature_demos/`
- Old `gallery/` → `notebooks/04_gallery/` + `scripts/04_gallery/`
- Old `streaming/` → `notebooks/06_streaming/` + `scripts/06_streaming/`

All file contents remain unchanged; only locations have been updated.
