# Gemini API Compatibility Audit — NLSQ v0.6.12 Examples
Date: 2026-05-09 | Files scanned: 125 (60 notebooks + 65 scripts)

## BROKEN (must fix)

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `notebooks/02_core_tutorials/advanced_features_demo.ipynb` | cell ~109 | `from nlsq import callbacks` — module object not bound in `nlsq` namespace | `import nlsq.callbacks as callbacks` |
| `scripts/10_cli-commands/output/generated_workflow.yaml` | 35 | Comment references non-existent `nlsq.workflow` | Correct to `nlsq.gui_qt.adapters` or remove |

## STALE (cosmetic — version strings)

| File | Lines | Stale string | Should be |
|------|-------|-------------|-----------|
| `scripts/07_global_optimization/README.md` | 3, 14, 174 | `v0.6.3` / `2026-01-11` | `v0.6.12` |
| `notebooks/07_global_optimization/README.md` | 3, 14, 174 | `v0.6.3` / `2026-01-11` | `v0.6.12` |
| `scripts/08_workflow_system/README.md` | 3, 9, 171, 202 | `v0.6.3` / `2026-01-11` | `v0.6.12` |
| `notebooks/08_workflow_system/README.md` | 3, 9, 156, 185 | `v0.6.3` / `2026-01-11` | `v0.6.12` |
| `scripts/09_gallery_advanced/README.md` | 4 | `v0.6.3` | `v0.6.12` |

## WARNINGS

1. **Missing built-in `lorentzian`**: Both spectroscopy examples define a local `lorentzian()` instead of using `from nlsq.core.functions import lorentzian` (added v0.6.10).
   - `scripts/04_gallery/physics/spectroscopy_peaks.py` line 61
   - `scripts/09_gallery_advanced/physics/spectroscopy_peaks.py` line 45

2. **No example for `check_plugin_conflicts()`**: New in v0.6.10. Add to `scripts/03_advanced/gpu_optimization_deep_dive.py` as a GPU setup check.

## OK (all clean)

- All `04_gallery` scripts: `from nlsq import curve_fit` ✓
- All `09_gallery_advanced` scripts: `from nlsq import fit` ✓
- All `07_global_optimization` scripts: global_optimization imports ✓
- `callbacks_demo.py`: correct `from nlsq.callbacks import ...` ✓
- `from nlsq.utils.error_messages import OptimizationError` ✓
- All `nlsq.core.functions` imports ✓
- No `MixedPrecision` references anywhere ✓
- No positional `CurveFit(f, xdata, ydata)` pre-v0.6 style ✓
- `HybridStreamingConfig`, `AdaptiveHybridStreamingOptimizer` exports ✓
