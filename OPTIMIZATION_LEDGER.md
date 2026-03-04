# Optimization Ledger — NLSQ

## Environment
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU (16GB, SM 8.9)
- JAX: 0.9.1, GPU backend
- Python: 3.13

## Profiling Summary (baseline)
| Dataset Size | Median (ms) | Iters | nfev | Dominant Cost |
|---|---|---|---|---|
| 100 pts | 69.27 | 4 | 5 | Python overhead (GPU sync) |
| 1K pts | 72.44 | 4 | 5 | Python overhead (GPU sync) |
| 10K pts | 66.56 | 4 | 5 | Python overhead (GPU sync) |
| 100K pts | 86.33 | 4 | 5 | SVD + Python overhead |
| 1M pts | 235.60 | 4 | 5 | SVD + data transfer |

## Bottleneck Registry

| ID | File:Line | Category | Severity | Owner | Status | Speedup |
|----|-----------|----------|----------|-------|--------|---------|
| B001 | trf.py:1048-1116 | GPU | CRITICAL | lead | RESOLVED | 4.0x (inner loop) |
| B002 | common_scipy.py:222-247 | JIT | CRITICAL | lead | RESOLVED | (part of B001) |
| B003 | common_scipy.py:630-642 | JIT | CRITICAL | lead | RESOLVED | (part of B001) |
| B004 | trf.py:1977-2096 | GPU | HIGH | lead | RESOLVED | 3.6x (outer loop) |
| B005 | sparse_jacobian.py:604 | CPU | MEDIUM | — | OPEN | — |
| B006 | trf.py:1236-1243 | GPU | MEDIUM | lead | RESOLVED | 2x (fused check) |
| B007 | trf.py:1334-1336 | GPU | LOW | lead | WONTFIX | — |

## Categories: MEMORY | CPU | GPU | IO | ALGORITHMIC | JIT | VECTORIZATION | CONVERGENCE
## Severity: CRITICAL | HIGH | MEDIUM | LOW
## Status: OPEN | IN_PROGRESS | RESOLVED | WONTFIX

---

### B001: JAX Array Materialization in Inner Loop (CRITICAL)

**Location:** `trf.py:1048-1116` (inner `while` loop of `_evaluate_step_acceptance`)

**Evidence:** 121 `_value` calls × 0.16ms = **20ms** (14% of wall time). 89 `__bool__` calls × 0.19ms = **17ms** (12% of wall time). Combined: ~37ms of ~70ms total = **53% of wall time**.

**Root Cause:** Python `while` loop conditions (`actual_reduction <= 0`, `nfev < max_nfev`) operate on JAX arrays, forcing GPU→CPU synchronization every iteration. Each comparison triggers `__bool__()` which calls `_value()` to transfer the scalar from GPU memory to host memory.

**Impacted Operations Per Inner Iteration:**
- `actual_reduction <= 0` → 1 sync
- `nfev < max_nfev` → 1 sync (though nfev is Python int)
- `not self.check_isfinite(f_new)` → 1 sync
- `cost - cost_new` → assignment triggers eval
- `actual_reduction > 0` → 1 sync (exit condition)
- `step_h_norm > TR_BOUNDARY_THRESHOLD * Delta` → 1 sync
- Total: ~6 GPU→CPU syncs per inner iteration

**Fix Strategy:** Convert `update_tr_radius` and `check_termination` to JIT-compiled functions. Batch convergence checks into a single JIT function returning a Python-compatible status code. Minimize total sync points to 1 per outer iteration.

---

### B002: `update_tr_radius` is Pure Python (CRITICAL)

**Location:** `common_scipy.py:222-247`

**Evidence:** Called 6 times in inner loop. Function itself is <0.001ms but its JAX array arguments force GPU materialization. The `if ratio < 0.25` and `elif ratio > 0.75 and bound_hit` conditionals require `ratio` (a JAX scalar from `actual_reduction / predicted_reduction`) to be materialized as Python float.

**Root Cause:** Pure Python function with `if/elif` branches operating on JAX scalars.

**Fix:** JIT-compile using `jnp.where` for branchless execution. Keep result as JAX scalar.

---

### B003: `check_termination` is Pure Python (CRITICAL)

**Location:** `common_scipy.py:630-642`

**Evidence:** Called 6 times in inner loop. Python `and`/`if` logic on JAX scalars forces materialization. `ftol_satisfied = dF < ftol * F and ratio > 0.25` requires both `dF`, `F`, and `ratio` to be Python floats.

**Root Cause:** Pure Python function with boolean logic operating on JAX scalars.

**Fix:** JIT-compile using `jnp.where` for branchless status code computation.

---

### B004: Outer Loop Python Overhead (HIGH)

**Location:** `trf.py:1977-2096` (main `while True` loop of `trf_no_bounds`)

**Evidence:** Each outer iteration has: gradient norm check (1 sync), logging (float conversions), subproblem solve, inner loop (multiple syncs), mixed precision check (multiple syncs), callback (float/array conversions). Total: ~15-20 syncs per outer iteration.

**Root Cause:** Python-level control flow around JIT-compiled operations prevents XLA from fusing the full iteration.

**Fix:** Reduce sync points by batching convergence criteria. Logging and callbacks are unavoidable sync points but can be guarded.

---

### B005: Sparsity Detection Overhead (MEDIUM)

**Location:** `sparse_jacobian.py:604` (`detect_sparsity_at_p0`)

**Evidence:** 6-18ms per call. Called once per `curve_fit` invocation.

**Root Cause:** Evaluates Jacobian at p0, computes sparsity pattern. For small problems, this overhead exceeds the optimization itself.

**Fix:** Cache sparsity pattern for same model function signature.

---

### B006: Mixed Precision NaN/Inf Check (MEDIUM)

**Location:** `trf.py:1236-1243`

**Evidence:** 6 calls to `jnp.isnan().any()` and `jnp.isinf().any()` across `f`, `J`, `g` — 12 separate JAX operations, each potentially triggering sync via `bool()`.

**Root Cause:** Separate `isnan` and `isinf` checks on each array instead of fused check.

**Fix:** Fuse into `jnp.all(jnp.isfinite(jnp.concatenate([f.ravel(), J.ravel(), g.ravel()])))` — single operation.

---

### B007: Callback Forces Array Materialization (LOW)

**Location:** `trf.py:1334-1336`

**Evidence:** `float(cost)`, `np.array(x)`, `float(g_norm)` in callback invocation. Only runs when callback is provided.

**Root Cause:** User callbacks receive Python floats/NumPy arrays by contract.

**Fix:** WONTFIX — callback API requires materialized values. Already guarded by `if callback is not None`.

---

## Phase 2 Findings (Agent Reports)

The following bottlenecks were identified by the analysis agents but have **not yet been implemented**. They are cataloged here for future optimization work.

### JAX Agent Findings

| ID | File:Line | Category | Severity | Status | Issue |
|----|-----------|----------|----------|--------|-------|
| J01 | least_squares.py:1146-1153 | JIT | HIGH | OPEN | Double residual evaluation at startup (for dimension check only) |
| J02 | least_squares.py:572-594 | JIT | HIGH | OPEN | Fragile bytecode-based function identity check (`co_code` comparison) |
| J03 | loss_functions.py:225-320 | JIT | MEDIUM | OPEN | Python `if cost_only` prevents XLA fusion across loss functions |
| J04 | loss_functions.py:362-370 | JIT | MEDIUM | OPEN | Loss pipeline wrapper not JIT-compiled (4-6 kernel launches/call) |
| J05 | svd_fallback.py:86-153 | GPU | CRITICAL | OPEN | try/except inside @jit breaks XLA compilation for SVD |
| J06 | svd_fallback.py:116,133 | GPU | LOW | OPEN | `Vt.T` unnecessary transpose allocation on every SVD call |
| J07 | trf_jit.py:356-428 | VECTORIZATION | HIGH | OPEN | Both CG solves run unconditionally; only one result used |
| J08 | trf_jit.py:287-288 | JIT | MEDIUM | OPEN | `max_iter=None` inside @jit causes retracing risk |
| J09 | least_squares.py:291-326 | ALGORITHMIC | MEDIUM | OPEN | `jacfwd` with scalar argnums instead of vector argnums |
| J10 | least_squares.py:860-869 | JIT | HIGH | OPEN | `stable_rfunc` forces device sync on every residual eval |
| J11 | least_squares.py:1477-1499 | VECTORIZATION | LOW | OPEN | if-elif arg unpacking in jac_func |
| J12 | svd_fallback.py:42-63 | GPU | MEDIUM | OPEN | `with_cpu_fallback` decorator is dead code |

### Systems Agent Findings

| ID | File:Line | Category | Severity | Status | Issue |
|----|-----------|----------|----------|--------|-------|
| S01 | trf.py:1977,2546 | CPU | CRITICAL | OPEN | Outer loop is Python while-loop, not JIT-compiled |
| S02 | trf.py:801-811 | GPU | CRITICAL | OPEN | `if g_norm < gtol` forces GPU→CPU sync every outer iteration |
| S03 | trf.py:1080,1642 | GPU | CRITICAL | OPEN | `if not check_isfinite(f_new)` forces sync in inner loop |
| S04 | trf.py:1236-1243 | GPU | HIGH | RESOLVED | 6 separate .any() calls (→ fused to 3 isfinite checks) |
| S05 | trf.py:1246-1258 | GPU | HIGH | OPEN | 5 float() calls per accepted iteration = 5 GPU→CPU syncs |
| S06 | trf.py:1127-1176 | CPU | HIGH | OPEN | Per-iteration dict allocation for result passing |
| S07 | trf.py:2547 + common_scipy.py:437 | CPU/GPU | HIGH | OPEN | `CL_scaling_vector` is pure NumPy, called every bounds iteration |
| S08 | common_jax.py:507-515 | GPU | HIGH | OPEN | `compute_jac_scale` forces GPU→CPU transfer per accepted step |
| S09 | common_scipy.py:407-434 | GPU | HIGH | OPEN | `make_strictly_feasible` copies full x GPU→CPU per inner iter |
| S10 | common_scipy.py:332-334 | GPU | HIGH | OPEN | `in_bounds` uses np.all() forcing materialization in inner loop |
| S11 | trf.py:174-210,2016 | JIT | MEDIUM | OPEN | SVDCache designed but dead code — SVD recomputed on rejection |
| S12 | trf.py:1480 | MEMORY | MEDIUM | OPEN | `jnp.zeros([n])` allocated per outer bounds iteration |
| S13 | trf.py:1477 | MEMORY | MEDIUM | OPEN | `jnp.diag()` allocates n×n matrix per outer bounds iteration |
| S14 | common_scipy.py:222-247 | CPU | MEDIUM | RESOLVED | update_tr_radius Python conditionals (→ JIT-compiled) |
| S15 | trf.py:1336 | CPU | MEDIUM | WONTFIX | Callback forces np.array(x) every iteration (API contract) |

### Python Agent Findings

| ID | File:Line | Category | Severity | Status | Issue |
|----|-----------|----------|----------|--------|-------|
| P01 | minpack.py:512-556 | CPU | CRITICAL | OPEN | 5 unconditional _logger.info() calls with f-strings in fit() |
| P02 | minpack.py:559-563,1150,1331 | CPU | CRITICAL | OPEN | 3 redundant prepare_bounds() allocations per call |
| P03 | minpack.py:432,480-482,520,658 | CPU | HIGH | OPEN | Lazy imports inside hot-path functions on every call |
| P04 | minpack.py:484-486,673,691,735 | CPU | HIGH | OPEN | 3-way isinstance() chain for config dispatch |
| P05 | minpack.py:469,697,1006,1540 | CPU | HIGH | OPEN | inspect.signature(f) called 4+ times per fit |
| P06 | least_squares.py:574-576 | CPU | MEDIUM | OPEN | __code__.co_code bytecode comparison (O(n) bytes) |
| P07 | validators.py:535-538 | CPU | MEDIUM | OPEN | np.unique(xdata) O(n log n) on every validation |
| P08 | guard.py:680 | CPU | HIGH | OPEN | NumericalStabilityGuard JIT-compiles 7 functions at import time |
| P09 | smart_cache.py:630 | CPU | MEDIUM | OPEN | SmartCache disk I/O at import time |
| P10 | smart_cache.py:125-211 | CPU | HIGH | OPEN | SmartCache overhead exceeds savings for small/medium problems |
| P11 | least_squares.py:1127-1134 | CPU | CRITICAL | OPEN | Structured logger builds keyword dicts regardless of log level |
| P12 | memory_manager.py:480-489 | MEMORY | HIGH | OPEN | Telemetry dict allocation on every memory_guard exit |
| P13 | validators.py:555-681 | CPU | HIGH | OPEN | 11 validation steps run unconditionally (even in fast_mode) |

### Priority Recommendations for Future Work

**Highest ROI (next sprint):**
1. **J05 + J10**: SVD try/except inside @jit + stable_rfunc device sync — CRITICAL GPU blockers
2. **J07**: Lazy CG solve — eliminates ~2x wasted work per subproblem
3. **S07-S10**: Rewrite bounds helpers (CL_scaling_vector, make_strictly_feasible, in_bounds, compute_jac_scale) as JAX JIT functions
4. **S11**: Activate dead SVDCache — eliminates redundant O(mn²) SVD on rejected steps
5. **P01 + P11**: Add isEnabledFor() guards to all logger calls in hot path

**Structural improvements (medium effort):**
6. **S01**: Convert outer loop to lax.while_loop (high effort, highest potential speedup)
7. **J01**: Derive m from ydata.shape instead of evaluating residuals at startup
8. **P02-P05**: Thread prepared_bounds, cache signatures, hoist lazy imports
