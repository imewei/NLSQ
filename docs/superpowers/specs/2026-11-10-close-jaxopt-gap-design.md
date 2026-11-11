# Closing the JAXopt gap: exact default for `curve_fit_large`, QR covariance, covariance opt-out (2026-11-10)

**Status:** Draft. The requester skipped the question round and gave the three changes directly. Choices
marked **[my call]** were made by the author and are listed again at the end for review.

**Sources:** `three-brain-out/2026-11-10-match-jaxopt/consensus.md` (profiles, Codex and Agy analysis),
`benchmarks/baselines/2026-11-01-library-comparison-1k-1M-10M-100M.md` (the only run with an
`exact_streaming` column), `benchmarks/baselines/2026-11-09-library-comparison-1k-100M.md` (rerun and A/B
addendum), `benchmarks/baselines/2026-10-31-exact-streaming-conditioning.md`.

## Intent

JAXopt is faster than NLSQ at every unbounded size on GPU. The three-brain evaluation found:

- At or above 10M points, the gap is mostly the default `curve_fit_large` path: chunked fits combined by GLS.
  On top of a TRF cost of 112 ms at 10M, it spends 194–201 ms in `create_chunks`, 263 ms of `_fit_chunked`
  self time and 171 ms in per-chunk covariance SVDs. That path is also not exact (1M two_exp misses the cost
  gate by 7.3e-5). The opt-in exact solver is already faster and exact.
- At 1M on GPU, the covariance SVD is about 30% of `curve_fit` (16 ms of 53 ms).
- Below 100k, parity is out of reach without dropping the SciPy contract (Python TRF loop, covariance). That
  is not attempted.

Expected end state: parity or better at 100M, 2–4x of JAXopt at 10M, about 5x at 1M, and 3–10x at 1k–100k
(unchanged).

## Goals

1. **G1:** `curve_fit_large` uses the exact chunk-accumulated LM by default for eligible inputs at or above
   `size_threshold`. Eligibility is defined below. The chunked GLS path stays reachable and remains the
   automatic fallback for ineligible inputs.
2. **G2:** The covariance SVD of `J` is computed as `R = qr(J, mode="r")` followed by `svd(R)`. This is the
   same `(s, VT)` in exact arithmetic, because `J = QR` with orthonormal `Q`. It applies to `curve_fit` and
   to the per-chunk fits of the chunked path, but only if the A/B gate passes.
3. **G3:** A new keyword, `compute_covariance=True`, skips the covariance when set to `False`. The default
   output stays SciPy-compatible.

## Non-goals

- Changing anything below `size_threshold`. The in-memory path still runs `curve_fit`; the 1M GPU result
  favours plain `curve_fit`.
- Changing `size_threshold` (1M).
- The TRF trust-region SVD (`trf_jit.py:409`, `_svd_no_bounds_jit`). It needs `U` and is a separate hot path.
- Compiling the TRF loop into one XLA program (A5, NO-GO twice: +230–440 ms compile per model).
- Gram/`eigh` for any covariance that is currently SVD-based. B2 was dropped because it squares `cond(J)`;
  this spec keeps that decision.
- Exposing `exact_streaming` through `nlsq.fit()` for small inputs. Large inputs that `fit()` forwards to
  `curve_fit_large` do get the new default, and that is intended.
- Retiring the chunked GLS path or `AdaptiveHybridStreamingOptimizer`.

## Anti-goals

- **Silent behaviour change.** Two rules (CLAUDE.md §3):
  - Any fallback from exact to chunked emits a `UserWarning`, because INFO logs are off by default
    (`nlsq/utils/logging.py`). The fallback keeps today's chunked result for that input, not a new
    approximation.
  - A skipped covariance is logged.

  The default change is a breaking change in the CHANGELOG.
- **Losing covariance accuracy on the default path relative to today's SVD covariance.** See "Covariance on
  the exact default" below.
- **Raising where the old default accepted input.** Options that the chunked path ignores today (`stability`,
  `rescale_data`, `max_jacobian_elements_for_svd`) must stay ignored on the new default, not raise. The
  explicit `method="exact_streaming"` keeps its current strict checks. One deliberate exception, decided as
  D1: inputs exact cannot fit and the caller did not choose an engine for now raise an actionable error.
  These are multi-feature `xdata`, a non-pointwise model, and non-finite `sigma` under `check_finite=False`.
  Also non-finite `xdata`/`ydata` under `check_finite=False` (ruling R17: chunked would absorb NaN, silent data loss).
  This is a CHANGELOG breaking entry.

## Supersedes

- `2026-10-31-exact-streaming-solver-design.md`:
  - Non-goal "changing the default `curve_fit_large` path".
  - Deferred item 1 ("promote to the default: later").

  Its revisit conditions now hold or are handled here:
  - Stage 3 is complete and the conditioning study is resolved.
  - Weights and bounds have shipped.
  - The `equivalent_to_full_batch` flag is untouched; the exact result already reports `True`.
  - The legacy GLS tests are pinned to `Chunked` (see Tests).
  - The `fit()` reach is accepted above.
- `2026-11-06-curve-fit-performance-design.md`:
  - Non-goal 3 ("no change to defaults") applied to that spec only.
  - Its microbenchmark found `qr` + `svd(R)` no faster than `svd` at 1M on GPU (14.84 vs 14.97 ms). The
    2026-11-10 measurement used `mode="r"` (no `Q`) and got 8.1 vs 14–18 ms. The two disagree, so G2 is
    gated on an interleaved A/B, not assumed.

## Change 2: QR-then-SVD covariance (first)

**What changes.** Only the body of `_covariance_svd_jit` (`nlsq/core/minpack.py:3579`) changes: compute
`R = jnp.linalg.qr(J, mode="r")`, then `svd(R, full_matrices=False)`, and return `(s, VT)`.

**What stays the same:**
- The `(s, VT)` contract.
- `_covariance_svd` and its A3 host-SVD routing.
- `pcov_from_jacobian`, its rank cutoff (`eps * max(n_eff, p) * s[0]`) and its fallbacks.
- `CurveFit.covariance_svd`, the injection hook that the per-chunk fits also reach.

**Edge cases the tests must cover:**
- **Sign or rotation of `VT` may differ** from the direct SVD. `pcov` is invariant to it.
  `tests/core/test_shared_jit_kernels.py:46` and `tests/core/test_device_policy.py` must compare `s` and
  `pcov`, never `VT` elementwise.
- **NaN/Inf in `J`** must still reach the all-`inf` fallback (non-finite `s`). Add a regression test.
- **Wide or degenerate `J`** (`n < p`, padded rows, all-zero column) must give the same rank decision and
  `pcov` as before. With `mode="r"`, `R` is `min(n, p) x p`.

**Host policy.** `HOST_SVD_MAX_ELEMENTS = 30_000` was measured for the full SVD. Remeasure the crossover in
the A/B with the new body. If it moves by more than 2x, update the constant with the measurement. Otherwise,
record it as unchanged.

**Gate (all must hold, or the change is dropped and recorded like A1/A4):**
1. Go/no-go is a kernel microbenchmark, run interleaved and synchronized with `block_until_ready`: the old
   body against the new one at 1M x 3 and 1M x 6 on GPU. The new body must be faster on both shapes, beyond
   the spread. An end-to-end A/B cannot resolve 8 ms inside a 661 ms bounded fit. The end-to-end A/B
   (`ab_nlsq.py`, absolute paths) is a no-regression check only:
   - `curve_fit` at 1M on GPU, all four problems;
   - `Chunked` at 10M.

   No ratio may be above the spread.
2. CPU `curve_fit` at 1M and 100k: no regression beyond the A/B spread.
3. `pcov` agrees with the old body to scale-aware `|dC_ij|/sqrt(C_ii C_jj) <= 1e-12` on the four benchmark
   problems. On one problem at `cond(J)` around 1e6 the gate is `<= 1e-9`: both bodies carry about
   `eps * cond(J)` error there, so `1e-12` is not attainable by either. Record the actual figure.

## Change 3: `compute_covariance=False` (second)

**API.** `curve_fit(..., compute_covariance: bool = True)` and the same keyword on `CurveFit.curve_fit` and
`curve_fit_large`. The default `True` leaves every output as it is today.

**When it is `False`:**
- `pcov` is returned as `None` in the tuple, the `CurveFitResult` and the `full_output` variants. **[my call]**
  An all-`inf` matrix would trigger the "could not be estimated" warning and read as a failure.
- No *final* covariance is computed and no final-covariance warning is emitted. One DEBUG log line says the
  covariance was skipped. The chunked path is the exception (below).
- `CurveFitResult` members that need `pcov` raise `AttributeError` naming `compute_covariance=False`. This
  follows the existing `pcov is None` guards in `confidence_intervals` (`curve_fit_result.py:480`) and
  `confidence_band` (`:682`); both messages get the new name. There is no `perr` member: standard errors
  are locals in `confidence_intervals`, `prediction_interval` (`:588`) and `summary` (`:884`).
  `prediction_interval` gets a guard. `summary` already skips the column when `pcov is None`. Any other
  reader found by grep must not return NaN.
- The chunked path still computes per-chunk covariances, because the GLS combiner needs them
  (`large_dataset.py:2121-2213`). The flag is not forwarded to the chunk `curve_fit` calls, and per-chunk
  covariance warnings still fire as today. Only the final `pcov` is dropped, so the flag saves nothing on
  that path, and the docstring says so.
- On the exact path, the final covariance (normal equations or TSQR, below) is skipped. Inside
  `exact_lm_fit` the keyword shadows the module function `exact_lm.compute_covariance`, so the fit calls
  `pcov_from_normal` directly. The module function stays for its tests (`tests/streaming/test_exact_lm.py:653-680`).
- The keyword must not leak into the kwargs forwarded to `least_squares`. It is added to the reserved and
  owned key sets of `InMemory` and `Chunked`, so `options={"compute_covariance": ...}` raises like the other
  owned keys.
- Paths in `curve_fit` that may not go through `CurveFit._compute_covariance` are handled one of two ways:
  honour the flag the same way, or raise `ValueError` if they cannot. These include `method="cmaes"`,
  `method="multi-start"`, the separate `multistart=True` / `global_search=True` branch, `hybrid_streaming`,
  `fallback=True` and `compute_diagnostics=True`. The builder picks per path and tests each one.

## Change 1: exact by default in `curve_fit_large` (last)

### Why

The comparison is exact against the current default (GLS), from the 2026-11-01 run on the same machine and
session:

| Size | Device | GLS default (exp / gauss / two_exp / bounded) | exact_streaming |
|---|---|---|---|
| 1M | GPU | 0.48 / 0.69 / DNRO / 2.71 s | 0.45 / 0.31 / 0.21 / 0.47 s |
| 1M | CPU | 0.58 / 0.81 / DNRO / 1.99 s | 0.19 / 0.46 / 0.46 / 0.61 s |
| 10M | GPU | 1.67 / 1.41 / 1.66 / 2.53 s | 0.26 / 0.42 / 0.51 / 0.42 s |
| 10M | CPU | 3.06 / 3.68 / 7.05 / 7.61 s | 1.34 / 3.94 / 6.13 / 7.18 s |
| 100M | GPU | 13.3 / 13.6 / 15.8 / 23.4 s | 0.60 / 1.87 / 2.76 / 1.94 s |
| 100M | CPU | 28.9 / 33.1 / 78.2 / 49.8 s | 8.65 / 26.4 / 45.7 / 42.6 s |

Exact is as fast or faster in every cell except CPU gaussian at 10M, which is 7% slower and inside drift. It
reaches the optimum where GLS does not (1M two_exp DNRO). Routing uses `n_points >= size_threshold`
(`specs.py:282-285`), so a 1M-point dataset already takes the large path under the default threshold, and
the 1M rows apply as they stand.

### Dispatch, step by step

`Auto` gains a field `exact: Exact | None = field(default_factory=Exact)`. **[my call]** It follows the
existing `in_memory`/`chunked` sub-strategy pattern. `Auto.resolve(n_points)` stays count-only:
- below the threshold (with detection on), it returns `in_memory`;
- otherwise it returns `exact` if set, else `chunked`.

Data eligibility cannot live in `resolve`, which only sees a count. It lives in `fit_with_strategy`
(`nlsq/streaming/large_fit.py`), in this order:

```text
1. if strategy is Exact (explicit method/strategy): unchanged; _fit_exact, strict checks, lazy sources ok
2. reject chunk factories (unchanged: they require explicit exact)
3. np.asarray + shape/length validation, n_points = xdata.shape[-1]   (unchanged, large_fit.py:116-133)
4. if Hybrid: unchanged
5. if Auto: auto = strategy; strategy = auto.resolve(n_points)
6. if strategy is Exact (only reachable via Auto):
       opts = {**auto.chunked.options, **kwargs}           # what chunked would have used
       hit = _exact_ineligible(auto, opts, f, xdata, n_points, sigma, p0, bounds, check_finite)
       if hit and hit.kind == "engine": warn (UserWarning) + use auto.chunked
       elif hit and hit.kind == "data": raise ValueError(actionable, names strategy=Chunked())
       else: _fit_exact(_effective_exact(auto, opts), ..., route="default")
7. InMemory / Chunked: unchanged
```

`auto_size_detection=False` means "always use the large path": exact if eligible, else chunked. **[my call]**
Lazy sources are not affected: the `Auto` path already materializes arrays in step 3 today, and only an
explicit `Exact` reads lazily.

**Sigma support added to exact (decided as D1, exact math).** The objective uses `(r_i/sigma_i)^2`, so it
depends on `sigma` only through `sigma^2`. Exact therefore accepts:
- a scalar `sigma`, broadcast per chunk without materializing an `n`-length array;
- negative nonzero `sigma`, used as `|sigma|`. This is the same objective chunked minimizes today
  (`_option_checks.py:88-95` admits it).

The change belongs in the shared source and validation code, not only the kernel:
- `normal_equations._block` slices sigma, so a scalar must pass through as a scalar block.
- The per-block positivity check in `normal_equations.py:332-335,365-375` becomes "nonzero", with `|sigma|`
  applied in the weighted stats and Jacobian kernels.
- Factory-yielded sigma blocks get the same treatment.
- The TSQR weighted path uses the same blocks.

Memory stays bounded (no `n`-length array). This applies to the explicit `method="exact_streaming"` too.
Zero `sigma` still raises `ValueError`, as
chunked does today (`_option_checks.py:89-95`). 2-D `sigma` keeps chunked's existing `ValueError`
(`large_dataset.py:2915-2928`).

**Eligibility (`_exact_ineligible` returns `(kind, reason)` or None, decided as D1).** It receives the full
validated `xdata` and `n_points`, and slices its probe internally after the dimension checks. It reads the
`Auto` object plus the call's extra keywords (`opts` above), so the legacy shim and a user-built `Auto`
follow the same rule. There are two kinds of reason.

**`kind="engine"`: the caller asked for a chunked-engine feature. Use `auto.chunked` and emit a
`UserWarning`.**
1. `auto.chunked` sets `method`, enables multistart, or has `opts` keys outside `{ftol, xtol, gtol,
   max_nfev}` (for example `loss`, `x_scale`, `tr_solver`, `nan_policy`, `verbose`).

**`kind="data"`: exact cannot fit this input and the caller chose nothing. Raise `ValueError`.** The
message names the reason, says that chunked GLS is an approximation that is not full-batch equivalent for
nonlinear models, and says to pass `strategy=Chunked()` to accept that.
2. `xdata.ndim != 1` (multi-feature `xdata`).
3. `sigma` has a non-finite value under `check_finite=False`. Chunked admits it (`large_dataset.py:2577-2593`);
   exact rejects it (`exact_lm.py:647-654`). With `check_finite=True`, both paths raise their existing
   finite-value error.
3b. `xdata` or `ydata` has a non-finite value under `check_finite=False` (ruling R17: chunked GLS would absorb
   NaN into a chunk, silent data loss). Checked only when `check_finite=False`.
3c. A 2-D `sigma` is a data reason checked before finiteness so the error names its shape, but it carries no
   `strategy=Chunked()` hint (chunked rejects 2-D `sigma` too); the message asks for a 1-D `sigma`.
4. The model fails `exact_lm.check_model_contract` on a probe of `min(n, 4096)` points at the normalized
   `p0`. The solver runs the same check on its first full chunk; that `ValueError` propagates unchanged, and
   on the default route the message gains the same `strategy=Chunked()` hint.

**Settings transfer (`_effective_exact(auto, opts)`).** The settings a user gave for the large path reach
exact, whichever way they were given (legacy keywords, `Auto(chunked=Chunked(chunk_size=500,
options={"ftol": 1e-10}))`, or call kwargs). For each `Exact` field:
- If `auto.exact` sets it to a non-default value, that value wins.
- Otherwise `chunk_size` comes from `auto.chunked.chunk_size`, and `ftol`, `xtol`, `gtol` and `max_nfev` come
  from `opts`.

The legacy shim (`_legacy_to_strategy`) is therefore unchanged: it keeps building `Auto` as today, and the
default `Exact()` picks the settings up from `Chunked`.

`stability`, `rescale_data` and `max_jacobian_elements_for_svd` live on `InMemory` only and are ignored on the
large path, as they are today. They are not a reason to fall back.

**Pointwise contract (a new compatibility condition).** Today's docstring requires only that the output
shape match `xdata` (`nlsq/__init__.py:1153-1168`). Exactness additionally needs `f(x, *p)[i]` to depend only
on `x[i]` and `p`, deterministically. The probe is a guard, not a proof (`exact_lm.py:360-366`): a model
whose output depends on other points only within later chunks is not detected. The docstring and the
CHANGELOG breaking entry state this as a new condition on the caller. A model that cannot meet it uses
`strategy=Chunked()`.

An engine-kind fallback warns once per call: `UserWarning("curve_fit_large: exact streaming not used
(<reason>); using chunked GLS, which is not full-batch equivalent for nonlinear models. Pass
strategy=Chunked() to choose it explicitly.")`. It also emits the same text at INFO.

**Opt-out.** No new API: `strategy=Chunked()`, or `strategy=Auto(exact=None)`, gives today's behaviour.

### Failure handling

A stalled or out-of-passes exact fit under the default raises `ExactSolverError` with `.result`, as the
explicit method does. **[my call]** There is no silent retry on GLS: a retry would hide a non-converged fit
and double the cost. On the default route the message adds "strategy=Chunked() restores the chunked GLS fit".

### Covariance on the exact default **[my call; the most important one to check]**

**What it replaces.** Today's default large-path `pcov` is not an SVD covariance of the full problem. The
GLS combiner inverts each chunk's SVD covariance (taken at that chunk's own optimum) into a precision, drops
unusable chunks, and returns the inverse of their sum (`large_dataset.py:2121-2213,2658-2710`). That is a
precision-sum (normal-equation-form) inversion, and it is not evaluated at the full-data optimum. The exact
path's `eigh(H)`, with `H` the full-data `J^T J` at the full-data optimum, is the theoretically exact
normal-equation covariance; its only difference from an SVD covariance is floating-point conditioning, handled
below. The non-goal "no Gram/eigh for a covariance that is
currently SVD-based" holds: `curve_fit` and the per-chunk fits keep SVD (Change 2).

**Rule.** Still, normal equations lose accuracy as `cond(J)^2`, so:
- After convergence, compute `kappa = sqrt(lambda_max / lambda_min)` of `H` with the same overflow-safe
  scaling `pcov_from_normal` uses (`covariance.py:152-160`: divide by a power of two, `errstate`). Factor it
  into a helper, `normal_condition(H) -> float`, in `nlsq/core/covariance.py`, used by both; this is a
  refactor, with no behaviour change to `pcov_from_normal`. A non-finite `H` or `lambda_min <= 0` gives
  `inf`.
  - If `kappa <= 1e3`: use `pcov_from_normal`.
    - The first-order estimate of its normwise relative error is `eps * kappa^2`, about 2.2e-10. This is an
      estimate, not a guaranteed bound: forming and summing `J^T J` adds data- and reduction-dependent
      rounding.
    - Near the threshold the classification can go either way. Both routes are valid there, so a
      misclassification costs speed or accuracy only within the estimate, never correctness.
    - Expected rank behaviour: with `lambda_min / lambda_max >= 1e-6`, neither cutoff (`p*eps*lambda_max`
      here, `eps*max(n,p)*s0` on the singular values in `pcov_from_jacobian`) should drop a direction.
    - Gate G-cov, below, verifies all of this by measurement rather than assuming it.
  - If `kappa > 1e3`: run one extra full-data pass at `popt` that accumulates a TSQR factor
    (`R <- qr(vstack(R, J_chunk), mode="r")`). Feed `R` to `pcov_from_jacobian(R, cost, n, p, absolute_sigma,
    _covariance_svd)`. That gives the SVD rank cutoff, the SVD fallbacks and the scaling of `curve_fit`.
- **Scope of TSQR.** TSQR fixes the conditioning of the covariance only. The LM step solve stays on normal
  equations; at large `kappa` the parameters themselves may stop short, which the existing `COND_WARN`
  warning reports.
- **Memory.** The accumulated state is `O(p^2)`. The per-chunk workspace is `O(chunk_size * p)`, the same as
  the existing stats kernel, on the same device and with the same chunking and padded-tail masking.
  Factory sources replay once more and verify their count as in `run_pass`.
- **Observability.** `OptimizeResult` from `exact_lm_fit` gains `covariance_method: "normal" | "tsqr" |
  "skipped"`.
- **Scope.** This applies to every `exact_lm_fit` call, the explicit `method="exact_streaming"` included. It
  closes the `shortcut:` note in `exact_lm.py:14-16` for the covariance. The step solve stays on normal
  equations, per the conditioning study.
- **Warnings.** `pcov_from_normal` and its `COND_WARN` text are unchanged; `adaptive_hybrid.py` also calls
  it. On the TSQR route `pcov_from_normal` is not called, so `exact_lm.py` emits its own `UserWarning` when
  `kappa > COND_WARN`. That warning says the parameters may stop short of the optimum, and that the
  covariance was computed by QR.

**Gate G-cov (decided as D2; must pass before Change 1 merges).** Compare the normal-route `pcov` and the
TSQR-route `pcov` against `pcov_from_jacobian` on the full Jacobian at the same `popt`. Cover:
- `kappa(J)` in {1, 10, 100, 500, 900, 1100, 2000};
- correlated parameters;
- column scales spanning 1e-6..1e6;
- `sigma` and `absolute_sigma` both ways;
- an active bound;
- one rank-deficient case.

The normal route must reach scale-aware `|dC_ij|/sqrt(C_ii C_jj) <= 1e-8` and the same rank decision at
every `kappa <= 1e3` tested.

If any case fails, set `TSQR_KAPPA` to the largest tested value strictly below the smallest failing
`kappa`, so every observed failing case goes to TSQR. Then rerun the whole gate. If the smallest tested
`kappa` (1) fails, the default becomes TSQR on every fit (option (b)), and the change is reported to the
requester. The threshold is set by this measurement, not by the estimate.

Problems:
- **Controlled cases.** A linear, pointwise model `f(x, theta) = B[x_idx] @ theta`, where
  `B = U diag(s) V^T` with orthonormal `U` (from QR of a seeded Gaussian), so the singular values `s`, and
  hence `kappa`, are exact by construction. `x` is the row index as float, cast inside the model.
- **Realism cases.** two_exp and gaussian at their measured `kappa`.

**Gate (route and cost):**
- On the four benchmark problems, `covariance_method == "normal"` at 10M. Check it with a one-off script
  calling `exact_lm_fit` directly. Measured on 2026-11-10, raw `kappa` at the true parameters on the
  harness grid:

  | Problem | kappa |
  |---|---|
  | exp_decay | 6.3 |
  | gaussian | 6.4 |
  | two_exp | **762** (narrow margin) |
  | bounded_exp | 7.3 |

  If noise puts two_exp's `popt` above 1e3, the result is still correct: record it and add the pass cost to
  the Change 1 numbers. Do not move the threshold to dodge it.
- Time the TSQR pass on GPU at 10M and 100M. The 11-06 GPU QR figure, about 15 ms per 1M rows, suggests
  about 1 s at 100M. If the pass costs more than one LM iteration, report it. It does not block, because it
  runs only when `kappa > 1e3`.
- On a `cond(J) ~ 1e5-1e6` problem, the TSQR `pcov` matches `pcov_from_jacobian` on the full Jacobian at the
  same `popt` to `1e-8` scale-aware. The same test checks the singular values of `R` against those of `J`,
  not only `R^T R`, which would hide squared-conditioning loss.

### Return contract

The return contract is unchanged:
- The large path returns `(popt, pcov)`, a tuple, for both exact and chunked.
- In-memory returns `curve_fit`'s native result.
- `fit()` (`nlsq/__init__.py:671`) gets the same tuple as before.

## Tests

- **Change 2:**
  - `s`/`pcov` parity old versus new on random tall, wide and rank-deficient `J`.
  - The NaN `J` fallback.
  - Existing `test_device_policy.py` and `test_shared_jit_kernels.py` pass, with `VT` comparisons rewritten
    as `pcov` comparisons where needed.
- **Change 3:**
  - Tuple, `CurveFitResult` and `full_output` return `None`.
  - Each pcov-dependent member raises.
  - `options=` collision raises `TypeError`.
  - Chunked still combines correctly with the flag off.
  - Each non-TRF `curve_fit` method is either honoured or raises.
- **Change 1:**
  - `threshold - 1` points goes in-memory; `threshold` and `threshold + 1` go to exact (spy on the dispatch).
  - (D1) An engine reason (method, multistart, a TRF option), via both the legacy shim and a user-built
    `Auto`, falls back to chunked with the warning.
  - (D1) A data reason raises the actionable error: multi-feature `xdata`, non-finite `sigma` or `xdata`/`ydata` under
    `check_finite=False`, a non-pointwise model caught by the probe, and one caught by the solver's
    first-chunk check. The same input with `strategy=Chunked()` runs.
  - Scalar and signed `sigma` run on exact and match `curve_fit` on `|sigma|`. Zero and 2-D `sigma` raise as
    today.
  - Gate G-cov is a test.
  - `Auto(exact=None)` and `strategy=Chunked()` reproduce today's results bit for bit.
  - The stall raises `ExactSolverError`.
  - `auto_size_detection=False` goes to exact.
  - TSQR:
    - trigger: `covariance_method`;
    - R singular values and `pcov` parity, including padded weighted tails, rank deficiency and factory
      replay;
    - the separate warning;
    - unchanged `pcov_from_normal` warning text.
  - Legacy GLS tests (`test_large_dataset*.py`, `test_large_dataset_equivalence_flag.py`,
    `test_streaming_pr30_regressions.py`, and any other test asserting GLS behaviour through the default) are
    pinned to `strategy=Chunked()` or `Auto(exact=None)`, not weakened.
- Markers follow CLAUDE.md: `serial` for arrays over 100K or subprocesses, `slow` over 1 s.
- `make verify` is clean after each change. The examples and notebooks regression suites
  (`tests/regression/`) pass after Change 1: 132 files across `nlsq/`, `examples/`, `tests/` and `docs/`
  mention `curve_fit_large`.

## Measurement

- Every speed claim comes from an interleaved A/B (`ab_nlsq.py`, absolute `--base`, `--cand` and
  `--out-root`), base against head, on this laptop.
- `ab_nlsq.check_correctness` (`ab_nlsq.py:66-82`) requires equal status and equal cost to `rtol` between the
  arms. Change 1 changes the algorithm, so it needs a new flag, `--cost-gate=no-worse`. The harness inside
  `ab_nlsq` runs one library per invocation, so its own `ok` status cannot flag a base result that missed the
  optimum (a single GLS result is its own best). The arms are therefore compared with each other, per case:
  - Both arms need status `ok` and a finite cost, or the case is a violation.
  - The candidate costs more than the base beyond `COST_RTOL`: a violation (the candidate is worse).
  - The base costs more than the candidate beyond `COST_RTOL`: the base missed the optimum (for example the
    GLS DNRO). The case is reported as `skipped` and excluded from timing and the speed verdict.
  - Otherwise the case is timed.

  The candidate's own accuracy is checked independently by the library-comparison rerun below (multi-library
  best-cost gate). Without the flag, behaviour is unchanged. Changes 2 and 3 use the default (equal-cost)
  gate. Skipped cases leave the timing loop (`ab_nlsq.py:170-198`), and a missing or non-finite time on a
  timed case is a violation.
- After all three changes, rerun the library comparison at 1k–100M (both devices) into a new
  `benchmarks/baselines/2026-11-xx-*.md`. Gate: every NLSQ cell is within cost `rtol 1e-6` of the best (the
  1M two_exp DNRO is gone).

## Order and delivery

Changes 2, 3 and 1, in that order, as separate commits on one branch, `perf/close-jaxopt-gap`, cut from the
current `fix/review-rounds-2026-11-07` HEAD. **[my call]** That branch carries the review fixes these changes
build on, and it is not merged yet. Each change has its own A/B and tests. Change
1 is the breaking one: CHANGELOG entry under "Changed (breaking)", the docstring of `curve_fit_large` (the
"chunked processing / GLS" text and the Notes), and the CLAUDE.md "Recent Changes" entry. No commit or push
without the requester's go-ahead.

## Left to the builder

- Internal names and helper placement for the eligibility check and the TSQR pass, as long as it stays O(p²)
  state and reuses `pcov_from_jacobian`.
- The exact log wording.
- How the TSQR chunks are batched on the device.
- Which non-TRF `curve_fit` paths honour versus reject `compute_covariance=False`.

## Decisions (three-model discussion, 2026-11-10)

Record: `three-brain-out/2026-11-10-decide-d1-d2/`. A GPU measurement informed D2: one TSQR pass costs
about 7.7 ms per 1M rows, about 17x a Gram pass. That is about 0.77 s at 100M, which would turn the 100M
exp_decay lead over JAXopt (0.60 vs 0.74 s) into a loss.

- **D1, decided (c), 2 to 1 (Claude and Agy accept; Codex dissents).**
  - Exact gains scalar and signed `sigma` (all three agree).
  - Engine-feature requests (`method`, multistart, TRF options) use chunked with a `UserWarning`.
  - Data and model incompatibilities (multi-feature `xdata`, a non-pointwise model, non-finite `sigma`,
    `xdata` or `ydata` under `check_finite=False`) raise an actionable error naming `strategy=Chunked()`.

  Codex's dissent: a feature request does not authorize an approximate result, so those should raise too.
  To adopt it, change engine-kind reasons to raise; nothing else changes.
- **D2, decided (a), qualified, unanimous.** Normal equations at `kappa <= TSQR_KAPPA`, TSQR above. The
  error and threshold claims are worded as estimates, and the threshold is set by Gate G-cov. There is no
  new `covariance=` keyword (all agree to defer it).

## Calls made by the author (please check)

1. (D2) The default exact path computes the covariance by TSQR when `kappa > TSQR_KAPPA` (1e3, or lower if
   Gate G-cov requires it); otherwise it uses normal equations.
2. A stalled exact fit under the default raises `ExactSolverError`; it does not fall back to GLS.
3. (D1) Multi-feature `xdata`, a non-pointwise model, and non-finite `sigma`, `xdata` or `ydata` under
   `check_finite=False` raise an actionable error. Scalar and signed `sigma` run on exact.
4. (D1) An explicit `method=` (including `"trf"`), multistart, or any TRF-only option uses chunked with a
   `UserWarning`, whether it comes from the legacy keywords or a user-built `Auto(chunked=...)`.
5. `auto_size_detection=False` means exact if eligible.
6. `compute_covariance=False` returns `pcov=None`, and the dependent result members raise.
7. The switch is an `Auto.exact` field, and `Auto(exact=None)` is the opt-out.
8. Under the default, `max_nfev` goes to exact, where it counts full-data passes and raises when exhausted;
   on chunked it counted per-chunk TRF evaluations. This is noted in the CHANGELOG breaking entry. It is not
   a reason to fall back to chunked.
