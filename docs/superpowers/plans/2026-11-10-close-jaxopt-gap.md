# Closing the JAXopt gap: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Make NLSQ faster where it trails JAXopt, without losing exactness, through three changes:
- QR-then-SVD covariance;
- a `compute_covariance=False` opt-out;
- the exact chunk-accumulated LM as the default `curve_fit_large` path at or above `size_threshold`.

**Architecture:**
- **Change 2** swaps the body of one JIT kernel.
- **Change 3** threads one keyword to the final covariance site on each path.
- **Change 1** adds an `exact` sub-strategy to `Auto`. `fit_with_strategy` checks data eligibility after
  validation and falls back to `Auto.chunked` with a `UserWarning`. The exact solver switches its covariance
  to a TSQR pass when `sqrt(cond(H)) > 1e3`.

**Tech Stack:** Python 3.12+, JAX (float64), NumPy, pytest with xdist, `uv`.

**Spec:** `docs/superpowers/specs/2026-11-10-close-jaxopt-gap-design.md`. Read its "Dispatch, step by step"
and "Covariance on the exact default" sections before Tasks 5–7.

## Global Constraints

**Correctness**
- Math is theoretically exact. No Gram/`eigh` replaces an SVD-based covariance (`curve_fit`, per-chunk).
  No random SVD. No silent downsampling.
- An engine-reason fallback from exact to chunked emits a `UserWarning` and logs the same text at INFO. A
  data-reason incompatibility raises an actionable `ValueError` (spec, D1). A skipped final
  covariance logs at DEBUG.

**Defaults and options**
- `compute_covariance` defaults to `True`. When `False`, the final `pcov` is `None`.
- `size_threshold` stays `1_000_000`. Routing uses `n_points >= size_threshold`. Nothing below the threshold
  changes.
- TSQR trigger: `kappa = sqrt(lambda_max/lambda_min)` of `H` `> 1e3` (`TSQR_KAPPA`). `COND_WARN = 1e4` and
  the `pcov_from_normal` warning text are unchanged.
- `stability`, `rescale_data` and `max_jacobian_elements_for_svd` are ignored on the large path. They never
  raise there and never cause a fallback.

**Tests and benchmarks**
- Test markers: `@pytest.mark.serial` for arrays over 100K elements or subprocesses; `@pytest.mark.slow` for
  tests over 1 s.
- A/B: `uv run python benchmarks/studies/ab_nlsq.py`, with absolute `--base`, `--cand` and `--out-root`. The
  base worktree goes under the session scratchpad's absolute path. GPU runs happen outside the sandbox.

**Git**
- Branch `perf/close-jaxopt-gap`, from `fix/review-rounds-2026-11-07` HEAD.
- No commit or push without the requester's go-ahead. Commit steps are prepared commands.
- Commit messages carry no AI attribution.

## Review Focus

1. **NaN in `ydata` under the default.** Expect `ValueError` matching "finite", not `ExactSolverError`.
   Test: Task 6.
2. **`p0=None` under the default.** Expect `p0` inferred from the signature and a successful fit. Test:
   Task 6.
3. **The docstring's `good_model`** (`xdata.astype(jnp.int32)`). Expect it to pass the probe and run exact.
   Test: Task 6.
4. **`sigma` with `absolute_sigma=True` under the default.** Expect `pcov` to match `curve_fit` to
   scale-aware 1e-6. Test: Task 6.
5. **`compute_covariance=False` with `full_output=True`.** Expect the 5-tuple with `pcov=None` and
   `infodict` intact. Test: Task 3.

---

### Task 1: QR-then-SVD covariance kernel (Change 2)

**Files:**
- Modify: `nlsq/core/minpack.py:3578-3582` (`_covariance_svd_jit`, uses `jax_svd` = `jax.scipy.linalg.svd`,
  imported at :75)
- Test: `tests/core/test_covariance.py`. Check `tests/core/test_shared_jit_kernels.py:46` and
  `tests/core/test_device_policy.py`.

**Interfaces:**
- Produces: `_covariance_svd_jit(jac) -> (s, VT)`, the same contract as before. Rows of `VT` may differ in
  sign.

- [ ] **Step 1: Write the tests** in `tests/core/test_covariance.py` (imports: `numpy as np`,
  `jax.numpy as jnp`, `pytest`, `contextlib`, `nlsq.core.minpack as minpack`,
  `from nlsq.core.minpack import _covariance_svd`,
  `from nlsq.core.covariance import pcov_from_jacobian`,
  `from nlsq.result import OptimizeWarning`):

```python
def _svd_ref(J):
    _, s, VT = np.linalg.svd(np.asarray(J), full_matrices=False)
    return s, VT


def _rel(a, b):
    d = np.sqrt(np.outer(np.diag(b), np.diag(b)))
    return np.max(np.abs(a - b) / np.where(d > 0, d, 1.0))


@pytest.mark.parametrize("shape,rank", [((200, 3), 3), ((200, 5), 3), ((50, 3), 2)])
def test_qr_covariance_matches_svd(shape, rank):
    rng = np.random.default_rng(0)
    J = jnp.asarray(
        rng.normal(size=(shape[0], rank)) @ rng.normal(size=(rank, shape[1]))
    )
    deficient = rank < shape[1]
    ctx = (
        pytest.warns(OptimizeWarning, match="rank-deficient")
        if deficient
        else contextlib.nullcontext()
    )
    with ctx:
        got, _ = pcov_from_jacobian(J, 0.7, shape[0], shape[1], False, _covariance_svd)
    with ctx:
        ref, _ = pcov_from_jacobian(J, 0.7, shape[0], shape[1], False, _svd_ref)
    assert _rel(got, ref) <= 1e-12


def test_qr_covariance_wide_absolute_sigma():
    rng = np.random.default_rng(1)
    J = jnp.asarray(rng.normal(size=(2, 4)))
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        got, _ = pcov_from_jacobian(J, 0.7, 2, 4, True, _covariance_svd)
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        ref, _ = pcov_from_jacobian(J, 0.7, 2, 4, True, _svd_ref)
    assert _rel(got, ref) <= 1e-12


def test_qr_covariance_wide_relative_sigma_is_inf():
    J = jnp.asarray(np.random.default_rng(2).normal(size=(2, 4)))
    with pytest.warns(OptimizeWarning):
        got, fired = pcov_from_jacobian(J, 0.7, 2, 4, False, _covariance_svd)
    assert fired and np.all(np.isinf(got))


def test_qr_covariance_kernel_uses_r_factor(monkeypatch):
    calls, real = [], jnp.linalg.qr
    monkeypatch.setattr(
        jnp.linalg,
        "qr",
        lambda a, mode="reduced": calls.append(mode) or real(a, mode=mode),
    )
    minpack._covariance_svd_body(jnp.ones((10, 2)) + jnp.eye(10, 2))
    assert calls == ["r"]


def test_nan_jacobian_still_gives_inf_pcov():
    J = jnp.asarray(np.array([[1.0, 0.0], [np.nan, 1.0], [0.0, 2.0]]))
    with pytest.warns(OptimizeWarning):
        pcov, fired = pcov_from_jacobian(J, 0.5, 3, 2, False, _covariance_svd)
    assert fired and np.all(np.isinf(pcov))
```

- [ ] **Step 2: Run.** Command:
  `uv run pytest tests/core/test_covariance.py -k "qr_covariance or nan_jacobian" -n 0 -v`.
  Expected: `test_qr_covariance_kernel_uses_r_factor` FAILS (`_covariance_svd_body` is undefined). The others
  pass; they pin behaviour.

- [ ] **Step 3: Implement.**
  - Add the module-level `_covariance_svd_body(jac) -> tuple[jnp.ndarray, jnp.ndarray]`:
    `R = jnp.linalg.qr(jac, mode="r")`, then `_, s, VT = jax_svd(R, full_matrices=False)`, then return
    `s, VT`.
  - Define `_covariance_svd_jit = jit(_covariance_svd_body)`.
  - Add one line to `_covariance_svd`'s docstring: "SVD of the QR R factor; same s and VT up to row signs."

- [ ] **Step 4: Run.** Command:
  `uv run pytest tests/core/test_covariance.py tests/core/test_device_policy.py tests/core/test_shared_jit_kernels.py -n 0 -q`.
  Expected: all pass. A test comparing `VT` elementwise is rewritten to compare `s` and `pcov`. No tolerance
  is loosened.

- [ ] **Step 5: Run the fast suite.** `make test-fast`. Expected: 0 failures.

### Task 2: Change 2 gate, then record or drop

**Files:**
- Create: `benchmarks/studies/bench_cov_kernel.py`
- Create: `benchmarks/baselines/2026-11-10-qr-covariance-ab.md`
- Modify (only if the crossover moves more than 2x): `nlsq/core/device_policy.py:17`

**Interfaces:**
- Consumes: `minpack._covariance_svd_body` (Task 1).

- [ ] **Step 1: Kernel microbenchmark (go/no-go).** In `bench_cov_kernel.py`:
  - Jit `old(J) = jax_svd(J, full_matrices=False)[1:]` and the new body.
  - Warm both up.
  - Run 20 interleaved rounds of each, each followed by `jax.block_until_ready`.
  - Print the medians and IQR for `(1_000_000, 3)` and `(1_000_000, 6)` float64 on GPU.

  Gate: the new median is below old minus old IQR on both shapes. If it fails, go to Step 6 and drop the
  change.

- [ ] **Step 2: Create the base checkout.** Run
  `git worktree add <scratchpad-abs>/ab-base HEAD`, with Task 1 uncommitted in the main tree. The candidate is
  `/home/wei/Documents/GitHub/NLSQ`. Run `uv sync` in the base checkout if `import nlsq` fails there.

- [ ] **Step 3: End-to-end no-regression check, GPU (outside the sandbox).** Run:

  ```
  uv run python benchmarks/studies/ab_nlsq.py --base <scratchpad-abs>/ab-base \
    --cand /home/wei/Documents/GitHub/NLSQ --platform gpu --libraries nlsq --sizes 1000000 \
    --problems exp_decay gaussian two_exp bounded_exp \
    --out-root /home/wei/Documents/GitHub/NLSQ/benchmarks/results/2026-11-10-ab-qr-cov
  ```

  Then repeat with `--libraries nlsq_large --sizes 10000000 --repeats 3`.

  Gate: no ratio is above the spread.

- [ ] **Step 3b: CPU.** Use the same command with
  `--platform cpu --sizes 100000 1000000 --libraries nlsq`. Gate: no ratio above the spread.

- [ ] **Step 4: Host-SVD crossover.** In `bench_cov_kernel.py --crossover`:
  - Time `_covariance_svd_jit` on a CPU-device copy (including the transfer) against the GPU array, for
    `n x 3` and `n x 6` at `n` in {2k, 5k, 10k, 20k, 50k}.
  - Find the largest `n*p` where the host path wins in every sweep.
  - If it differs from 30,000 by more than 2x, set `HOST_SVD_MAX_ELEMENTS` to it and cite the baseline file in
    the comment. Otherwise leave it.

- [ ] **Step 5: Accuracy gate.** Compare old and new `pcov` (`pcov_from_jacobian` with each body):
  - the four benchmark problems' Jacobians at the true parameters, 1M points: `<= 1e-12` scale-aware;
  - two_exp with rates 1.0 and 1.0001 (`cond(J)` around 1e6): `<= 1e-9`.

  Record the actual figures.

- [ ] **Step 6: Record or drop.**
  - Write the baseline file: kernel medians and IQR, A/B ratios with spread, crossover, accuracy figures,
    and machine state (`free -g`, `cpupower frequency-info -p` or the governor file, `nvidia-smi -q -d CLOCK`).
  - If any gate failed, revert Task 1 and mark the change "dropped (measured)".

- [ ] **Step 7: Commit (on approval).** Command:
  `git add nlsq/core/minpack.py nlsq/core/device_policy.py tests/core/ benchmarks/studies/bench_cov_kernel.py benchmarks/baselines/2026-11-10-qr-covariance-ab.md && git commit -m "perf(covariance): factor J by QR before the covariance SVD"`

### Task 3: `compute_covariance` on `curve_fit` (Change 3, in-memory)

**Files:**
- Modify: `nlsq/core/minpack.py`:
  - `curve_fit` (2999) signature and docstring;
  - `CurveFit.curve_fit` (4700);
  - the covariance site (5056);
  - the `method="cmaes"`, `method="multi-start"`, `multistart=True`/`global_search=True`,
    `method="hybrid_streaming"`, `fallback=True` and `compute_diagnostics=True` branches.
- Modify: `nlsq/result/curve_fit_result.py:480` (message), `:588` (new guard in `prediction_interval`) and
  `:682` (message in `confidence_band`).
- Test: create `tests/core/test_compute_covariance.py`.

**Interfaces:**
- Produces:
  - `curve_fit(..., compute_covariance: bool = True)` and
    `CurveFit.curve_fit(..., compute_covariance: bool = True)`;
  - when `False`, `pcov` is `None` in every return shape;
  - `CurveFitResult.confidence_intervals`, `confidence_band` and `prediction_interval` raise
    `AttributeError` containing `"compute_covariance=False"` when `pcov is None`;
  - `summary()` omits the error column (it already does).

- [ ] **Step 1: Write the tests.** Module setup:
  `x = np.linspace(0, 4, 200)`, `model = lambda x, a, b: a * jnp.exp(-b * x)`,
  `y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.default_rng(0).standard_normal(200)`, `p0 = [1.0, 1.0]`.

```python
def test_default_unchanged():
    popt, pcov = curve_fit(model, x, y, p0=p0)
    assert pcov.shape == (2, 2)


def test_skip_returns_none_and_no_svd(monkeypatch):
    ref_popt = curve_fit(model, x, y, p0=p0)[0]  # before patching
    calls = []
    real = minpack._covariance_svd_jit
    monkeypatch.setattr(
        minpack, "_covariance_svd_jit", lambda J: calls.append(1) or real(J)
    )
    res = curve_fit(model, x, y, p0=p0, compute_covariance=False)
    popt, pcov = res
    assert pcov is None and res.pcov is None and calls == []
    np.testing.assert_allclose(popt, ref_popt, rtol=1e-12)


def test_skip_full_output_5tuple():
    popt, pcov, info, mesg, ier = curve_fit(
        model, x, y, p0=p0, full_output=True, compute_covariance=False
    )
    assert pcov is None and "fvec" in info and ier in (1, 2, 3, 4)


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.confidence_intervals(),
        lambda r: r.confidence_band(),
        lambda r: r.prediction_interval(),
    ],
)
def test_dependent_members_raise(call):
    res = curve_fit(model, x, y, p0=p0, compute_covariance=False)
    with pytest.raises(AttributeError, match="compute_covariance=False"):
        call(res)


def test_summary_without_pcov():
    curve_fit(model, x, y, p0=p0, compute_covariance=False).summary()  # no exception


def test_not_forwarded_to_least_squares(monkeypatch):
    seen = {}
    orig = LeastSquares.least_squares
    monkeypatch.setattr(
        LeastSquares,
        "least_squares",
        lambda self, *a, **k: seen.update(k) or orig(self, *a, **k),
    )
    curve_fit(model, x, y, p0=p0, compute_covariance=False)
    assert "compute_covariance" not in seen
```

  Add one test per branch: `method="cmaes"`, `method="multi-start"`, `multistart=True`, `global_search=True`,
  `method="hybrid_streaming"`, `fallback=True` and `compute_diagnostics=True`. Each asserts `pcov is None`, or
  `pytest.raises(ValueError, match="compute_covariance")`, matching Step 3's per-branch choice. Then run
  `grep -rn "pcov" nlsq/core/ nlsq/utils/diagnostics.py nlsq/result/`. Every reader reachable from
  `curve_fit` (for example `orchestration/covariance_computer.py`) gets a test with `pcov is None`.

- [ ] **Step 2: Run.** `uv run pytest tests/core/test_compute_covariance.py -n 0 -q`. Expected: FAIL
  (unexpected keyword).

- [ ] **Step 3: Implement.**
  - Pop `compute_covariance` before kwargs reach `least_squares`.
  - At `minpack.py:5056`: when `False`, `pcov, warn_cov = None, False` and
    `self.logger.debug("covariance skipped", reason="compute_covariance=False")`.
  - Each branch above either reaches this site (honoured) or raises `ValueError` naming the branch and
    `compute_covariance`. Document the per-branch result in the parameter's docstring.
  - Result guards as listed in Files.

- [ ] **Step 4: Run.** The new file, then `make test-fast`. Expected: all pass.

### Task 4: `compute_covariance` on `curve_fit_large` (Change 3, large paths)

**Files:**
- Modify: `nlsq/__init__.py`. `curve_fit_large` gains `compute_covariance: bool = True` as a top-level
  parameter, not in `_LEGACY_PATH_OPTIONS`, and passes it to `fit_with_strategy`. Update the docstring.
- Modify: `nlsq/streaming/large_fit.py`. `fit_with_strategy(..., compute_covariance: bool = True)` threads
  it into `_fit_in_memory` (as a `curve_fit` kwarg), `_fit_chunked` (final `pcov` replaced by `None` after
  `unpack`; never forwarded to chunk fits), `_fit_exact` (passed to `large_strategy.fit(...,
  compute_covariance=...)`), and `_fit_hybrid` (`False` raises
  `ValueError("compute_covariance=False is not supported by hybrid_streaming")`).
- Modify: `nlsq/streaming/specs.py:49,54`. `_IN_MEMORY_RESERVED` and `_CHUNKED_RESERVED` gain
  `"compute_covariance"`.
- Modify: `nlsq/streaming/exact_lm.py:511`. `exact_lm_fit(..., compute_covariance: bool = True)`. At :717 the
  fit calls `pcov_from_normal(H, cost, n, p, absolute_sigma)[0]` directly (the parameter shadows the module
  function `compute_covariance`, which stays for `tests/streaming/test_exact_lm.py:653-680`). When `False`,
  `pcov=None`.
- Modify: `nlsq/streaming/strategy.py:120,150`. `unpack` return annotations become
  `tuple[np.ndarray, np.ndarray | None]`.
- Test: create `tests/streaming/test_compute_covariance_large.py`.

**Interfaces:**
- Consumes: Task 3's `curve_fit(compute_covariance=...)`.
- Produces:
  - `curve_fit_large(..., compute_covariance=True)` returns `(popt, None)` on the chunked and exact paths
    when `False`.
  - `exact_lm_fit(..., compute_covariance=True)`. Task 5 extends its covariance site.

- [ ] **Step 1: Write the tests.** Use `n = 5_000` points of exp_decay.
  - `strategy=InMemory()` gives `pcov is None`.
  - `strategy=Chunked(chunk_size=1000)` gives `pcov is None`, and `popt` equals the flag-on `popt` exactly.
    Patch `LargeDatasetFitter._chunk_precision` with a recording wrapper that calls the original, and assert
    every recorded `pcov_chunk` is not `None`.
  - `strategy=Exact(chunk_size=1000)` gives `pcov is None`.
  - `Chunked(options={"compute_covariance": False})` raises `TypeError`.
  - `InMemory(options={"compute_covariance": False})` raises `TypeError`.
  - `method="hybrid_streaming", compute_covariance=False` raises `ValueError`.

- [ ] **Step 2: Run.** `uv run pytest tests/streaming/test_compute_covariance_large.py -n 0 -q`. Expected:
  FAIL.

- [ ] **Step 3: Implement** as listed in Files.

- [ ] **Step 4: Run.** The new file, then `uv run pytest tests/streaming -m "not slow" -q`, then
  `make test-fast`. Expected: all pass.

- [ ] **Step 5: Commit Tasks 3 and 4 (on approval).** Command:
  `git commit -m "feat(curve_fit): add compute_covariance=False to skip pcov"`

### Task 5: TSQR covariance for ill-conditioned exact fits

**Files:**
- Modify: `nlsq/streaming/normal_equations.py`. Add `make_chunk_jacobian` beside `make_chunk_stats` (:91),
  with the same weighting and masking.
- Modify: `nlsq/streaming/exact_lm.py`:
  - new `TSQR_KAPPA` and `tsqr_r_factor`;
  - the covariance site at :717 (as changed in Task 4);
  - the `covariance_method` result field;
  - its own TSQR-route warning;
  - the module docstring `shortcut:` note (:14-16).
- Test: `tests/streaming/test_exact_lm.py`.
- Modify: `nlsq/core/covariance.py`. Extract `normal_condition(H) -> float` from `pcov_from_normal`
  (:152-160, the power-of-two scaling and `errstate`); `pcov_from_normal` calls it. Its warning text and
  behaviour are unchanged, because `adaptive_hybrid.py:3642` relies on them.

**Interfaces:**
- Consumes:
  - the pass machinery of `run_pass` (`normal_equations.py:339`): `_block`, `FactorySource` (:219) and
    padded-tail masks;
  - `pcov_from_jacobian` (`nlsq/core/covariance.py:33`);
  - `_covariance_svd` (`nlsq/core/minpack.py:3584`).
- Produces:
  - `make_chunk_jacobian(model, weighted=False, *, jacobian=None) -> jitted (theta, x, mask) -> J` (weighted
    variant: `(theta, x, sigma, mask)`). Rows are divided by `sigma`; masked rows are set to 0 with
    `jnp.where`.
  - `TSQR_KAPPA = 1e3`.
  - `normal_condition(H: np.ndarray) -> float` in `nlsq/core/covariance.py`: `sqrt(lambda_max/lambda_min)`
    via the scaled `eigvalsh`; `inf` when `H` is non-finite or `lambda_min <= 0`.
  - `_open_source(x, y, sigma, chunk, p, memory_limit_gb, n_points) -> (source, weighted: bool)` in
    `exact_lm.py`, factored out of `exact_lm_fit`'s existing source setup (`exact_lm.py:592-598,630-634`):
    `FactorySource(x, chunk, n_points)` for factories, `_place` for arrays. `n_points` is the validated
    declared count (`None` when undeclared), and the factory verifies it on the first pass and every replay
    (`normal_equations.py:236-238,316-323`). `exact_lm_fit` uses it; there is no behaviour change.
  - `tsqr_r_factor(model, theta, source, chunk, *, weighted: bool) -> np.ndarray`: `min(n, p) x p` R of the
    full weighted Jacobian at `theta`, from one pass. State `O(p^2)`; workspace `O(chunk * p)`. Factory
    sources verify their count as in `run_pass`.
  - `OptimizeResult.covariance_method` in `{"normal", "tsqr", "skipped"}`.

- [ ] **Step 1: Write the tests.** Module setup (runnable as written; `jax_enable_x64` comes from the
  existing conftest; imports `types`, `jax`, `jax.numpy as jnp`, `numpy as np`, `pytest`, `nlsq.streaming.exact_lm as exact_lm`,
  `from nlsq.streaming.exact_lm import exact_lm_fit`, `from nlsq.core.covariance import pcov_from_jacobian,
  pcov_from_normal, normal_condition`, `from nlsq.core.minpack import _covariance_svd`,
  `from nlsq.result import OptimizeWarning`):

```python
two_exp = lambda x, a1, k1, a2, k2: a1 * jnp.exp(-k1 * x) + a2 * jnp.exp(-k2 * x)
exp_decay = lambda x, a, b, c: a * jnp.exp(-b * x) + c
x = np.linspace(0.0, 5.0, 3001)
noise = 1e-3 * np.random.default_rng(0).standard_normal(x.size)
y_exp = np.asarray(exp_decay(x, 2.5, 1.3, 0.5)) + noise
TRUE_ILL = (2.0, 1.0, 1.5, 1.0001)  # cond(J) ~ 1e5-1e6
y_ill = np.asarray(two_exp(x, *TRUE_ILL)) + noise
P0_ILL = [2.0, 0.9, 1.5, 1.1]
TIGHT = dict(ftol=1e-12, xtol=1e-12, gtol=1e-12)


def full_jac(model, x, theta, sigma=None):
    J = np.asarray(
        jax.jacfwd(lambda q: model(jnp.asarray(x), *q))(jnp.asarray(theta, float))
    )
    return J if sigma is None else J / np.asarray(sigma)[:, None]


def factory(xs, ys):
    return lambda cs: ((xs[i : i + cs], ys[i : i + cs]) for i in range(0, xs.size, cs))


@pytest.fixture
def spy_tsqr(monkeypatch):
    real = exact_lm.tsqr_r_factor
    spy = types.SimpleNamespace(calls=0)

    def wrapper(*a, **k):
        spy.calls += 1
        return real(*a, **k)

    monkeypatch.setattr(exact_lm, "tsqr_r_factor", wrapper)
    return spy


def _sv(A):
    return np.linalg.svd(np.asarray(A), compute_uv=False)


@pytest.mark.parametrize("weighted", [False, True])
def test_tsqr_r_factor_matches_full_jacobian(weighted):
    sigma = 0.5 + np.random.default_rng(1).random(x.size) if weighted else None
    theta = np.array(TRUE_ILL)
    src, w = exact_lm._open_source(
        x, y_ill, sigma, 700, 4, None, None
    )  # 3001 = 4*700 + 201: padded tail
    R = exact_lm.tsqr_r_factor(two_exp, theta, src, 700, weighted=w)
    np.testing.assert_allclose(
        _sv(R), _sv(full_jac(two_exp, x, theta, sigma)), rtol=1e-12
    )


def test_tsqr_r_factor_factory_replay():
    theta = np.array(TRUE_ILL)
    src_a, _ = exact_lm._open_source(x, y_ill, None, 700, 4, None, None)
    src_f, _ = exact_lm._open_source(
        factory(x, y_ill), None, None, 700, 4, None, x.size
    )
    R_a = exact_lm.tsqr_r_factor(two_exp, theta, src_a, 700, weighted=False)
    R_f = exact_lm.tsqr_r_factor(two_exp, theta, src_f, 700, weighted=False)
    np.testing.assert_allclose(_sv(R_f), _sv(R_a), rtol=1e-13)


def test_tsqr_pass_detects_changed_replay_count():
    state = {"short": False}  # flips the factory to n-1 points for later passes

    def fac(cs):
        m = x.size - 1 if state["short"] else x.size
        return (
            (x[i : min(i + cs, m)], y_ill[i : min(i + cs, m)]) for i in range(0, m, cs)
        )

    src, _ = exact_lm._open_source(fac, None, None, 700, 4, None, x.size)
    exact_lm.tsqr_r_factor(
        two_exp, np.array(TRUE_ILL), src, 700, weighted=False
    )  # full count: ok
    state["short"] = True
    with pytest.raises(ValueError, match="points"):
        exact_lm.tsqr_r_factor(two_exp, np.array(TRUE_ILL), src, 700, weighted=False)


def test_declared_count_mismatch_raises():
    with pytest.raises(ValueError, match="n_points"):
        exact_lm_fit(
            two_exp,
            factory(x, y_ill),
            None,
            P0_ILL,
            chunk_size=700,
            n_points=x.size + 1,
        )


def test_tsqr_r_factor_rank_deficient():
    dup = lambda x, a, b, k: a * jnp.exp(-k * x) + b * jnp.exp(
        -k * x
    )  # a, b not separable
    theta = np.array([1.0, 1.0, 1.3])
    src, _ = exact_lm._open_source(x, y_exp, None, 700, 3, None, None)
    R = exact_lm.tsqr_r_factor(dup, theta, src, 700, weighted=False)
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        got, _ = pcov_from_jacobian(R, 0.5, x.size, 3, False, _covariance_svd)
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        ref, _ = pcov_from_jacobian(
            full_jac(dup, x, theta), 0.5, x.size, 3, False, _covariance_svd
        )
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-12 * np.abs(ref).max())


def test_ill_conditioned_exact_uses_tsqr(spy_tsqr):
    res = exact_lm_fit(two_exp, x, y_ill, P0_ILL, chunk_size=1000, **TIGHT)
    assert res.success
    ref, _ = pcov_from_jacobian(
        full_jac(two_exp, x, res.x), res.cost, x.size, 4, False, _covariance_svd
    )
    d = np.sqrt(np.outer(np.diag(ref), np.diag(ref)))
    assert res.covariance_method == "tsqr" and spy_tsqr.calls == 1
    assert np.max(np.abs(res.pcov - ref) / d) <= 1e-8


def test_well_conditioned_exact_skips_tsqr(spy_tsqr):
    res = exact_lm_fit(exp_decay, x, y_exp, [1.0, 1.0, 0.0], chunk_size=1000)
    assert res.covariance_method == "normal" and spy_tsqr.calls == 0


def test_skip_never_calls_tsqr(spy_tsqr):
    res = exact_lm_fit(
        two_exp, x, y_ill, P0_ILL, chunk_size=1000, compute_covariance=False, **TIGHT
    )
    assert (
        res.covariance_method == "skipped" and res.pcov is None and spy_tsqr.calls == 0
    )


def test_failed_fit_still_reports_route(spy_tsqr):
    res = exact_lm_fit(two_exp, x, y_ill, P0_ILL, chunk_size=1000, max_nfev=2)
    assert not res.success and res.covariance_method in ("normal", "tsqr")


def test_tsqr_route_warns_when_kappa_above_cond_warn():
    y2 = np.asarray(two_exp(x, 2.0, 1.0, 1.5, 1.00001)) + noise
    with pytest.warns(UserWarning, match="computed by QR"):
        exact_lm_fit(two_exp, x, y2, P0_ILL, chunk_size=1000, **TIGHT)


def test_pcov_from_normal_warning_text_unchanged():
    with pytest.warns(UserWarning, match="the covariance may be inaccurate"):
        pcov_from_normal(np.diag([1.0, 1e-10]), 0.5, 10, 2, False)


@pytest.mark.parametrize(
    "H,expected",
    [
        (np.diag([1e300, 1e300]), 1.0),  # extreme but well-conditioned: no overflow
        (np.diag([4.0, 1e-6]), 2000.0),
        (np.diag([1.0, 0.0]), np.inf),
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), np.inf),
    ],
)
def test_normal_condition(H, expected):
    assert normal_condition(H) == pytest.approx(expected, rel=1e-12)
```

  If `TRUE_ILL` gives `kappa <= 1e3` at the fitted `popt` on this grid, move the second rate closer (1.00001)
  until `test_ill_conditioned_exact_uses_tsqr` exercises the TSQR route, and record the value used.

- [ ] **Step 2: Run.** `uv run pytest tests/streaming/test_exact_lm.py -k "tsqr or pcov_from_normal_warning or normal_condition or failed_fit" -n 0 -q`.
  Expected: FAIL.

- [ ] **Step 3: Implement.**
  - `make_chunk_jacobian` and `tsqr_r_factor`. Per chunk: `R = jnp.linalg.qr(jnp.vstack([R, Jc]),
    mode="r")`, float64 on the chunk's device. Start from an empty `(0, p)` array. Return NumPy.
  - At the covariance site:
    - `compute_covariance=False` gives `"skipped"`.
    - Otherwise compute `kappa = normal_condition(H)`.
    - If `kappa > TSQR_KAPPA`:
      - `R = tsqr_r_factor(...)`;
      - `pcov = pcov_from_jacobian(R, cost, n, p, absolute_sigma, _covariance_svd)[0]`;
      - `covariance_method = "tsqr"`;
      - if `kappa > COND_WARN`, `warnings.warn(UserWarning("... sqrt(cond(H)) > 1e4; covariance computed by
        QR; the parameters may stop short of the optimum: check the result"), skip_file_prefixes=_NLSQ_DIR)`.
    - Otherwise use `pcov_from_normal(...)[0]` and `"normal"`.
  - Import `_covariance_svd` lazily inside the function, so `exact_lm` does not import `minpack` at module
    load.
  - Replace the `shortcut:` note: "covariance: TSQR when kappa > 1e3; step solve stays on normal equations
    (conditioning study)."

- [ ] **Step 4: Run.** `uv run pytest tests/streaming/test_exact_lm.py tests/streaming/test_exact_lm_large.py -n 0 -q`.
  Expected: all pass. A test that asserted the `pcov_from_normal` warning on an exact fit with
  `kappa > 1e4` now expects the TSQR-route warning. Update only its `match=`.

- [ ] **Step 4a: Gate G-cov (spec, decided as D2).** Add
  `tests/streaming/test_exact_cov_accuracy.py`, marked `slow`. It is parametrized over:
  - controlled `kappa(J)` in {1, 10, 100, 500, 900, 1100, 2000}, from the linear pointwise model
    `f(x, *theta) = B[x.astype(int)] @ theta` with `B = U @ diag(s) @ V.T`. `U` is the orthonormal Q of a
    seeded `(n, p)` Gaussian, `V` is a seeded orthogonal `(p, p)`, and `s = geomspace(kappa, 1, p)`. Verify
    `np.linalg.svd(B)` reproduces `s` to 1e-12 before using it;
  - realism cases: two_exp and gaussian at their measured `kappa`;
  - correlated parameters;
  - column scales spanning 1e-6..1e6;
  - `sigma` with `absolute_sigma` both ways;
  - an active bound;
  - one rank-deficient case.

  For each case, compare `exact_lm_fit(...).pcov` against
  `pcov_from_jacobian(full_jac(popt), ..., _covariance_svd)`. Assert scale-aware `<= 1e-8` and an equal rank
  decision.
  - If any case fails, set `TSQR_KAPPA` to the largest tested value strictly below the smallest failing
    `kappa`, then rerun the whole gate.
  - If `kappa = 1` fails, make TSQR unconditional and stop to report to the requester.
  - Record the value and the evidence in the baseline file.

- [ ] **Step 4b: Measure on GPU.** Write a one-off script, `benchmarks/studies/exact_cov_route.py`:
  - For the four benchmark problems at 10M (`benchmarks/compare/problems.py` data, `seed=0`), call
    `exact_lm_fit` directly and print `covariance_method` and `kappa`.
  - Time `tsqr_r_factor` alone, synchronized, at 10M and 100M on exp_decay.
  - Append the results to `benchmarks/baselines/2026-11-10-qr-covariance-ab.md`.

  Expected: all four print `"normal"`. If two_exp prints `"tsqr"`, record it and its pass cost; do not change
  `TSQR_KAPPA`.

- [ ] **Step 5: Commit (on approval).** Command:
  `git commit -m "fix(exact_lm): compute ill-conditioned covariance by TSQR"`

### Task 6: `Auto.exact`, eligibility and dispatch (Change 1, core)

**Files:**
- Modify: `nlsq/streaming/specs.py:265-290` (`Auto`)
- Modify: `nlsq/streaming/large_fit.py:47-153` (`fit_with_strategy`); add `_exact_ineligible`
- Modify (D1 sigma support; explicit route too):
  - `nlsq/streaming/exact_lm.py:426-455` (`_check_inputs`) and `:647-654`;
  - `nlsq/streaming/normal_equations.py`:
    - `_block` (:145), so a scalar sigma passes through as a scalar block;
    - the per-block sigma checks (:332-335, :365-375), changed from "positive" to "nonzero";
    - `make_chunk_stats` and `make_chunk_jacobian`, which use `jnp.abs(s)`, broadcasting a scalar `s`;
    - the factory-block path (`FactorySource`, :219), the same rules for yielded sigma;
  - `nlsq/streaming/strategy.py:254` (`check_exact_sources`), accepting a scalar.

  Memory stays bounded: never build an `n`-length sigma. Zero still raises.
- Test: create `tests/streaming/test_default_exact.py`

**Interfaces:**
- Consumes:
  - `_fit_exact` (`large_fit.py:156`), `_fit_chunked` (`:314`) and `_fit_in_memory`;
  - `normalize_p0_bounds` (`nlsq/streaming/strategy.py:244`);
  - `exact_lm.check_model_contract(model, theta, x_chunk)`, which raises `ValueError("model is not
    pointwise...")`;
  - `exact_lm.ExactSolverError`.
- Produces:
  - `Auto.exact: Exact | None = field(default_factory=Exact)`.
  - `Auto.resolve(n_points: int) -> InMemory | Chunked | Exact`.
  - `_EXACT_OPTION_KEYS = frozenset({"ftol", "xtol", "gtol", "max_nfev"})`.
  - `_effective_exact(auto: Auto, opts: Mapping[str, Any]) -> Exact`. Per field: a non-default value in
    `auto.exact` wins; otherwise `chunk_size` comes from `auto.chunked.chunk_size`, and `ftol`, `xtol`, `gtol`
    and `max_nfev` come from `opts`. Use `dataclasses.replace`.
  - `_exact_ineligible(auto: Auto, opts: Mapping[str, Any], f, xdata: np.ndarray, n_points: int, sigma, p0,
    bounds, check_finite: bool) -> tuple[Literal["engine", "data"], str] | None`. `xdata` is the full
    validated array. The probe slice `xdata[: min(n_points, 4096)]` is taken inside, after the `ndim` check.
    It returns one of:
    - `("engine", "Auto.chunked sets method")`;
    - `("engine", "Auto.chunked enables multistart")`;
    - `("engine", "Auto.chunked option(s) <keys> are not supported by exact")`;
    - `("data", "multi-feature xdata")`;
    - `("data", "sigma has non-finite values (check_finite=False)")`;
    - `("data", "model is not pointwise: <message>")`;
    - `None`.

    Scalar, signed and 2-D `sigma` are not reasons: the first two run on exact, and 2-D raises
    `_check_inputs`' existing error. For 2-D `sigma`, check the chunked error text and make both errors name
    `sigma` shape.
  - `_fit_exact(..., route: Literal["explicit", "default"] = "explicit")`.

- [ ] **Step 1: Write the tests.**
  - Setup: `n = 20_000`, `size_threshold=10_000`, and exp_decay data.
  - Fixture `dispatch`: monkeypatch `large_fit._fit_in_memory`, `_fit_chunked` and `_fit_exact` with wrappers
    that append `"in_memory"`, `"chunked"` or `"exact"` to a list and call the original.

```text
@pytest.mark.parametrize("n,expected", [(9_999, "in_memory"), (10_000, "exact"), (10_001, "exact")])
def test_threshold_routing(dispatch, n, expected):
    curve_fit_large(m, x[:n], y[:n], p0=p0, size_threshold=10_000); assert dispatch == [expected]
def test_auto_size_detection_false_runs_exact(dispatch): ...  # n=5_000 → ["exact"]
def test_auto_exact_none_is_old_behaviour():  # popt, pcov bitwise equal to strategy=Chunked()
@pytest.mark.parametrize("kw,reason", [
    (dict(method="trf", size_threshold=10_000), "sets method"),
    (dict(multistart=True, size_threshold=10_000), "multistart"),
    (dict(loss="soft_l1", size_threshold=10_000), "not supported by exact"),
    (dict(strategy=Auto(size_threshold=10_000, chunked=Chunked(options={"loss": "soft_l1"}))), "not supported by exact"),
])
def test_configured_chunked_falls_back(dispatch, kw, reason):
    with pytest.warns(UserWarning, match=reason):
        curve_fit_large(m, x, y, p0=p0, **kw)
    assert dispatch == ["chunked"]
def test_inf_sigma_unchecked_raises(dispatch):
    # sigma = ones, sigma[5] = np.inf, check_finite=False → ValueError matching "strategy=Chunked()"; dispatch == []
@pytest.mark.parametrize("sigma", [2.0, -np.ones(20_000), -2.0])
def test_scalar_and_signed_sigma_run_exact(dispatch, sigma):
    # dispatch == ["exact"]; popt and pcov match curve_fit(m, x, y, sigma=np.abs(np.broadcast_to(sigma, n)))
    # to scale-aware 1e-6, with absolute_sigma both False and True
def test_explicit_exact_accepts_scalar_sigma():  # method="exact_streaming", sigma=2.0 → succeeds
@pytest.mark.parametrize("sigma", [2.0, "signed"])
def test_sigma_variants_padded_tail_and_factory(sigma):
    # n=3001, chunk_size=700 (padded tail); arrays AND a chunk factory yielding (x, y, sigma) blocks
    # ("signed": alternating ±(0.5+rand)) → popt/pcov equal curve_fit on |sigma| to scale-aware 1e-6;
    # tsqr_r_factor(weighted=True) gives the same singular values as full_jac(..., |sigma|)
def test_zero_sigma_raises(dispatch):  # ValueError matching "sigma", as today
def test_2d_sigma_raises(dispatch):  # ValueError naming sigma shape, as chunked does today
def test_valid_sigma_longer_than_probe_runs_exact(dispatch):  # positive 1-D sigma, n=20_000 > 4096 → ["exact"]
def test_auto_chunked_settings_reach_exact(monkeypatch):
    # strategy=Auto(size_threshold=10_000, chunked=Chunked(chunk_size=500, options={"ftol": 1e-10}))
    # → exact_lm_fit called with chunk_size=500, ftol=1e-10 (spy on exact_lm.exact_lm_fit kwargs)
def test_explicit_auto_exact_fields_win(monkeypatch):
    # Auto(..., exact=Exact(ftol=1e-11), chunked=Chunked(options={"ftol": 1e-10})) → ftol=1e-11
def test_fit_with_strategy_call_kwargs_reach_exact(monkeypatch):
    # fit_with_strategy(Auto(size_threshold=10_000), m, x, y, p0, ..., ftol=1e-10) → ftol=1e-10
    # fit_with_strategy(Auto(size_threshold=10_000), ..., loss="soft_l1") → warns, ["chunked"]
def test_2d_xdata_raises(dispatch):  # xdata (2, 20_000) → ValueError matching "multi-feature" and "strategy=Chunked()"; dispatch == []
def test_2d_xdata_with_explicit_chunked_runs(dispatch):  # same data, strategy=Chunked() → ["chunked"], no warning
def test_non_pointwise_model_raises(dispatch):  # lambda x, a: a * (x - jnp.mean(x)) → ValueError "not pointwise" + "strategy=Chunked()"
def test_non_pointwise_beyond_probe_raises():  # a*x + jnp.where(x.size > 4096, jnp.mean(x), 0.0) → solver's first-chunk check raises, hint appended
def test_engine_fallback_warning_text():  # method="trf" warning contains "not full-batch equivalent" and "strategy=Chunked()"
def test_stall_raises_exact_solver_error():  # max_nfev=1 → ExactSolverError, "strategy=Chunked()" in str(e)
def test_ignored_options_do_not_raise(dispatch):  # stability="auto", rescale_data=False, max_jacobian_elements_for_svd=10 → ["exact"]
def test_tolerances_reach_exact(monkeypatch):  # curve_fit_large(..., ftol=1e-10, chunk_size=500) → exact_lm_fit gets both
# Review Focus 1-4
def test_nan_y_raises_value_error(): y2 = y.copy(); y2[7] = np.nan → pytest.raises(ValueError, match="finite")
def test_p0_none_fits(): popt within rtol 1e-3 of truth (noise 1e-3), dispatch == ["exact"]
def test_docstring_good_model_fits(): model(x, a, b) = a*jnp.exp(-b*x.astype(jnp.int32)) on x = arange → ["exact"]
def test_absolute_sigma_pcov_matches_curve_fit(): scale-aware |dC| <= 1e-6 vs curve_fit(..., sigma, absolute_sigma=True)
```

- [ ] **Step 2: Run.** `uv run pytest tests/streaming/test_default_exact.py -n 0 -q`. Expected: FAIL.

- [ ] **Step 3: Implement**, following the spec's "Dispatch, step by step" order exactly.
  - `Auto.resolve`: below the threshold with detection on, return `in_memory`; otherwise
    `self.exact if self.exact is not None else self.chunked`.
  - In `fit_with_strategy`, after validation:
    - `if isinstance(strategy, Auto): auto = strategy; strategy = auto.resolve(n_points)`;
    - then `if isinstance(strategy, Exact):` compute `opts = {**auto.chunked.options, **kwargs}`, then call
      `_exact_ineligible(auto, opts, f, xdata, n_points, sigma, p0, bounds, check_finite)`. It normalizes `p0` with
      `normalize_p0_bounds` and slices the probe itself;
    - on an `"engine"` reason: `warnings.warn(msg, UserWarning, stacklevel=3)` with the spec's text, log it at
      INFO, and use `auto.chunked`;
    - on a `"data"` reason: `raise ValueError(f"curve_fit_large: exact streaming cannot fit this input
      ({reason}). Chunked GLS is an approximation, not full-batch equivalent for nonlinear models; pass
      strategy=Chunked() to accept it.")`;
    - otherwise `_fit_exact(_effective_exact(auto, opts), ..., kwargs={}, route="default")`. The tolerance
      keys already sit in the `Exact`, so `reject_owned_kwargs` (`large_fit.py:176`) sees no duplicates.
  - `_fit_exact` with `route="default"`:
    - catches `ValueError` whose message starts "model is not pointwise" and re-raises it, with the same
      `strategy=Chunked()` hint appended (`raise ... from err`);
    - catches `ExactSolverError` and re-raises it with the same `.result` and " (default exact path;
      strategy=Chunked() restores the chunked GLS fit)" appended.
  - The explicit route is unchanged, including its strict checks.
  - Update the `Auto` docstring.

- [ ] **Step 4: Run.** The new file, `uv run pytest tests/streaming -m "not slow" -q`, and
  `grep -rln "Auto(" tests/`. Expected: new tests pass. Write the existing failures to
  `<scratchpad>/task6-failures.txt` for Task 7. Do not fix them here.

### Task 7: Legacy shim, pinned GLS tests, caller sweep (Change 1, wiring)

**Files:**
- Modify: `nlsq/__init__.py:900-978` (`_legacy_to_strategy`) and the `curve_fit_large` docstring (1013-1210)
- Modify: tests listed in `task6-failures.txt`, plus every test asserting GLS or chunk-level behaviour
  through the default (`tests/streaming/test_large_dataset*.py`, `test_large_dataset_equivalence_flag.py`,
  `test_streaming_pr30_regressions.py`)
- Modify: GUI and CLI callers, only where they read GLS-only output
- Test: `tests/streaming/test_default_exact.py`

**Interfaces:**
- Consumes: `Auto.exact` and `_EXACT_OPTION_KEYS` (Task 6).
- Produces: no shim code change. `_legacy_to_strategy` keeps building `Auto` as today. The default
  `Auto.exact = Exact()` picks up `chunk_size` and tolerances from `Chunked` via `_effective_exact`
  (Task 6). This task pins tests, sweeps callers and updates docs.

- [ ] **Step 1: Write the shim test.** `_legacy_to_strategy(ftol=1e-10, chunk_size=500).exact == Exact()`,
  and `_effective_exact(that_auto, {"ftol": 1e-10}) == Exact(chunk_size=500, ftol=1e-10)`. Run it; it passes
  once Task 6 is in, and pins the transfer.

- [ ] **Step 4: Pin the legacy tests.**
  - Every test asserting GLS or chunk-level behaviour through the default gets `strategy=Chunked(...)`,
    reproducing its old chunk settings, or `strategy=Auto(..., exact=None)` when it tests the threshold
    switch. Never loosen an assertion.
  - Run `grep -rn "\.resolve(" nlsq/ tests/`: each caller handles an `Exact` return.
  - Run `grep -rn "curve_fit_large" nlsq/gui_qt nlsq/cli nlsq/core/workflow.py nlsq/__init__.py`: a caller
    that reads GLS-only output (a chunk count, `equivalent_to_full_batch` from a chunked result) passes
    `strategy=Chunked()`.

- [ ] **Step 5: Docstring.** Rewrite the "Automatically selects processing strategy" block and the Notes:
  - exact by default at or above the threshold;
  - the eligibility list: engine reasons warn and use chunked; data reasons raise (D1);
  - scalar and signed `sigma` support;
  - `ExactSolverError`;
  - `max_nfev` counting full-data passes;
  - the pointwise model contract (the probe is a guard);
  - the opt-out `strategy=Chunked()`.

- [ ] **Step 6: Run.**
  - `make test`.
  - `uv run pytest tests/regression -n 0 -q`, in the background. These are serial and slow.

  Expected: 0 failures. A failing example points at a caller relying on GLS; fix the caller, not the test.

### Task 8: A/B gate flag, docs, measurement, baseline (Change 1, close-out)

**Files:**
- Modify: `benchmarks/studies/ab_nlsq.py:66-82,145-160`. Add the `--cost-gate {equal,no-worse}` flag
  (default `equal`, today's behaviour).
- Test: `tests/benchmarks/test_ab_nlsq.py` (create if absent)
- Modify: `CHANGELOG.md`. Unreleased:
  - "Changed (breaking)":
    - default exact;
    - engine-reason fallback warning;
    - data reasons raise unless `strategy=Chunked()` is given (multi-feature `xdata`, non-pointwise model,
      non-finite `sigma` under `check_finite=False`);
    - `max_nfev` meaning;
    - `ExactSolverError`.
  - "Added": `compute_covariance`, TSQR covariance, `covariance_method`.
- Modify: `CLAUDE.md` ("Recent Changes": replace the opt-in exact line)
- Create: `benchmarks/baselines/2026-11-1x-default-exact-ab.md` and
  `benchmarks/baselines/2026-11-1x-library-comparison-1k-100M.md`

**Interfaces:**
- Consumes: Tasks 1–7 on the branch.
- Produces:
  - `check_correctness(expected, base, cand, cost_gate="equal") -> tuple[list[str], set[Case]]`, returning
    `(violations, skipped)`.
    - `"no-worse"`: both arms need status `ok` and a finite cost, or the case is a violation. Then:
      - `c_cost > b_cost * (1 + COST_RTOL)` is a violation;
      - `b_cost > c_cost * (1 + COST_RTOL)` (base missed the optimum) puts the case in `skipped`, untimed;
      - otherwise the case is timed.
    - `"equal"`: unchanged, and `skipped` is empty.
  - The timing loop (`ab_nlsq.py:170-198`) iterates `expected - skipped`. A missing or non-finite
    `warm_median_s` on a timed case is a violation. Skipped cases are printed in their own section and
    excluded from the verdict. Update the existing callers of `check_correctness` for the tuple return.

- [ ] **Step 1: Write the tests** for `check_correctness` with fixture dicts:
  - equal mode is unchanged;
  - `no-worse` accepts a lower cand cost;
  - `no-worse` rejects a higher cost;
  - `no-worse` skips a base that is status-`ok` but costlier than cand beyond rtol (the inaccurate-but-ok
    GLS case), and the timing loop does not index it;
  - a non-`ok` status in either arm is a violation;
  - a timed case with `warm_median_s=None` is a violation, not a `TypeError`.

  Run them (FAIL), implement, run them again (PASS).

- [ ] **Step 2: A/B, base = branch point, cand = branch tip,** with `--libraries nlsq_large --cost-gate
  no-worse`:
  - GPU and CPU at 1M and 10M;
  - 100M with `--rounds 3 --repeats 2` (base GLS on CPU is 30–80 s per fit);
  - all four problems;
  - absolute paths.

  Gate:
  - every candidate median is at or below base on GPU at 10M and 100M;
  - no other cell is slower beyond the spread;
  - Task 5 Step 4b's route table shows `"normal"` for all four problems, or the TSQR cost is recorded.

- [ ] **Step 3: Library comparison rerun.** Run `benchmarks/compare/run_library_comparison.py` at 1k–100M on
  both devices, one `--out` per invocation.

  Gate: every NLSQ cell is within cost `rtol 1e-6` of the best, and the 1M two_exp DNRO is gone.

  Write the baseline in the format of `2026-11-09-library-comparison-1k-100M.md`, with recomputed
  cross-invocation cost gaps and machine state.

- [ ] **Step 4: CHANGELOG and CLAUDE.md.** One factual paragraph each, with the numbers from Steps 2–3.

- [ ] **Step 5: Run.** `make verify`, then `graphify update .`. Expected: clean.

- [ ] **Step 6: Commit (on approval).** Command:
  `git commit -m "feat(curve_fit_large): use exact streaming LM by default above threshold"`, with a body
  stating the breaking change and the opt-out.
