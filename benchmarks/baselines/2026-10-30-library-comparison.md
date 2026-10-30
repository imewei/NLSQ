<!-- Measured at commit 3bc0e9b (final code of branch chore/slim-install-and-overhead), AFTER the small-fit overhead work. Machine was not idle (load average ~4-7: another process was using a core), so absolute numbers carry noise; ratios between libraries in the same run are the comparable quantity. The pre-optimisation run is 2026-10-30-library-comparison-before-overhead.md. -->

# Library comparison

- Backend: `cpu`; devices: cpu
- CPU: 13th Gen Intel(R) Core(TM) i9-13900H (20 logical)

Caveats:

- Stopping criteria are analogous, not identical: every library gets the same tolerance and iteration budget for each criterion it exposes. A library is ranked only if its final cost matches the best cost in the case (relative 1e-6); its own `converged` flag is reported but does not decide that. A ranked library can still show `lib converged` = False; the flag is not decisive and its cause is not diagnosed here.
- lmfit wraps SciPy (`leastsq` / `least_squares`); its timings include lmfit's own overhead on top of SciPy.
- JAXopt: README states "JAXopt is no longer maintained nor developed".
- Cold = first fit call in a fresh process, after library imports and a trivial backend warm-up (imports and library loading are excluded; JIT tracing/compilation and first-call allocation are included); warm = median and IQR of repeated calls.
- Sequential libraries (scipy, lmfit) show cold close to warm; vmapped JAX batches (jaxopt, optimistix, nlsq) still pay first-call compilation, so their cold time is far above warm.
- At equal cost a ranked library can show a larger max parameter error than its peers on flat-valley problems (two_exp: optimistix 0.47 vs 0.07). Cost is the ranking criterion; parameter error is reported for information.
- GPU was not measured: this run used the CPU backend.

## Results

### bounded_exp, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| scipy | end_to_end | 0.00646 | 0.0045 | 0.000174 | 18 | 0 | 0.2 | True |
| lmfit | end_to_end | 0.00938 | 0.00745 | 0.000435 | 72 | 0 | 0.2 | True |
| nlsq | end_to_end | 2.05 | 0.0473 | 0.00689 | 18 | 9.99e-13 | 0.2 | True |

### bounded_exp, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| nlsq | end_to_end | 2.39 | 0.338 | 0.218 | 18 | 6.98e-13 | 0.2 | True |
| scipy | end_to_end | 0.332 | 0.387 | 0.022 | 18 | 0 | 0.2 | True |
| lmfit | end_to_end | 0.348 | 0.529 | 0.171 | 72 | 0 | 0.2 | True |

### bounded_exp, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| nlsq | end_to_end | 5.49 | 2.21 | 0.0697 | 18 | 6.38e-13 | 0.2 | True |
| scipy | end_to_end | 2.8 | 3 | 0.125 | 18 | 0 | 0.2 | True |
| lmfit | end_to_end | 3 | 3.02 | 0.214 | 72 | 0 | 0.2 | True |

### exp_decay, n=200, batch=256

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.603 | 0.0263 | 0.00149 | - | 0 | 0.0465 | True |
| optimistix | end_to_end | 0.568 | 0.0849 | 0.00262 | - | 0 | 0.0465 | True |
| scipy | end_to_end | 0.215 | 0.21 | 0.000796 | 5 | 1.84e-13 | 0.0465 | True |
| lmfit | end_to_end | 0.433 | 0.431 | 0.00347 | 21 | 1.14e-13 | 0.0465 | True |
| nlsq | end_to_end | 2.13 | 0.961 | 0.107 | 5 | 1.84e-13 | 0.0465 | True |

#### exp_decay, n=200, batch=256: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.583 | 0.0268 | 0.00204 | - | 0 | 0.0465 | True |
| optimistix | jitted_core | 0.547 | 0.0877 | 0.00481 | - | 0 | 0.0465 | True |

### exp_decay, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.527 | 0.000358 | 5.99e-05 | - | 0 | 0.00839 | True |
| optimistix | end_to_end | 0.238 | 0.000756 | 0.000193 | - | 0 | 0.00839 | True |
| scipy | end_to_end | 0.00276 | 0.00106 | 7.43e-05 | 5 | 4.47e-15 | 0.00839 | True |
| lmfit | end_to_end | 0.00325 | 0.00182 | 0.000245 | 21 | 1.68e-15 | 0.00839 | True |
| nlsq | end_to_end | 1 | 0.00596 | 0.00268 | 5 | 4.47e-15 | 0.00839 | True |

#### exp_decay, n=1000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.538 | 0.000788 | 0.00017 | - | 0 | 0.00839 | True |
| optimistix | jitted_core | 0.242 | 0.00108 | 7.64e-05 | - | 0 | 0.00839 | True |

### exp_decay, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.287 | 0.009 | 0.00104 | - | 1.14e-16 | 0.000569 | True |
| optimistix | end_to_end | 0.258 | 0.0219 | 0.00206 | - | 0 | 0.000569 | True |
| lmfit | end_to_end | 0.0283 | 0.0272 | 0.00138 | 21 | 3.41e-16 | 0.000569 | True |
| nlsq | end_to_end | 0.977 | 0.0581 | 0.00272 | 7 | 1.14e-16 | 0.000569 | True |
| scipy | end_to_end | 0.0871 | 0.0844 | 0.00715 | 5 | 1.14e-16 | 0.000569 | True |

#### exp_decay, n=100000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.277 | 0.00824 | 0.000896 | - | 1.14e-16 | 0.000569 | True |
| optimistix | jitted_core | 0.265 | 0.0221 | 0.00164 | - | 0 | 0.000569 | True |

### exp_decay, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.385 | 0.113 | 0.00294 | - | 1.82e-16 | 7.62e-05 | True |
| optimistix | end_to_end | 0.481 | 0.25 | 0.0116 | - | 1.82e-16 | 7.62e-05 | True |
| lmfit | end_to_end | 0.449 | 0.395 | 0.00855 | 21 | 1.82e-16 | 7.62e-05 | True |
| nlsq | end_to_end | 1.45 | 0.657 | 0.0263 | 7 | 1.82e-16 | 7.62e-05 | True |
| scipy | end_to_end | 0.915 | 0.98 | 0.0555 | 5 | 0 | 7.62e-05 | True |

#### exp_decay, n=1000000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.373 | 0.111 | 0.00812 | - | 1.82e-16 | 7.62e-05 | True |
| optimistix | jitted_core | 0.465 | 0.257 | 0.0137 | - | 1.82e-16 | 7.62e-05 | True |

### gaussian, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.618 | 0.000877 | 0.000215 | - | 0 | 0.000733 | True |
| optimistix | end_to_end | 0.253 | 0.00105 | 0.000104 | - | 0 | 0.000733 | True |
| scipy | end_to_end | 0.00323 | 0.00173 | 0.000107 | 9 | 0 | 0.000733 | True |
| lmfit | end_to_end | 0.00353 | 0.00236 | 0.000313 | 38 | 5.24e-14 | 0.000733 | True |
| nlsq | end_to_end | 1.01 | 0.00759 | 0.00196 | 9 | 0 | 0.000733 | True |

#### gaussian, n=1000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.599 | 0.000929 | 0.000988 | - | 0 | 0.000733 | True |
| optimistix | jitted_core | 0.255 | 0.00131 | 8.79e-05 | - | 0 | 0.000733 | True |

### gaussian, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.411 | 0.0382 | 0.00127 | - | 0 | 0.000156 | True |
| lmfit | end_to_end | 0.0472 | 0.0447 | 0.000287 | 34 | 1.14e-16 | 0.000156 | True |
| optimistix | end_to_end | 0.327 | 0.0636 | 0.0013 | - | 1.14e-16 | 0.000156 | True |
| nlsq | end_to_end | 1.09 | 0.116 | 0.00388 | 13 | 1.14e-16 | 0.000156 | True |
| scipy | end_to_end | 0.276 | 0.153 | 0.0401 | 9 | 1.14e-16 | 0.000156 | True |

#### gaussian, n=100000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.399 | 0.037 | 0.000829 | - | 0 | 0.000156 | True |
| optimistix | jitted_core | 0.328 | 0.065 | 0.00423 | - | 1.14e-16 | 0.000156 | True |

### gaussian, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.734 | 0.417 | 0.0151 | - | 1.82e-16 | 6.62e-05 | True |
| lmfit | end_to_end | 0.715 | 0.641 | 0.0372 | 34 | 1.82e-16 | 6.62e-05 | True |
| optimistix | end_to_end | 1.03 | 0.763 | 0.0413 | - | 0 | 6.62e-05 | True |
| nlsq | end_to_end | 2.3 | 1.43 | 0.0339 | 13 | 0 | 6.62e-05 | True |
| scipy | end_to_end | 1.83 | 1.64 | 0.128 | 9 | 1.82e-16 | 6.62e-05 | True |

#### gaussian, n=1000000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.747 | 0.399 | 0.0183 | - | 1.82e-16 | 6.62e-05 | True |
| optimistix | jitted_core | 1 | 0.744 | 0.0309 | - | 0 | 6.62e-05 | True |

### two_exp, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.687 | 0.000895 | 0.000143 | - | 3.72e-16 | 0.186 | True |
| scipy | end_to_end | 0.00416 | 0.00321 | 0.000139 | 15 | 4.56e-10 | 0.185 | True |
| lmfit | end_to_end | 0.00501 | 0.00366 | 0.000282 | 73 | 3.26e-09 | 0.184 | True |
| optimistix | end_to_end | 0.323 | 0.00551 | 0.000242 | - | 0 | 0.186 | True |
| nlsq | end_to_end | 0.987 | 0.0101 | 0.00106 | 15 | 4.52e-10 | 0.185 | True |

#### two_exp, n=1000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.717 | 0.00315 | 0.00202 | - | 3.72e-16 | 0.186 | True |
| optimistix | jitted_core | 0.27 | 0.00542 | 0.000208 | - | 0 | 0.186 | True |

### two_exp, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.386 | 0.0472 | 0.0065 | - | 0 | 0.0666 | True |
| lmfit | end_to_end | 0.077 | 0.0737 | 0.00132 | 42 | 6.82e-16 | 0.0666 | True |
| optimistix | end_to_end | 0.362 | 0.0901 | 0.00372 | - | 0 | 0.473 | True |
| nlsq | end_to_end | 1.1 | 0.122 | 0.00472 | 11 | 2.27e-16 | 0.0666 | True |
| scipy | end_to_end | 0.34 | 0.263 | 0.0619 | 9 | 0 | 0.0666 | True |

#### two_exp, n=100000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.386 | 0.0432 | 0.000868 | - | 0 | 0.0666 | True |
| optimistix | jitted_core | 0.342 | 0.0901 | 0.00709 | - | 0 | 0.473 | True |

### two_exp, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.993 | 0.696 | 0.172 | - | 0 | 0.0125 | True |
| lmfit | end_to_end | 1.21 | 1.09 | 0.0421 | 42 | 0 | 0.0125 | True |
| optimistix | end_to_end | 1.86 | 1.71 | 0.692 | - | 0 | 0.0125 | True |
| nlsq | end_to_end | 2.75 | 1.94 | 0.0412 | 11 | 0 | 0.0125 | True |
| scipy | end_to_end | 2.1 | 2.06 | 0.0868 | 9 | 0 | 0.0125 | True |

#### two_exp, n=1000000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 1.17 | 0.945 | 0.198 | - | 0 | 0.0125 | True |
| optimistix | jitted_core | 2.71 | 2.54 | 0.1 | - | 0 | 0.0125 | True |

## Capability matrix

| library | bounds | sigma/weights | covariance | vmappable | maintained | note |
|---|---|---|---|---|---|---|
| nlsq | yes | yes | yes | no | yes |  |
| scipy | yes | yes | yes | no | yes |  |
| lmfit | yes | yes | yes | no | yes | wraps SciPy leastsq / least_squares |
| jaxopt | no | no | no | yes | no | README: "JAXopt is no longer maintained nor developed"; stop_criterion=madsen-nielsen: xtol and gtol active, tol unused (the default grad-l2-norm ignores xtol/gtol) |
| optimistix | no | no | no | yes | yes | rtol/atol only; no separate gtol |

## Unavailable, unsupported or excluded

- jaxopt (-) on bounded_exp n=1000: unsupported. no bounds support
- optimistix (-) on bounded_exp n=1000: unsupported. no bounds support
- jaxopt (-) on bounded_exp n=100000: unsupported. no bounds support
- optimistix (-) on bounded_exp n=100000: unsupported. no bounds support
- jaxopt (-) on bounded_exp n=1000000: unsupported. no bounds support
- optimistix (-) on bounded_exp n=1000000: unsupported. no bounds support

## Package versions

- numpy: 2.5.4
- scipy: 1.18.1
- jax: 0.12.0.1
- jaxlib: 0.12.0.1
- nlsq: 0.7.7.dev3
- lmfit: 1.3.4
- jaxopt: 0.8.5
- optimistix: 0.1.0
