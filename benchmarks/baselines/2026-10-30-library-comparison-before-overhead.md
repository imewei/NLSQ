<!-- Measured at commit 933461d, BEFORE the small-fit overhead work (shared JIT kernels, wrapper cache, quiet logging). Kept as the "before" half of the before/after comparison; see 2026-10-30-library-comparison.md for the build that ships. -->

# Library comparison

- Backend: `cpu`; devices: cpu
- CPU: 13th Gen Intel(R) Core(TM) i9-13900H (20 logical)

Caveats:

- Stopping criteria are analogous, not identical: every library gets the same tolerance and iteration budget for each criterion it exposes. A library is ranked only if its final cost matches the best cost in the case (relative 1e-6); its own `converged` flag is reported but does not decide that. A ranked library can still show `lib converged` = False; the flag is not decisive and its cause is not diagnosed here.
- lmfit wraps SciPy (`leastsq` / `least_squares`); its timings include lmfit's own overhead on top of SciPy.
- JAXopt: README states "JAXopt is no longer maintained nor developed".
- Cold = first fit call in a fresh process, after library imports and a trivial backend warm-up (imports and library loading are excluded; JIT tracing/compilation and first-call allocation are included); warm = median and IQR of repeated calls.
- In the batch case cold is close to warm because the first call is the entire 256-fit batch.
- At equal cost a ranked library can show a larger max parameter error than its peers on flat-valley problems (two_exp: optimistix 0.47 vs 0.07). Cost is the ranking criterion; parameter error is reported for information.
- GPU was not measured: this run used the CPU backend.

## Results

### bounded_exp, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| scipy | end_to_end | 0.0054 | 0.00435 | 1.12e-05 | 18 | 0 | 0.2 | True |
| lmfit | end_to_end | 0.00828 | 0.00712 | 0.000123 | 72 | 0 | 0.2 | True |
| nlsq | end_to_end | 1.42 | 0.237 | 0.00894 | 18 | 9.99e-13 | 0.2 | True |

### bounded_exp, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| lmfit | end_to_end | 0.218 | 0.278 | 0.00497 | 72 | 0 | 0.2 | True |
| scipy | end_to_end | 0.281 | 0.281 | 0.039 | 18 | 0 | 0.2 | True |
| nlsq | end_to_end | 1.66 | 0.41 | 0.0248 | 18 | 6.98e-13 | 0.2 | True |

### bounded_exp, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| nlsq | end_to_end | 2.91 | 1.53 | 0.0272 | 18 | 6.38e-13 | 0.2 | True |
| scipy | end_to_end | 2.49 | 2.48 | 0.131 | 18 | 0 | 0.2 | True |
| lmfit | end_to_end | 2.43 | 2.52 | 0.0623 | 72 | 0 | 0.2 | True |

### exp_decay, n=200, batch=256

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.5 | 0.0177 | 0.000772 | - | 0 | 0.0465 | True |
| optimistix | end_to_end | 0.458 | 0.0584 | 0.0113 | - | 0 | 0.0465 | True |
| scipy | end_to_end | 0.21 | 0.206 | 0.00224 | 5 | 1.84e-13 | 0.0465 | True |
| lmfit | end_to_end | 0.414 | 0.414 | 0.00227 | 21 | 1.14e-13 | 0.0465 | True |
| nlsq | end_to_end | 44.3 | 48.4 | 3.55 | 5 | 1.84e-13 | 0.0465 | True |

#### exp_decay, n=200, batch=256: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.485 | 0.0188 | 0.0038 | - | 0 | 0.0465 | True |
| optimistix | jitted_core | 0.464 | 0.0577 | 0.00692 | - | 0 | 0.0465 | True |

### exp_decay, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.48 | 0.000837 | 8.77e-05 | - | 0 | 0.00839 | True |
| scipy | end_to_end | 0.00196 | 0.00101 | 3.58e-05 | 5 | 4.47e-15 | 0.00839 | True |
| optimistix | end_to_end | 0.204 | 0.00119 | 0.000128 | - | 0 | 0.00839 | True |
| lmfit | end_to_end | 0.00259 | 0.00179 | 7.21e-05 | 21 | 1.68e-15 | 0.00839 | True |
| nlsq | end_to_end | 0.824 | 0.167 | 0.0163 | 5 | 4.47e-15 | 0.00839 | True |

#### exp_decay, n=1000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.47 | 0.000759 | 7.99e-05 | - | 0 | 0.00839 | True |
| optimistix | jitted_core | 0.201 | 0.00111 | 0.000136 | - | 0 | 0.00839 | True |

### exp_decay, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.255 | 0.0116 | 0.00294 | - | 1.14e-16 | 0.000569 | True |
| lmfit | end_to_end | 0.0247 | 0.0229 | 0.00018 | 21 | 3.41e-16 | 0.000569 | True |
| optimistix | end_to_end | 0.221 | 0.0311 | 0.0107 | - | 0 | 0.000569 | True |
| scipy | end_to_end | 0.0519 | 0.058 | 0.00864 | 5 | 1.14e-16 | 0.000569 | True |
| nlsq | end_to_end | 0.805 | 0.219 | 0.00959 | 7 | 1.14e-16 | 0.000569 | True |

#### exp_decay, n=100000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.27 | 0.013 | 0.0047 | - | 1.14e-16 | 0.000569 | True |
| optimistix | jitted_core | 0.223 | 0.0264 | 0.00692 | - | 0 | 0.000569 | True |

### exp_decay, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.407 | 0.167 | 0.0226 | - | 1.82e-16 | 7.62e-05 | True |
| lmfit | end_to_end | 0.291 | 0.248 | 0.00293 | 21 | 1.82e-16 | 7.62e-05 | True |
| optimistix | end_to_end | 0.474 | 0.25 | 0.0217 | - | 1.82e-16 | 7.62e-05 | True |
| nlsq | end_to_end | 1.09 | 0.63 | 0.0337 | 7 | 1.82e-16 | 7.62e-05 | True |
| scipy | end_to_end | 0.635 | 0.681 | 0.0488 | 5 | 0 | 7.62e-05 | True |

#### exp_decay, n=1000000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.414 | 0.148 | 0.0132 | - | 1.82e-16 | 7.62e-05 | True |
| optimistix | jitted_core | 0.45 | 0.264 | 0.0282 | - | 1.82e-16 | 7.62e-05 | True |

### gaussian, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.54 | 0.00097 | 0.000152 | - | 0 | 0.000733 | True |
| optimistix | end_to_end | 0.215 | 0.000977 | 0.000198 | - | 0 | 0.000733 | True |
| scipy | end_to_end | 0.0026 | 0.00172 | 9.34e-05 | 9 | 0 | 0.000733 | True |
| lmfit | end_to_end | 0.00301 | 0.00223 | 0.000152 | 38 | 5.24e-14 | 0.000733 | True |
| nlsq | end_to_end | 0.774 | 0.171 | 0.00676 | 9 | 0 | 0.000733 | True |

#### gaussian, n=1000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.527 | 0.000955 | 0.00014 | - | 0 | 0.000733 | True |
| optimistix | jitted_core | 0.211 | 0.00105 | 0.000359 | - | 0 | 0.000733 | True |

### gaussian, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.365 | 0.0352 | 0.00239 | - | 0 | 0.000156 | True |
| lmfit | end_to_end | 0.0382 | 0.0369 | 0.000175 | 34 | 1.14e-16 | 0.000156 | True |
| optimistix | end_to_end | 0.321 | 0.0897 | 0.00516 | - | 1.14e-16 | 0.000156 | True |
| scipy | end_to_end | 0.104 | 0.111 | 0.00207 | 9 | 1.14e-16 | 0.000156 | True |
| nlsq | end_to_end | 0.983 | 0.302 | 0.0341 | 13 | 1.14e-16 | 0.000156 | True |

#### gaussian, n=100000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.365 | 0.0357 | 0.00187 | - | 0 | 0.000156 | True |
| optimistix | jitted_core | 0.306 | 0.0904 | 0.00804 | - | 1.14e-16 | 0.000156 | True |

### gaussian, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| lmfit | end_to_end | 0.46 | 0.408 | 0.00429 | 34 | 1.82e-16 | 6.62e-05 | True |
| jaxopt | end_to_end | 0.959 | 0.625 | 0.0624 | - | 1.82e-16 | 6.62e-05 | True |
| optimistix | end_to_end | 0.9 | 0.828 | 0.0233 | - | 0 | 6.62e-05 | True |
| scipy | end_to_end | 1.08 | 1.15 | 0.0607 | 9 | 1.82e-16 | 6.62e-05 | True |
| nlsq | end_to_end | 1.77 | 1.21 | 0.0644 | 13 | 0 | 6.62e-05 | True |

#### gaussian, n=1000000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.925 | 0.625 | 0.0947 | - | 1.82e-16 | 6.62e-05 | True |
| optimistix | jitted_core | 1.05 | 0.859 | 0.0558 | - | 0 | 6.62e-05 | True |

### two_exp, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.573 | 0.000933 | 0.000138 | - | 3.72e-16 | 0.186 | True |
| scipy | end_to_end | 0.00404 | 0.00311 | 2.6e-05 | 15 | 4.56e-10 | 0.185 | True |
| lmfit | end_to_end | 0.00428 | 0.00348 | 0.000161 | 73 | 3.26e-09 | 0.184 | True |
| optimistix | end_to_end | 0.215 | 0.0042 | 0.00113 | - | 0 | 0.186 | True |
| nlsq | end_to_end | 0.785 | 0.176 | 0.00333 | 15 | 4.52e-10 | 0.185 | True |

#### two_exp, n=1000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.572 | 0.000988 | 8.02e-05 | - | 3.72e-16 | 0.186 | True |
| optimistix | jitted_core | 0.211 | 0.00303 | 0.00126 | - | 0 | 0.186 | True |

### two_exp, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | end_to_end | 0.364 | 0.0436 | 0.00127 | - | 0 | 0.0666 | True |
| lmfit | end_to_end | 0.0639 | 0.0616 | 0.00111 | 42 | 6.82e-16 | 0.0666 | True |
| optimistix | end_to_end | 0.358 | 0.127 | 0.00573 | - | 0 | 0.473 | True |
| scipy | end_to_end | 0.138 | 0.16 | 0.00829 | 9 | 0 | 0.0666 | True |
| nlsq | end_to_end | 0.993 | 0.356 | 0.0299 | 11 | 2.27e-16 | 0.0666 | True |

#### two_exp, n=100000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.347 | 0.0429 | 0.00191 | - | 0 | 0.0666 | True |
| optimistix | jitted_core | 0.334 | 0.132 | 0.0128 | - | 0 | 0.473 | True |

### two_exp, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| lmfit | end_to_end | 0.804 | 0.701 | 0.00366 | 42 | 0 | 0.0125 | True |
| jaxopt | end_to_end | 1.39 | 1.02 | 0.118 | - | 0 | 0.0125 | True |
| scipy | end_to_end | 1.41 | 1.31 | 0.0215 | 9 | 0 | 0.0125 | True |
| nlsq | end_to_end | 2.16 | 1.6 | 0.0932 | 11 | 0 | 0.0125 | True |
| optimistix | end_to_end | 1.93 | 1.79 | 0.0368 | - | 0 | 0.0125 | True |

#### two_exp, n=1000000: resident data (jitted_core)

Inputs are already on the device. The timed region is the jitted solve plus synchronisation and result materialisation (`np.asarray` of the device outputs and per-fit result construction); it excludes the host-to-device transfer of the inputs that `end_to_end` includes, so it is not comparable with the headline table.

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 1.24 | 1 | 0.0943 | - | 0 | 0.0125 | True |
| optimistix | jitted_core | 1.92 | 1.69 | 0.125 | - | 0 | 0.0125 | True |

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
- nlsq: 0.7.7.dev8
- lmfit: 1.3.4
- jaxopt: 0.8.5
- optimistix: 0.1.0
