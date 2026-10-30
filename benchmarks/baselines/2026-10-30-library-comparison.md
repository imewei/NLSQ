# Library comparison

- Backend: `cpu`; devices: cpu
- CPU: 13th Gen Intel(R) Core(TM) i9-13900H (20 logical)

Caveats:

- Stopping criteria are analogous, not identical: every library gets the same tolerance and iteration budget for each criterion it exposes. A library is ranked only if its final cost matches the best cost in the case (relative 1e-6); its own `converged` flag is reported but does not decide that. A ranked library can still show `lib converged` = False: it reached the optimum but ran out its iteration budget, so its time includes the unused iterations.
- lmfit wraps SciPy (`leastsq` / `least_squares`); its timings include lmfit's own overhead on top of SciPy.
- JAXopt: README states "JAXopt is no longer maintained nor developed".
- Cold = first call in a fresh process after a trivial backend warm-up (includes JIT compilation); warm = median and IQR of repeated calls.
- GPU was not measured: this run used the CPU backend.

## Results

### bounded_exp, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| scipy | end_to_end | 0.0932 | 0.00439 | 8.08e-05 | 18 | 0 | 0.2 | True |
| lmfit | end_to_end | 0.405 | 0.0069 | 0.000188 | 72 | 0 | 0.2 | True |
| nlsq | end_to_end | 1.58 | 0.253 | 0.0276 | 18 | 9.99e-13 | 0.2 | True |

### bounded_exp, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| lmfit | end_to_end | 0.707 | 0.28 | 0.00602 | 72 | 0 | 0.2 | True |
| scipy | end_to_end | 0.363 | 0.284 | 0.0129 | 18 | 0 | 0.2 | True |
| nlsq | end_to_end | 1.77 | 0.424 | 0.0214 | 18 | 6.98e-13 | 0.2 | True |

### bounded_exp, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| nlsq | end_to_end | 2.73 | 1.53 | 0.146 | 18 | 6.38e-13 | 0.2 | True |
| scipy | end_to_end | 2.44 | 2.55 | 0.105 | 18 | 0 | 0.2 | True |
| lmfit | end_to_end | 2.96 | 2.59 | 0.165 | 72 | 0 | 0.2 | True |

### exp_decay, n=200, batch=256

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| optimistix | jitted_core | 0.473 | 0.0576 | 0.00303 | - | 0 | 0.0465 | True |
| optimistix | end_to_end | 0.479 | 0.0596 | 0.00263 | - | 0 | 0.0465 | True |
| scipy | end_to_end | 0.292 | 0.205 | 0.0023 | 5 | 1.84e-13 | 0.0465 | True |
| lmfit | end_to_end | 0.807 | 0.417 | 0.00256 | 21 | 1.14e-13 | 0.0465 | True |
| jaxopt | jitted_core | 2.59 | 2.18 | 0.148 | - | 0 | 0.0465 | False |
| jaxopt | end_to_end | 2.49 | 2.19 | 0.0815 | - | 0 | 0.0465 | False |
| nlsq | end_to_end | 44.1 | 48.4 | 3.54 | 5 | 1.84e-13 | 0.0465 | True |

### exp_decay, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| optimistix | jitted_core | 0.205 | 0.000505 | 0.000146 | - | 0 | 0.00839 | True |
| jaxopt | jitted_core | 0.508 | 0.000632 | 6.65e-05 | - | 0 | 0.00839 | True |
| jaxopt | end_to_end | 0.485 | 0.000814 | 7.38e-05 | - | 0 | 0.00839 | True |
| optimistix | end_to_end | 0.21 | 0.000985 | 4.07e-05 | - | 0 | 0.00839 | True |
| scipy | end_to_end | 0.0928 | 0.0011 | 8.35e-05 | 5 | 4.47e-15 | 0.00839 | True |
| lmfit | end_to_end | 0.435 | 0.00179 | 0.00013 | 21 | 1.68e-15 | 0.00839 | True |
| nlsq | end_to_end | 0.832 | 0.184 | 0.0297 | 5 | 4.47e-15 | 0.00839 | True |

### exp_decay, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.261 | 0.00745 | 0.00513 | - | 1.14e-16 | 0.000569 | True |
| jaxopt | end_to_end | 0.28 | 0.00818 | 0.000355 | - | 1.14e-16 | 0.000569 | True |
| lmfit | end_to_end | 0.43 | 0.0229 | 0.000183 | 21 | 3.41e-16 | 0.000569 | True |
| optimistix | jitted_core | 0.213 | 0.0284 | 0.00661 | - | 0 | 0.000569 | True |
| optimistix | end_to_end | 0.23 | 0.0332 | 0.00592 | - | 0 | 0.000569 | True |
| scipy | end_to_end | 0.158 | 0.067 | 0.00666 | 5 | 1.14e-16 | 0.000569 | True |
| nlsq | end_to_end | 0.856 | 0.229 | 0.0101 | 7 | 1.14e-16 | 0.000569 | True |

### exp_decay, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| lmfit | end_to_end | 0.685 | 0.255 | 0.0105 | 21 | 1.82e-16 | 7.62e-05 | True |
| optimistix | jitted_core | 0.464 | 0.273 | 0.0281 | - | 1.82e-16 | 7.62e-05 | True |
| optimistix | end_to_end | 0.477 | 0.307 | 0.0161 | - | 1.82e-16 | 7.62e-05 | True |
| nlsq | end_to_end | 1.11 | 0.624 | 0.0457 | 7 | 1.82e-16 | 7.62e-05 | True |
| scipy | end_to_end | 0.678 | 0.689 | 0.0898 | 5 | 0 | 7.62e-05 | True |
| jaxopt | jitted_core | 18.5 | 17.8 | 0.852 | - | 1.82e-16 | 7.62e-05 | False |
| jaxopt | end_to_end | 17.7 | 18 | 0.495 | - | 1.82e-16 | 7.62e-05 | False |

### gaussian, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.536 | 0.000875 | 6.04e-05 | - | 0 | 0.000733 | True |
| optimistix | jitted_core | 0.217 | 0.000998 | 0.000134 | - | 0 | 0.000733 | True |
| optimistix | end_to_end | 0.216 | 0.00141 | 0.000192 | - | 0 | 0.000733 | True |
| scipy | end_to_end | 0.0851 | 0.00174 | 5.5e-05 | 9 | 0 | 0.000733 | True |
| lmfit | end_to_end | 0.429 | 0.00223 | 0.000176 | 38 | 5.24e-14 | 0.000733 | True |
| jaxopt | end_to_end | 0.535 | 0.00235 | 0.000227 | - | 0 | 0.000733 | True |
| nlsq | end_to_end | 0.842 | 0.182 | 0.0143 | 9 | 0 | 0.000733 | True |

### gaussian, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| lmfit | end_to_end | 0.434 | 0.0372 | 0.000929 | 34 | 0 | 0.000156 | True |
| optimistix | end_to_end | 0.304 | 0.0852 | 0.0153 | - | 0 | 0.000156 | True |
| optimistix | jitted_core | 0.312 | 0.0854 | 0.00205 | - | 0 | 0.000156 | True |
| scipy | end_to_end | 0.192 | 0.114 | 0.000784 | 9 | 0 | 0.000156 | True |
| nlsq | end_to_end | 0.97 | 0.319 | 0.0308 | 13 | 0 | 0.000156 | True |
| jaxopt | end_to_end | 1.71 | 1.37 | 0.00574 | - | 0 | 0.000156 | False |
| jaxopt | jitted_core | 1.67 | 1.37 | 0.015 | - | 0 | 0.000156 | False |

### gaussian, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| lmfit | end_to_end | 0.871 | 0.421 | 0.0093 | 34 | 1.82e-16 | 6.62e-05 | True |
| optimistix | jitted_core | 1.05 | 0.861 | 0.0469 | - | 0 | 6.62e-05 | True |
| optimistix | end_to_end | 0.942 | 0.897 | 0.0512 | - | 0 | 6.62e-05 | True |
| nlsq | end_to_end | 1.77 | 1.19 | 0.045 | 13 | 0 | 6.62e-05 | True |
| scipy | end_to_end | 1.33 | 1.21 | 0.0613 | 9 | 1.82e-16 | 6.62e-05 | True |
| jaxopt | end_to_end | 27.1 | 26.9 | 0.617 | - | 1.82e-16 | 6.62e-05 | False |
| jaxopt | jitted_core | 28.7 | 27.2 | 0.61 | - | 1.82e-16 | 6.62e-05 | False |

### two_exp, n=1000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| jaxopt | jitted_core | 0.585 | 0.00105 | 0.000135 | - | 3.72e-16 | 0.186 | True |
| jaxopt | end_to_end | 0.594 | 0.00146 | 0.000257 | - | 3.72e-16 | 0.186 | True |
| optimistix | end_to_end | 0.211 | 0.00293 | 0.00128 | - | 0 | 0.186 | True |
| optimistix | jitted_core | 0.21 | 0.00305 | 0.000974 | - | 0 | 0.186 | True |
| scipy | end_to_end | 0.0862 | 0.00312 | 6.65e-05 | 15 | 4.56e-10 | 0.185 | True |
| lmfit | end_to_end | 0.399 | 0.00354 | 0.000216 | 73 | 3.26e-09 | 0.184 | True |
| nlsq | end_to_end | 0.855 | 0.175 | 0.00596 | 15 | 4.52e-10 | 0.185 | True |

### two_exp, n=100000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| lmfit | end_to_end | 0.456 | 0.0628 | 0.001 | 42 | 6.82e-16 | 0.0666 | True |
| optimistix | end_to_end | 0.339 | 0.124 | 0.00346 | - | 0 | 0.473 | True |
| optimistix | jitted_core | 0.351 | 0.126 | 0.00481 | - | 0 | 0.473 | True |
| scipy | end_to_end | 0.218 | 0.167 | 0.00742 | 9 | 0 | 0.0666 | True |
| nlsq | end_to_end | 0.894 | 0.338 | 0.00998 | 11 | 2.27e-16 | 0.0666 | True |
| jaxopt | jitted_core | 1.57 | 1.25 | 0.0116 | - | 0 | 0.0666 | False |
| jaxopt | end_to_end | 1.59 | 1.26 | 0.0153 | - | 0 | 0.0666 | False |

### two_exp, n=1000000

| library | mode | cold (s) | warm median (s) | warm IQR (s) | nfev | cost gap | max param err | lib converged |
|---|---|---|---|---|---|---|---|---|
| lmfit | end_to_end | 1.2 | 0.725 | 0.0206 | 42 | 0 | 0.0125 | True |
| scipy | end_to_end | 1.57 | 1.38 | 0.134 | 9 | 0 | 0.0125 | True |
| nlsq | end_to_end | 2.06 | 1.65 | 0.0961 | 11 | 0 | 0.0125 | True |
| optimistix | end_to_end | 1.83 | 1.78 | 0.088 | - | 0 | 0.0125 | True |
| optimistix | jitted_core | 1.96 | 1.79 | 0.136 | - | 0 | 0.0125 | True |
| jaxopt | jitted_core | 23.9 | 23 | 0.469 | - | 0 | 0.0125 | False |
| jaxopt | end_to_end | 22.8 | 23.6 | 0.525 | - | 0 | 0.0125 | False |

## Capability matrix

| library | bounds | sigma/weights | covariance | vmappable | maintained | note |
|---|---|---|---|---|---|---|
| nlsq | yes | yes | yes | no | yes |  |
| scipy | yes | yes | yes | no | yes |  |
| lmfit | yes | yes | yes | no | yes | wraps SciPy leastsq / least_squares |
| jaxopt | no | no | no | yes | no | README: "JAXopt is no longer maintained nor developed"; tol/xtol/gtol all honoured |
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
