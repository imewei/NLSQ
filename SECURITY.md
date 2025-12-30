# Security Policy

## Supported Versions

We provide security updates for the following versions of NLSQ:

| Version | Supported          |
| ------- | ------------------ |
| 0.x (Beta) | :white_check_mark: |

As NLSQ is currently in Beta, we recommend always using the latest version from the `main` branch.

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities via one of the following methods:

### 1. GitHub Security Advisories (Preferred)

Use GitHub's private vulnerability reporting feature:

1. Go to: https://github.com/imewei/NLSQ/security/advisories/new
2. Click "Report a vulnerability"
3. Fill in the details

### 2. Direct Email

Email the maintainer directly:
- **Email**: wchen@anl.gov
- **Subject**: [SECURITY] NLSQ Vulnerability Report
- **PGP Key**: Available upon request

### What to Include

Please include as much of the following information as possible:

- **Type of vulnerability** (e.g., buffer overflow, SQL injection, XSS, etc.)
- **Full paths** of source files related to the vulnerability
- **Location** of the affected source code (tag/branch/commit or URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if available)
- **Impact** of the vulnerability (what an attacker could do)
- **Potential mitigations** (if you have suggestions)

---

## Response Timeline

- **Initial Response**: Within 48 hours
- **Triage**: Within 1 week
- **Fix & Disclosure**: Coordinated with reporter, typically within 30 days

---

## Security Update Process

1. **Acknowledgment**: We'll confirm receipt of your report
2. **Validation**: We'll reproduce and validate the vulnerability
3. **Fix Development**: We'll develop and test a fix
4. **Advisory**: We'll create a security advisory (if appropriate)
5. **Release**: We'll release a patched version
6. **Disclosure**: We'll publicly disclose the vulnerability (coordinated with you)

---

## Security Best Practices for NLSQ Users

### Input Validation

NLSQ is designed for scientific computing with **trusted data**. If you're using NLSQ with untrusted input:

- ✅ **Validate all inputs** before passing to `curve_fit()`
- ✅ **Sanitize data arrays** to prevent injection attacks
- ✅ **Set reasonable bounds** on parameters to prevent resource exhaustion
- ✅ **Use memory limits** via `memory_limit_gb` parameter
- ✅ **Validate fit functions** - don't use `eval()` or `exec()` with untrusted code

### Example: Safe Usage with Untrusted Data

```python
import nlsq
import numpy as np

# ❌ UNSAFE: Direct use of untrusted data
# data = load_untrusted_data()
# popt, _ = nlsq.curve_fit(fit_func, data['x'], data['y'])


# ✅ SAFE: Validate inputs first
def safe_curve_fit(fit_func, xdata, ydata, **kwargs):
    # Validate input types
    if not isinstance(xdata, np.ndarray):
        xdata = np.array(xdata, dtype=np.float64)
    if not isinstance(ydata, np.ndarray):
        ydata = np.array(ydata, dtype=np.float64)

    # Validate shapes
    if xdata.ndim != 1 or ydata.ndim != 1:
        raise ValueError("Data must be 1D arrays")
    if len(xdata) != len(ydata):
        raise ValueError("x and y must have same length")

    # Validate size (prevent DoS)
    if len(xdata) > 1_000_000:
        raise ValueError("Dataset too large (max 1M points)")

    # Validate no NaN/Inf
    if not np.all(np.isfinite(xdata)) or not np.all(np.isfinite(ydata)):
        raise ValueError("Data contains NaN or Inf values")

    # Set memory limit to prevent resource exhaustion
    return nlsq.curve_fit(fit_func, xdata, ydata, memory_limit_gb=2.0, **kwargs)
```

### Dependency Security

NLSQ relies on JAX, NumPy, and SciPy. Keep these dependencies updated:

⚠️ **Important**: NLSQ requires **NumPy 2.0+** (breaking change from NumPy 1.x)

```bash
# Check for known vulnerabilities
pip install pip-audit
pip-audit

# Update dependencies (ensure NumPy 2.0+, JAX 0.6.0+)
pip install --upgrade nlsq "jax>=0.6.0" "jaxlib>=0.6.0" "numpy>=2.0.0" "scipy>=1.14.0"

# Verify versions
python -c "import numpy; import jax; print(f'NumPy: {numpy.__version__}, JAX: {jax.__version__}')"
```

For detailed dependency requirements and version management, see [REQUIREMENTS.md](REQUIREMENTS.md).

---

## Known Dependency Vulnerabilities

This section tracks known vulnerabilities in NLSQ's dependencies that cannot be immediately fixed due to lack of upstream patches.

### CVE-2025-53000: nbconvert (Windows only)

| Field | Value |
|-------|-------|
| **CVE** | [CVE-2025-53000](https://github.com/advisories/GHSA-xm59-rqc7-hhvf) |
| **Severity** | HIGH (CVSS 8.5) |
| **Affected Package** | `nbconvert` (all versions up to 7.16.6) |
| **Dependency Chain** | `nlsq[jupyter]` → `jupyterlab` → `jupyter_server` → `nbconvert` |
| **Upstream Issue** | [jupyter/nbconvert#2258](https://github.com/jupyter/nbconvert/issues/2258) |
| **Fix Available** | No (as of 2025-12-29) |

**Description**: Uncontrolled search path vulnerability on Windows. When running `jupyter nbconvert --to pdf` on notebooks containing SVG output, a malicious `inkscape.bat` file in the current directory could be executed.

**Impact on NLSQ**:
- Only affects Windows users
- Only triggers when converting notebooks to PDF with SVG content
- Does not affect core curve fitting functionality
- Is in an optional dependency (`jupyter` extra)

**Mitigation**: Avoid running `jupyter nbconvert --to pdf` from untrusted directories on Windows until upstream releases a patch.

**Status**: Waiting for upstream fix. This entry will be removed once `nbconvert` releases a patched version.

---

## Known Security Considerations

### 1. Arbitrary Code Execution via Fit Functions

⚠️ **NLSQ uses JAX JIT compilation**, which executes the fit function you provide.

- **Risk**: If you allow untrusted users to define fit functions, they could execute arbitrary code
- **Mitigation**: Only use fit functions from trusted sources
- **Example Attack**: `fit_func = lambda x, a: os.system('rm -rf /')` (DON'T DO THIS!)

### 2. Resource Exhaustion (DoS)

⚠️ **Large datasets or complex fit functions** can consume significant memory/CPU.

- **Risk**: Malicious inputs could cause out-of-memory errors or CPU exhaustion
- **Mitigation**: Use `memory_limit_gb` parameter and validate input sizes
- **Example**: Fitting 1 billion points could exhaust system memory

### 3. Numerical Stability Issues

⚠️ **Poorly conditioned problems** can cause numerical overflow/underflow.

- **Risk**: Could lead to NaN/Inf propagation or infinite loops
- **Mitigation**: NLSQ includes automatic stability checks and recovery mechanisms
- **Example**: Fitting with extreme parameter scales (1e-300 to 1e300)

---

## Security Scanning

NLSQ uses automated security scanning:

- ✅ **Bandit**: Python security linter (runs in CI)
- ✅ **pip-audit**: Dependency vulnerability scanner
- ✅ **Safety**: Checks against known vulnerabilities
- ✅ **CodeQL**: Advanced semantic code analysis
- ✅ **Dependabot**: Automated dependency updates

View security scan results:
- **Security tab**: https://github.com/imewei/NLSQ/security
- **Code scanning**: https://github.com/imewei/NLSQ/security/code-scanning

---

## Responsible Disclosure

We follow responsible disclosure practices:

1. **Private reporting period**: 90 days (or coordinated with reporter)
2. **Fix development**: Priority assigned based on severity
3. **Credit**: We'll credit researchers who report vulnerabilities (if desired)
4. **CVE assignment**: We'll request CVEs for critical vulnerabilities
5. **Public disclosure**: After fix is released, we'll publish advisory

---

## Security Hall of Fame

We appreciate security researchers who help keep NLSQ secure:

<!-- Future researchers will be listed here -->

---

## Contact

- **Maintainer**: Wei Chen (Argonne National Laboratory)
- **Email**: wchen@anl.gov
- **GitHub**: @imewei

---

**Last Updated**: 2025-12-29
