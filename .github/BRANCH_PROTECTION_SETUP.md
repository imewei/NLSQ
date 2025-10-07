# Branch Protection Setup Guide

## ⚠️ MANUAL SETUP REQUIRED

Branch protection rules cannot be configured via files - they must be set up through GitHub's web interface or API.

## Recommended Settings

### For `main` branch:

1. **Go to**: Repository Settings → Branches → Branch protection rules → Add rule

2. **Branch name pattern**: `main`

3. **Required settings**:

   #### Protect matching branches
   - ✅ **Require a pull request before merging**
     - ✅ Require approvals: 1
     - ✅ Dismiss stale pull request approvals when new commits are pushed
     - ✅ Require review from Code Owners

   - ✅ **Require status checks to pass before merging**
     - ✅ Require branches to be up to date before merging
     - Required status checks (select these):
       - `pre-commit`
       - `test (ubuntu-latest, 3.12, fast)`
       - `test (ubuntu-latest, 3.12, slow)`
       - `package`
       - `check-status`

   - ✅ **Require conversation resolution before merging**

   - ✅ **Require signed commits** (recommended for security)

   - ✅ **Require linear history** (optional, for clean git history)

   - ✅ **Do not allow bypassing the above settings**
     - Exception: Allow administrators to bypass (for emergency fixes)

   - ✅ **Restrict who can push to matching branches**
     - Add: Repository maintainers only

   - ✅ **Allow force pushes**: DISABLED

   - ✅ **Allow deletions**: DISABLED

4. **Click "Create"** or **"Save changes"**

---

## Additional Recommendations

### For `develop` branch (if you create one):

Same settings as `main`, but:
- Require approvals: 0 (allow self-merge for faster iteration)
- Allow administrators to bypass: YES

---

## Verification

After setup, verify by:

1. Try pushing directly to `main` - should be blocked
2. Create a test PR - should require status checks
3. Check that Dependabot can create PRs - should work with automated reviews

---

## Security Enhancements

### Enable Repository Security Features:

**Settings → Security & analysis**:
- ✅ Dependency graph: Enabled
- ✅ Dependabot alerts: Enabled
- ✅ Dependabot security updates: Enabled
- ✅ Dependabot version updates: Enabled (after adding `.github/dependabot.yml`)
- ✅ Secret scanning: Enabled
- ✅ Secret scanning push protection: Enabled

**Settings → Code security and analysis**:
- ✅ CodeQL analysis: Enable for Python
- ✅ Private vulnerability reporting: Enabled

---

## Current Status

- [ ] Branch protection for `main` configured
- [ ] Branch protection for `develop` configured (if applicable)
- [x] Dependabot configuration added
- [x] CODEOWNERS file added
- [ ] Security features enabled
- [ ] CodeQL analysis configured

---

## Quick Setup Script (via GitHub CLI)

If you have `gh` CLI installed:

```bash
# Enable branch protection for main
gh api repos/imewei/NLSQ/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["pre-commit","test (ubuntu-latest, 3.12, fast)","test (ubuntu-latest, 3.12, slow)","package","check-status"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true,"require_code_owner_reviews":true}' \
  --field restrictions=null \
  --field required_linear_history=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field required_conversation_resolution=true

# Enable security features
gh api repos/imewei/NLSQ \
  --method PATCH \
  --field has_vulnerability_alerts=true \
  --field security_and_analysis='{"secret_scanning":{"status":"enabled"},"secret_scanning_push_protection":{"status":"enabled"},"dependabot_security_updates":{"status":"enabled"}}'
```

---

**Last Updated**: 2025-10-07
**Maintainer**: Wei Chen (wchen@anl.gov)
