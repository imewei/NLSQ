# PyPI Publishing Setup Guide

## Setting up GitHub Secrets for PyPI Publishing

The publish workflow now supports both API token authentication and trusted publishing. To use API tokens, you need to add the following secrets to your GitHub repository:

### 1. Generate PyPI API Tokens

#### For TestPyPI:
1. Go to https://test.pypi.org/manage/account/token/
2. Create a new API token with scope "Entire account" or project-specific
3. Copy the token (starts with `pypi-`)

#### For PyPI:
1. Go to https://pypi.org/manage/account/token/
2. Create a new API token with scope "Entire account" or project-specific
3. Copy the token (starts with `pypi-`)

### 2. Add Secrets to GitHub Repository

1. Go to your repository on GitHub: https://github.com/imewei/NLSQ
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add the following secrets:

   - **Name**: `TEST_PYPI_API_TOKEN`
     **Value**: Your TestPyPI token (pypi-...)

   - **Name**: `PYPI_API_TOKEN`
     **Value**: Your PyPI token (pypi-...)

### 3. Alternative: Trusted Publishing (Recommended)

For enhanced security, you can set up trusted publishing instead:

#### For TestPyPI:
1. Go to https://test.pypi.org/manage/project/nlsq/settings/publishing/
2. Add a new trusted publisher:
   - Owner: `imewei`
   - Repository: `NLSQ`
   - Workflow name: `publish.yml`
   - Environment: `testpypi`

#### For PyPI:
1. Go to https://pypi.org/manage/project/nlsq/settings/publishing/
2. Add a new trusted publisher:
   - Owner: `imewei`
   - Repository: `NLSQ`
   - Workflow name: `publish.yml`
   - Environment: `pypi`

## Testing the Workflow

### Manual trigger to TestPyPI:
```bash
gh workflow run publish.yml --field test_pypi=true
```

### Create a release (publishes to PyPI):
```bash
git tag v0.1.1
git push origin v0.1.1
gh release create v0.1.1 --title "v0.1.1" --notes "Release notes here"
```

## Verification

After publishing, verify the package:

### TestPyPI:
```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nlsq
```

### PyPI:
```bash
pip install nlsq
```
