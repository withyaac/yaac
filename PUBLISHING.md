# Publishing yaac to PyPI

This document describes how to publish the `yaac` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **TestPyPI Account** (recommended for testing): Create an account at https://test.pypi.org/account/register/
3. **Build Tools**: Install build tools:
   ```bash
   pip install build twine
   ```

## Pre-Publishing Checklist

- [x] Build system configured in `pyproject.toml`
- [x] Package metadata (version, description, classifiers) added
- [x] Package exports defined in `__init__.py` files
- [x] `.gitignore` excludes build artifacts (`build/`, `dist/`, `*.egg-info/`)
- [x] LICENSE file present
- [x] README.md present and informative
- [ ] Tests pass
- [ ] Version number updated (if needed)

## Building the Package

1. **Clean previous builds** (if any):
   ```bash
   rm -rf build/ dist/ *.egg-info/
   ```

2. **Build the package**:
   ```bash
   python -m build
   ```

   This creates:
   - `dist/yaac-0.1.0.tar.gz` (source distribution)
   - `dist/yaac-0.1.0-py3-none-any.whl` (wheel distribution)

3. **Verify the build**:
   ```bash
   # Check the built files
   ls -lh dist/
   
   # Verify the package contents
   tar -tzf dist/yaac-0.1.0.tar.gz | head -20
   ```

## Testing on TestPyPI (Recommended)

Before publishing to production PyPI, test on TestPyPI:

1. **Upload to TestPyPI**:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

   You'll be prompted for:
   - Username: `__token__`
   - Password: Your TestPyPI API token (create at https://test.pypi.org/manage/account/token/)

2. **Test installation from TestPyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ yaac
   ```

   Note: The `--extra-index-url` is needed because TestPyPI doesn't mirror all dependencies.

3. **Verify the installation**:
   ```bash
   python -c "import yaac; print(yaac.__version__)"
   python -c "from yaac import TrainableModel, load_model_from_checkpoint; print('Imports work!')"
   ```

## Publishing to Production PyPI

Once you've tested on TestPyPI:

1. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```

   You'll be prompted for:
   - Username: `__token__`
   - Password: Your PyPI API token (create at https://pypi.org/manage/account/token/)

2. **Verify the publication**:
   - Check https://pypi.org/project/yaac/
   - Test installation: `pip install yaac`

## Updating the Package

When you need to publish a new version:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # or "0.2.0", etc.
   ```

2. **Update `__version__`** in `yaac/__init__.py`:
   ```python
   __version__ = "0.1.1"
   ```

3. **Follow the build and upload steps above**

## API Token Setup

For security, use API tokens instead of passwords:

1. Go to https://pypi.org/manage/account/token/ (or TestPyPI equivalent)
2. Create a new API token with scope "Entire account" or "Project: yaac"
3. Copy the token (starts with `pypi-`)
4. Use `__token__` as username and the token as password when uploading

## Troubleshooting

### "Package already exists" error
- The version number must be incremented for each upload
- PyPI doesn't allow overwriting existing versions

### "Invalid distribution" error
- Make sure you're uploading both `.tar.gz` and `.whl` files
- Verify the build completed successfully

### Import errors after installation
- Check that `__init__.py` files properly export the public API
- Verify package structure matches what's in `pyproject.toml`

## Package Structure

The package exports:

- **Top-level**: `TrainableModel`, `load_model_from_checkpoint`
- **Models**: `yaac.models.sic.SIC`, `yaac.models.sic.make_model`
- **Common**: `yaac.common.trainable_model.TrainableModel`, `yaac.common.model_loader.load_model_from_checkpoint`

Users can import like:
```python
from yaac import TrainableModel, load_model_from_checkpoint
from yaac.models.sic import SIC, make_model
```
