# Contributing to yaac

Thank you for your interest in contributing to yaac!

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/withyaac/yaac.git
   cd yaac
   ```

2. **Run the setup script**:
   ```bash
   ./setup/build_env.sh
   ```

   This will:
   - Install uv if not present
   - Create a virtual environment
   - Install all dependencies
   - Set up convenient aliases (`ycd` and `yactivate`)

## Development Tools

The project uses mypy for type checking, ruff for linting, and ruff format for code formatting.

### Type Checking with mypy

```bash
# Check all Python files in the project
mypy yaac/

# Check a specific file
mypy yaac/specific_file.py

# Run with verbose output
mypy -v yaac/

# Faster options:
# Use incremental mode (caches results)
mypy --incremental yaac/

# Use mypy daemon for even faster checks
dmypy start
dmypy run -- yaac/
dmypy stop
```

### Linting with ruff

```bash
# Check all Python files in the project
ruff check yaac/

# Check a specific file
ruff check yaac/specific_file.py

# Auto-fix issues where possible
ruff check --fix yaac/

# Show all available rules
ruff rule list

# Run with verbose output
ruff check -v yaac/
```

### Code Formatting with ruff format

```bash
# Format all Python files in the project
ruff format yaac/

# Format a specific file
ruff format yaac/specific_file.py

# Check formatting without making changes
ruff format --check yaac/

# Show what would be changed without making changes
ruff format --diff yaac/
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests for a specific module
pytest tests/models/sic/

# Run a specific test
pytest tests/models/sic/test_sic.py::test_make_model_shapetype
```

## Editor Integration

You can integrate these tools with your editor:

- **VS Code**: Install the "Python" and "Ruff" extensions
- **PyCharm**: Install the "Mypy" and "Ruff" plugins

## Code Style

- Follow Google Python style guide
- Use type hints for all functions and methods
- Keep line length to 88 characters
- See `general-code-practices.mdc` for detailed guidelines

## Submitting Changes

1. Create a branch for your changes
2. Make your changes with tests
3. Ensure all tests pass: `pytest`
4. Ensure type checking passes: `mypy yaac/`
5. Ensure linting passes: `ruff check yaac/`
6. Submit a pull request
