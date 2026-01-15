#!/bin/bash
# Upload package to production PyPI using token from .env file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if token is set
if [ -z "$PYPI_TOKEN" ]; then
    echo "Error: PYPI_TOKEN not found in .env file"
    echo "Please add PYPI_TOKEN to .env file"
    exit 1
fi

# Upload using environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="$PYPI_TOKEN"

echo "Uploading to PyPI (production)..."
python -m twine upload dist/*

echo "✓ Upload complete!"
