#!/bin/bash
# Upload package to TestPyPI using token from .env file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if token is set
if [ -z "$TESTPYPI_TOKEN" ]; then
    echo "Error: TESTPYPI_TOKEN not found in .env file"
    echo "Please create .env file with: TESTPYPI_TOKEN=pypi-..."
    exit 1
fi

# Upload using environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="$TESTPYPI_TOKEN"

echo "Uploading to TestPyPI..."
python -m twine upload --repository testpypi dist/*

echo "✓ Upload complete!"
