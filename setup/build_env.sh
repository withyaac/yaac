#!/bin/bash

# Exit on error
set -e

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "Setting up yaac development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install/update dependencies
echo "Installing dependencies..."
uv pip install -e ".[dev]"

# Add aliases to bashrc if they don't exist
BASHRC="$HOME/.bashrc"
ALIASES=(
    "alias ycd='cd $PROJECT_DIR'"
    "alias yactivate=\"source $VENV_DIR/bin/activate\""
)

for alias in "${ALIASES[@]}"; do
    if ! grep -q "$alias" "$BASHRC"; then
        echo "Adding alias to $BASHRC..."
        echo "$alias" >> "$BASHRC"
    fi
done

echo "Setup complete! 🎉"
echo "You can now use:"
echo "  - ycd to navigate to the project"
echo "  - yactivate to activate the virtual environment"
echo ""
echo "Please restart your terminal or run 'source ~/.bashrc' to use the new aliases." 