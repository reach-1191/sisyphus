#!/usr/bin/env bash
# ==============================================================================
# Sisyphus AI Coach - Automated Setup Script
# ==============================================================================
# Automates environment preparation, dependency installation, and database initialization.
# Compatible with Linux, macOS, Google Cloud Shell, and GitHub Codespaces.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "       Initializing Sisyphus AI Coach Environment                "
echo "================================================================="

# 1. Determine Python 3 executable
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "[-] Error: Python 3 is required but could not be found."
    echo "    Please install Python 3.10+ and re-run this script."
    exit 1
fi

echo "[1/4] Checking Python environment: $($PYTHON_CMD --version)"

# 2. Create and activate virtual environment (.venv)
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating virtual environment (.venv)..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "[2/4] Virtual environment (.venv) already exists."
fi

# Activate virtual environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3. Upgrade pip and install requirements
echo "[3/4] Upgrading pip and installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# 4. Initialize SQLite database tables
echo "[4/4] Initializing SQLite database tables..."
python -c "from app import app, db, init_db; init_db(); print('  [+] SQLite database tables initialized successfully.')"

echo ""
echo "================================================================="
echo "   Sisyphus AI Coach setup completed successfully!               "
echo "================================================================="
echo ""
echo "To start the application, run:"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open your browser to:"
echo "  http://localhost:8080"
echo "================================================================="
