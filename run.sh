#!/bin/sh

if [[ "$(uname)" = "Darwin" ]]; then
  # macOS specific env:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi

if [[ -d ".venv" ]]; then
  echo "Activating venv..."
  source .venv/bin/activate
else
  echo "Creating venv..."
  requirements_file="requirements/main.txt"

  # Check if Python is installed
  if ! command -v python; then
    echo "Python not found. Please install Python using your package manager or via PyEnv."
  fi

  python -m venv .venv
  source .venv/bin/activate

  # Check if required packages are up-to-date and update them if not
  pip install --upgrade -r "${requirements_file}"
fi

# Run the main script
python web.py --pycmd python3
