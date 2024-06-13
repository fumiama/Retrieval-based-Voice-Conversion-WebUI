#!/bin/sh

set -fa

# Check if Python is installed
if ! command -v python; then
  echo "Python not found. Please install Python using your package manager or via PyEnv."
  exit 1
fi

requirements_file="requirements/main.txt"
venv_path=".venv"

if [[ ! -d "${venv_path}" ]]; then
  echo "Creating venv..."

  python -m venv "${venv_path}"
  source "${venv_path}/bin/activate"

  # Check if required packages are up-to-date
  pip install --upgrade -r "${requirements_file}"
fi
echo "Activating venv..."
source "${venv_path}/bin/activate"

# Run the main script
python web.py --pycmd python
