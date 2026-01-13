#!/bin/bash
#SBATCH --job-name=ccval-preprocess
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

: "${IRIS_VENV_PATH:=~/iris/bin/activate}"
if [ -f "$IRIS_VENV_PATH" ]; then
    # Source the configured virtual environment if it exists
    # Falls back to ~/iris/bin/activate if IRIS_VENV_PATH is not set
    # and tolerates missing files to preserve existing behavior.
    # shellcheck disable=SC1090
    source "$IRIS_VENV_PATH" || true
fi

ccval preprocess workflows/recipes/preprocess_land.yml
