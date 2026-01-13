#!/bin/bash
#SBATCH --job-name=ccval-preprocess
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load python/3.11 || true
source .venv/bin/activate || true

ccval preprocess workflows/recipes/preprocess_land.yml
