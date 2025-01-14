#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -e

if [ -z "$VIRTUAL_ENV" ]; then
    source ../venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
fi

make train || { echo "Training failed"; exit 1; }

echo "Training completed successfully"
