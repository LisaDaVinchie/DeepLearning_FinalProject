#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:00:10
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1

set -e

if [ -z "$VIRTUAL_ENV" ]; then
    source ../../DL_env/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
fi


make train || { echo "Training failed"; exit 1; }

echo "Training completed successfully"
