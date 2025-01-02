#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=logs.%j.out
#SBATCH --error=logs.%j.err
#SBATCH --time=0-00:30
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e

if [ -z "$VIRTUAL_ENV" ]; then
    source ../../DL_env/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
fi

module load cuda/ || { echo "Failed to load cuda module"; exit 1; }

make train || { echo "Training failed"; exit 1; }

echo "Training completed successfully"
module unload cuda || { echo "Failed to unload cuda module"; exit 1; }