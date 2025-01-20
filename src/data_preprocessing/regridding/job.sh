#!/bin/bash
#SBATCH --job-name=interpolation
#SBATCH --output=output.out
#SBATCH --error=errors.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1

if [ -z "$VIRTUAL_ENV" ]; then
	source /u/ldavinchie/thesis_venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
	echo "Virtual environmnt activated"
else
	echo "Virtual environment already activated"
fi


make run || { echo "Makefile execution failed"; exit 1; }