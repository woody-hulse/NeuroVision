#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=1:00:00
#SBATCH --job-name=train_job
#SBATCH --output=train.out
#SBATCH --error=train.err

module load python3

python3 train.py
