#!/bin/bash

#SBATCH --output=TRavailGPU%j.out
#SBATCH --error=TRavailGPU%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpu:1
#SBATCH --cpus-per-task=3

set -x
srun python -u script.py
