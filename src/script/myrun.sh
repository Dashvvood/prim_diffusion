#!/bin/bash

#SBATCH --output=TRavailGPU%j.out
#SBATCH --error=TRavailGPU%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpu:1
#SBATCH --cpus-per-task=4

set -x
srun python -u $*
