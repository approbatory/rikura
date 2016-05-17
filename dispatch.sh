#!/bin/bash -l
#SBATCH -p cortex
##SBATCH --job-name=test
##SBATCH --exclusive
#SBATCH -x n0000.cortex0,n0001.cortex0,n0012.cortex0,n0013.cortex0
module load python/anaconda2

module unload intel
python pyscript_to_run.py
