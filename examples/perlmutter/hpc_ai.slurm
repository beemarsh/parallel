#!/bin/bash

# This file should be run after installing mvapich plus and pytorch fork from osu-hidl
# This file runs a simple distributed training 

#SBATCH -C gpu
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=debug
#SBATCH -o output/titan.%j.out
#SBATCH -A account_name

#First source the environment.
# This environment is in env_scripts/perlmutter directory.
source mvapichPlus_installation.sh

# When submitting jobs, the shells are non-interactive. So the conda initialization from .bashrc wont happen
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV_PATH=$PSCRATCH/hpc-ai-build

conda activate $CONDA_ENV_PATH

srun --mpi=cray_shasta python simple_distributed.py
