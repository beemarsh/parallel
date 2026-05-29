#!/bin/bash

#SBATCH -o output/mpi_job.%j.out    # Name of stdout output file (%j expands to jobId)

module load impi

echo "Running on hosts: $SLURM_JOB_NODELIST"
echo "Total tasks: $SLURM_NTASKS"

ibrun ./build/$TARGET
