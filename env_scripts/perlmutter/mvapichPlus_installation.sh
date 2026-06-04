#!/bin/bash

## MVAPICH Plus 4.1 Installation on PerlMutter Env File
## Usage: source this file
## Follow the instructions at docs/perlmutter

# Export the location. It should match where you installed it
export MVAPICH_HOME=$SCRATCH/opt/mvapich/plus/4.1/cuda12.4/ofi/slurm/gcc13.2.0

export CC=gcc
export CXX=g++

module load PrgEnv-gnu
module load gcc-native/13.2
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load cray-libsci libfabric xpmem cray-pmi
export PATH=$MVAPICH_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVAPICH_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$MVAPICH_HOME/include:$CPATH
