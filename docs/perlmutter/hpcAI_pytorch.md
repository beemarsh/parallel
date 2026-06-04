## Pytorch installation with MVAPICH Plus

Pull the osu-hidl git repo:
```
git pull --recursive https://github.com/OSU-Nowlab/osu-hidl
cd frameworks/pytorch
```
or pull pytorch from:
```
git clone --recursive https://github.com/OSU-Nowlab/pytorch
```

Get an allocation.
```
salloc -N 1 -C gpu -q interactive -t 04:00:00 -A <your_account> --gpus-per-node=4
```

First load the environment. Its the same as the previous environment in MVAPICH PLus Installation. See the env_scripts.
```bash
source mvapichPlus_install.sh
```

Create a conda environment:
```bash
conda create -p ./hpc-ai-build python=3.11 -y
```

Activate the environment
```bash
conda activate hpc-ai-build
```

Check the environment
```bash
which python               # should be inside the hpc-ai-build env
which mpicc                # must STILL be $MVAPICH_HOME/bin/mpicc
```

Install required packages
```bash
pip install six pyyaml typing-extensions numpy ninja cmake
```

Load modules and unset CONDA variable
This keeps conda's compiler_compat/ld shim and its cross-compile sysconfig out of the way so we link with system gcc 13.2:
```bash
module load nccl
unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "compiler_compat" | paste -sd:)
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"
```

Build
```bash
MAX_JOBS=16 \
USE_XNNPACK=0 \
USE_CUDA=1 \
USE_CUFILE=0 \
USE_MPI=1 \
USE_CUDA_MPI=1 \
USE_DISTRIBUTED=1 \
BUILD_TEST=0 \
TORCH_CUDA_ARCH_LIST="8.0" \
USE_NCCL=1 \ 
USE_SYSTEM_NCCL=1 \
USE_ROCM=0 \
NCCL_INCLUDE_DIR=$NCCL_HOME/include \
NCCL_LIB_DIR=$NCCL_HOME/lib \
python setup.py develop
```
TORCH_CUDA_ARCH_LIST="8.0" for Ampere Arch.

Now check:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch.distributed as dist; print(f'MPI: {dist.is_mpi_available()}')"
```
