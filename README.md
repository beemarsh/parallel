## Parallel Computing

File: src/logger.cpp
This file simply assings a dedicated rank (0) logging task. Logging is simply blocking send from one rank to this dedicated rank.


File: src/blocking.cpp
Simple blocking send and recieve using MPI_Send and MPI_Recv

File: src/mpi_broadcast.cpp
Sends a simple broadcast using MPI_Bcast

File: src/mpi_reductions.cpp
Run a simple neural network that runs on different nodes/ranks in a cluster.
In this file, you can see gradients being shared via MPI_Allreduce operation.


## Compile
```bash
mkdir ./build
cd build
cmake ..
make
```

## Submit the job in a cluster
```bash
./scripts/submit taskname
For taskname see CMakeLists.txt

eg:
./scripts/submit reduction
```

## Torchtitan training runs
```bash
For debug
./scripts/submit_titan debug 

For llama3 8B training
./scripts/submit_titan llama3_8b
```
