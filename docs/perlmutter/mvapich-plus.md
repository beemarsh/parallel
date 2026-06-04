## MVAPICH Plus 4.1 Installation on PerlMutter
Download the rpm file:
```bash
wget https://mvapich.cse.ohio-state.edu/download/mvapich/plus/private/mvapich-plus-4.1-cuda12.4.rhel9.mofed24.10.ofi.gcc13.2.0.slurm-4.1-1.x86_64.rpm
```

Extract it:
```bash
rpm2cpio path_to_file.rpm | cpio -idmv
```
Export the location. It should match where you installed it
```bash
export MVAPICH_HOME=$SCRATCH/opt/mvapich/plus/4.1/cuda12.4/ofi/slurm/gcc13.2.0
export CC=gcc
export CXX=g++
```

Load the correct modules
```bash
module load PrgEnv-gnu
module load gcc-native/13.2
module load cudatoolkit/12.4
module load craype-accel-nvidia80
module load cray-libsci libfabric xpmem cray-pmi
export PATH=$MVAPICH_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVAPICH_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$MVAPICH_HOME/include:$CPATH


```
Confirm
```bash
ldd $MVAPICH_HOME/lib/libmpi.so | grep -i 'not found'
ldd $MVAPICH_HOME/bin/mpichversion | grep -i 'not found'
mpichversion        # confirm 4.1, ch4:ofi
```

Test
```bash
salloc -N 2 -C gpu -q interactive -t 0:30:00 -A your_account --gpus-per-node=4
export FI_PROVIDER=cxi
OMB=$MVAPICH_HOME/libexec/osu-micro-benchmarks/mpi
srun -N2 --ntasks-per-node=1 $OMB/pt2pt/osu_latency
```

Bandwidth Tests
```bash
srun -N2 --ntasks-per-node=1 --mpi=cray_shasta $OMB/pt2pt/osu_bw -d cuda D D
```

Make sure that PMI is cray_shasta.
You can list them using:
```bash
srun --mpi=list

Output:
	none
	cray_shasta
	pmi2
	pmix
```
For cray_shasta, load:
```bash
module load cray-pmi
```

More about PMI is this paper: https://www.mcs.anl.gov/papers/P1760.pdf

## More fixes
You might get  the following error somewhere:
```
Imported target "MPI::MPI_C" includes non-existent path
  "/opt/mvapich/plus/4.1/cuda12.4/ofi/slurm/gcc13.2.0/include"
```
This is associated with RPMs original install prefix.
We can confirm it via:
```bash
mpicc -show
```
This will give -I/opt/mvapich/... and -L/opt/mvapich/...

We can try to see where we got the old prefixes.
```bash
OLD=/opt/mvapich/plus/4.1/cuda12.4/ofi/slurm/gcc13.2.0
NEW=$MVAPICH_HOME

grep -rl "$OLD" $MVAPICH_HOME/bin $MVAPICH_HOME/lib/pkgconfig $MVAPICH_HOME/lib/*.la 2>/dev/null
```

Patch Them
```bash
sed -i "s|$OLD|$NEW|g" \
  $MVAPICH_HOME/bin/mpicc   $MVAPICH_HOME/bin/mpicxx $MVAPICH_HOME/bin/mpic++ \
  $MVAPICH_HOME/bin/mpif77  $MVAPICH_HOME/bin/mpif90 $MVAPICH_HOME/bin/mpifort \
  $MVAPICH_HOME/bin/mpivars $MVAPICH_HOME/lib/pkgconfig/mvapich-plus.pc \
  $MVAPICH_HOME/lib/*.la
```

Now verify:
```bash
mpicc -show                          # -I/-L should now be under $MVAPICH_HOME
ls $MVAPICH_HOME/include/mpi.h       # sanity: the include dir really exists there
```
