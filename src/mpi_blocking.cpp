#include<iostream>
#include<mpi.h>
#include<string>
#include "logger.h"
#include<chrono>
#include<thread>

const int ROOT{0};

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);

  int rank{0}, size{0};

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  init_logger(rank, size);

  if(rank != 0){
    log("Booting up!!!");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    log("Matrix Matrix multiplication complete!!");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    log("Computation Complete");

    shutdown_logger();
  }


  MPI_Finalize();

  return 0;
}

