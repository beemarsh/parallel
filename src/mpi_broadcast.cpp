#include<iostream>
#include<mpi.h>
#include<string>
#include "logger.h"
#include<chrono>
#include<thread>
#include<random>

const int ROOT{0};

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int rank{0}, size{0};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // New communicator: All ranks except loggging rank
  MPI_Comm WORKER_COMM;
  int is_worker = (rank == ROOT) ? MPI_UNDEFINED : 1;
  MPI_Comm_split(MPI_COMM_WORLD, is_worker, rank, &WORKER_COMM);

  init_logger(rank, size);

  if(rank != ROOT){
        int wrank{0}, wsize{0};
        MPI_Comm_rank(WORKER_COMM, &wrank);
        MPI_Comm_size(WORKER_COMM, &wsize);

        log("Booting up!!!");

        // Pick a random broadcaster. However, this needs to run on single rank because every single rank comes up with a random broadcaster.
        // That is not acceptable.
        // Thus, pick a random broadcaster from the worker rank 0.
        // And then broadcast the broadcaster. Wow!!

        int broadcaster{0};
        if (wrank == 0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> distrib(0, wsize - 1);
            broadcaster = distrib(gen);
        }
        MPI_Bcast(&broadcaster, 1, MPI_INT, 0, WORKER_COMM);

        log("[Worker Rank: " + std::to_string(wrank) + " ]:: Got broadcast from Worker Rank " + std::to_string(broadcaster) + " : Decision has been made that I will be the broadcaster" );

        std::string msg;
        int msg_size{0};
        if (wrank == broadcaster) {
            msg = "Hi! Patrolling from Global Rank " + std::to_string(rank) + ", Worker Rank: " + std::to_string(broadcaster);
            msg_size = static_cast<int>(msg.size());
        }

        //First broadcast message size, then the actual content
        MPI_Bcast(&msg_size, 1, MPI_INT, broadcaster, WORKER_COMM);
        if (wrank != broadcaster) msg.resize(msg_size); //Resizing is important
        MPI_Bcast(&msg[0], msg_size, MPI_CHAR, broadcaster, WORKER_COMM);

        log("[Worker Rank: " + std::to_string(wrank) + " ]:: Got broadcast from Worker Rank " + std::to_string(broadcaster) + " : " + msg);

        MPI_Comm_free(&WORKER_COMM);
  }

  shutdown_logger();
  MPI_Finalize();

  return 0;
}
