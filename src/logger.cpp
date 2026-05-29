#include "logger.h"
#include<mpi.h>
#include<queue>
#include<iostream>

// This is  a simple logger which runs on a ROOT rank.
// The ROOT rank simply Probes for any incoming messages.
// If it gets any, it calls the blocking MPI_Recv and prints the message.
// The printed message is handled by slurm into its output file.

// Other ranks simply do a blocking send of their message

static int curr_rank{-1};
static int world_size{-1};

const int TAG_LOG=0;
const int TAG_QUIT=1;

std::queue<std::string>logs;  //Temporarily queued logs

void init_logger(int rank, int size) {
    curr_rank = rank;
    world_size=size;

    //Run the logger only on ROOT
    if (curr_rank != ROOT) return;
    
    std::cout << "[RANK " << ROOT << "] === LOGGER STARTED ===\n";

    int active_workers = world_size - 1;

    while(active_workers > 0){
      MPI_Status status;

      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == TAG_LOG) {
            int msg_len{0};
            MPI_Get_count(&status, MPI_CHAR, &msg_len);
            std::string incoming_msg;
            incoming_msg.resize(msg_len);

            MPI_Recv(&incoming_msg[0], msg_len, MPI_CHAR, status.MPI_SOURCE, TAG_LOG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "[RANK " << status.MPI_SOURCE << "]: " << incoming_msg << '\n';
            
        }

      else if (status.MPI_TAG == TAG_QUIT) {
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, TAG_QUIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            active_workers--;
        }
    }
}

void log(const std::string &msg){
  size_t msg_len = msg.size();
  char buf[msg_len];
  msg.copy(buf,msg_len);

  //Send msg to logging rank
  MPI_Send(buf, msg.size(), MPI_CHAR, ROOT, TAG_LOG, MPI_COMM_WORLD);
  
  return;
}

void shutdown_logger() {
    if (curr_rank != ROOT) {
        int dummy = 0;
        // Send the poison pill to Rank 0
        MPI_Send(&dummy, 1, MPI_INT, ROOT, TAG_QUIT, MPI_COMM_WORLD);
    }
}
