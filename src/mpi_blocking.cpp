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

  init_logger(rank, size);

  if(rank != 0){
    log("Booting up!!!");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(1, 100);
    int random_num = distrib(gen);

    // Writing a message
    std::string msg_outgoing = "Hi! Code: " + std::to_string(random_num);
    size_t msg_outgoing_size = msg_outgoing.size();
    char msg_outgoing_buff[msg_outgoing_size];
    msg_outgoing.copy(msg_outgoing_buff, msg_outgoing_size);

    // Arrange the ranks in a circle
    // Send to right
    // Recieve from left
    int dest = rank % (size - 1) + 1;

    MPI_Send(msg_outgoing_buff, msg_outgoing_size, MPI_CHAR, dest, 0, MPI_COMM_WORLD);
    log("Message sent to Rank " + std::to_string(dest));

    //Instead of defining each source, we are just accepting from any source
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    int incoming_msg_len{0};
    MPI_Get_count(&status, MPI_CHAR, &incoming_msg_len);

    std::string incoming_msg;
    incoming_msg.resize(incoming_msg_len);

    MPI_Recv(&incoming_msg[0], incoming_msg_len, MPI_CHAR, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    log("Recieved message from Rank " + std::to_string(status.MPI_SOURCE) + " :: " + incoming_msg);
            
    shutdown_logger();
  }


  MPI_Finalize();

  return 0;
}

