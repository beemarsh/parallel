#include "logger.h"
#include "reduction.h"
#include <cmath>
#include <ctime>
#include <mpi.h>
#include <string>
#include <vector>

// This is a simple neural network running 100 epochs.
// There is a dedicated logger in rank 0.
// There is sub-communicator for the training.
// It generates a random data in input.bin and labels in label.bin file
// It has two hidden layers.
// After back-propagation, all ranks share their gradients using Allreduce
// For simplicity, same data is used in all epochs.

const int ROOT{0};
int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    log("Please run this script with at least 2 MPI processes for this ");
    shutdown_logger();
    MPI_Finalize();
    return 1;
  }

  int num_features = 20;
  int num_hidden_layers = 2;
  std::vector<int> num_neurons = {15, 10};
  double learning_rate = 0.01;
  int num_epochs = 100;

  // New communicator: All ranks except loggging rank
  MPI_Comm WORKER_COMM;
  int is_worker = (rank == ROOT) ? MPI_UNDEFINED : 1;
  MPI_Comm_split(MPI_COMM_WORLD, is_worker, rank, &WORKER_COMM);

  init_logger(rank, size);

  if (rank != ROOT) {
    int wrank{0}, wsize{0};
    MPI_Comm_rank(WORKER_COMM, &wrank);
    MPI_Comm_size(WORKER_COMM, &wsize);

    log("[WRANK: " + std::to_string(wrank) + "]:: Booting up!!!");

    // Worker Rank 0 will generate an input file.
    if (wrank == 0) {
      generate_input(wsize, num_features);
      log("Generated Input");
    }
    MPI_Barrier(
        WORKER_COMM); // Every rank waits till the input file is generated

    // Local X
    std::vector<double> X(num_features);
    double Y;
    read_input(X, Y, wrank, num_features);

    std::string x_str = "[ ";
    for (const auto &num : X) {
      x_str += std::to_string(num) + " ";
    }
    x_str += " ],  True Label: " + std::to_string(Y);
    log("Read Input, Input: " + x_str);

    // Initialize weights to 0.1 and bias to 0
    // Hidden Layer 1 (15 neurons looking at 20 features)
    std::vector<std::vector<double>> W1(num_neurons[0],
                                        std::vector<double>(num_features, 0.1));
    std::vector<std::vector<double>> dW1(
        num_neurons[0], std::vector<double>(num_features, 0.0));
    std::vector<double> b1(num_neurons[0], 0.0);
    std::vector<double> db1(num_neurons[0], 0.0);

    // Hidden Layer 2 (5 neurons looking at 10 features)
    std::vector<std::vector<double>> W2(
        num_neurons[1], std::vector<double>(num_neurons[0], 0.1));
    std::vector<std::vector<double>> dW2(
        num_neurons[1], std::vector<double>(num_neurons[0], 0.0));
    std::vector<double> b2(num_neurons[1], 0.0);
    std::vector<double> db2(num_neurons[1], 0.0);

    std::vector<double> W3(num_neurons[1], 0.2);
    std::vector<double> dW3(num_neurons[1], 0.0);
    double b3 = 0.0;
    double db3{0};

    double dL_dyhat{0};
    double loss{0};

    // Calculate the total number of parameters
    int total_params =
        (num_neurons[0] * num_features) + num_neurons[0] +   // W1 + b1
        (num_neurons[1] * num_neurons[0]) + num_neurons[1] + // W2 + b2
        num_neurons[1] + 1;                                  // W3 + b3
    std::vector<double> local_grads(total_params, 0.0);
    std::vector<double> global_grads(total_params, 0.0);

    // Run Epochs
    for (int ep{0}; ep < num_epochs; ep++) {
      std::vector<double> z1(num_neurons[0], 0.0);
      std::vector<double> a1(num_neurons[0], 0.0);
      std::vector<double> z2(num_neurons[1], 0.0);
      std::vector<double> a2(num_neurons[1], 0.0);
      double y_hat = b3; // Reset y_hat to the current bias!

      log("Starting Epoch" + std::to_string(ep));
      // Forward Pass, first layer
      for (int i = 0; i < num_neurons[0]; ++i) {
        for (int j = 0; j < num_features; ++j) {
          z1[i] += W1[i][j] * X[j];
        }
        z1[i] += b1[i];
        a1[i] = relu(z1[i]);
      }
      log("First Layer Forward Pass Complete!!");

      // Hidden Layer 2 (5 neurons looking at 10 outputs from Layer 1)
      for (int i = 0; i < num_neurons[1]; ++i) {
        for (int j = 0; j < num_neurons[0]; ++j) {
          z2[i] += W2[i][j] * a1[j];
        }
        z2[i] += b2[i];
        a2[i] = relu(z2[i]);
      }
      log("Second Layer Forward Pass Complete!!");

      // Output Layer (1 final output looking at 5 outputs from Layer 2)
      for (int i = 0; i < num_neurons[1]; ++i) {
        y_hat += W3[i] * a2[i];
      }
      log("Forward Pass Complete!!");

      loss = 0.5 * std::pow(y_hat - Y, 2);

      log("Prediction: " + std::to_string(y_hat) +
          " | Loss: " + std::to_string(loss));

      std::vector<double> dz1(num_neurons[0], 0.0);
      std::vector<double> dz2(num_neurons[1], 0.0);
      // Backward Pass
      log("Starting Backward Pass");

      // Output Layer Gradient
      dL_dyhat = y_hat - Y;
      db3 = dL_dyhat;
      for (int i = 0; i < num_neurons[1];
           ++i) { // Gradient for Output Weights: Output error * incoming
                  // activation from
        dW3[i] = dL_dyhat * a2[i];
      }

      // Push the error backward to Layer 2
      for (int i = 0; i < num_neurons[1]; ++i) {
        dz2[i] = dL_dyhat * W3[i] * relu_derivative(z2[i]);
      }
      log("Pushed Error Backward to Layer 2");

      // Second Hidden Layer Gradients (W2 and b2)

      for (int i = 0; i < num_neurons[1]; ++i) {
        db2[i] = dz2[i]; // Bias gradient is just the delta

        for (int j = 0; j < num_neurons[0]; ++j) {
          // Gradient for W2: Layer 2 error * incoming activation from Layer 1
          dW2[i][j] = dz2[i] * a1[j];
          // Accumulate the error flowing backward into Layer 1 across W2
          dz1[j] += dz2[i] * W2[i][j];
        }
      }

      for (int j = 0; j < num_neurons[0]; ++j) {
        dz1[j] *= relu_derivative(z1[j]);
      }
      log("Pushed Error backward to Layer 1");

      // First Hidden Layer Gradients (W1 and b1)
      for (int i = 0; i < num_neurons[0]; ++i) {
        db1[i] = dz1[i]; // Bias gradient is just the delta

        for (int j = 0; j < num_features; ++j) {
          // Gradient for W1: Layer 1 error * incoming raw input X
          dW1[i][j] = dz1[i] * X[j];
        }
      }

      log("Backward Pass Complete!!");

      log("Flattening local gradients for MPI synchronization...");
      // FLATTEN all our 2D and 1D gradients into a single 1D array sequentially
      int index = 0;

      for (int i = 0; i < num_neurons[0]; ++i) {
        for (int j = 0; j < num_features; ++j) {
          local_grads[index++] = dW1[i][j];
        }
        local_grads[index++] = db1[i];
      }
      for (int i = 0; i < num_neurons[1]; ++i) {
        for (int j = 0; j < num_neurons[0]; ++j) {
          local_grads[index++] = dW2[i][j];
        }
        local_grads[index++] = db2[i];
      }
      for (int i = 0; i < num_neurons[1]; ++i) {
        local_grads[index++] = dW3[i];
      }
      local_grads[index++] = db3;

      // Sum all local_grads across all workers into global_grads simultaneously
      MPI_Allreduce(local_grads.data(), global_grads.data(), total_params,
                    MPI_DOUBLE, MPI_SUM, WORKER_COMM);
      log("MPI Allreduce complete. Applying Gradient Descent...");

      // AVERAGE AND UPDATE
      index = 0;

      // Update W1 and b1
      for (int i = 0; i < num_neurons[0]; ++i) {
        for (int j = 0; j < num_features; ++j) {
          double avg_dW1 = global_grads[index++] / wsize;
          W1[i][j] -= learning_rate * avg_dW1;
        }
        double avg_db1 = global_grads[index++] / wsize;
        b1[i] -= learning_rate * avg_db1;
      }

      // Update W2 and b2
      for (int i = 0; i < num_neurons[1]; ++i) {
        for (int j = 0; j < num_neurons[0]; ++j) {
          double avg_dW2 = global_grads[index++] / wsize;
          W2[i][j] -= learning_rate * avg_dW2;
        }
        double avg_db2 = global_grads[index++] / wsize;
        b2[i] -= learning_rate * avg_db2;
      }

      // Update W3 and b3
      for (int i = 0; i < num_neurons[1]; ++i) {
        double avg_dW3 = global_grads[index++] / wsize;
        W3[i] -= learning_rate * avg_dW3;
      }
      double avg_db3 = global_grads[index] / wsize;
      b3 -= learning_rate * avg_db3;

      log("Global Weights Successfully Synchronized and Updated!");
      log("Epoch: " + std::to_string(ep) + " Complete");
    }

    MPI_Comm_free(&WORKER_COMM);
  }

  shutdown_logger();
  MPI_Finalize();
  return 0;
}
