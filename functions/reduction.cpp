#include <ctime>
#include <fstream>
#include <mpi.h>
#include <string>
#include <vector>

void generate_input(int size, int num_features,
                    std::string filename = "input.bin",
                    std::string label_filename = "labels.bin") {
  std::srand(std::time(nullptr));
  std::vector<double> global_x(size * num_features);
  std::vector<double> global_y(size);

  // Fill with random data
  for (int i = 0; i < size * num_features; ++i) {
    global_x[i] = static_cast<double>(std::rand()) / RAND_MAX;
  }
  for (int i = 0; i < size; ++i) {
    global_y[i] = (static_cast<double>(std::rand()) / RAND_MAX) * 10.0;
  }
  std::ofstream x_file(filename, std::ios::binary);
  std::ofstream y_file(label_filename, std::ios::binary);

  if (x_file.is_open() && y_file.is_open()) {
    x_file.write(reinterpret_cast<char *>(global_x.data()),
                 global_x.size() * sizeof(double));
    y_file.write(reinterpret_cast<char *>(global_y.data()),
                 global_y.size() * sizeof(double));
    x_file.close();
    y_file.close();
  }
  return;
}

void read_input(std::vector<double> &X, double &Y, int rank,
                int num_features, std::string filename = "input.bin",
                std::string label_filename = "labels.bin") {
  // Calculate exact byte offsets for this specific rank
  std::streampos x_offset = rank * num_features * sizeof(double);
  std::streampos y_offset = rank * sizeof(double);

  std::ifstream x_infile(filename, std::ios::binary);
  std::ifstream y_infile(label_filename, std::ios::binary);

  if (x_infile.is_open() && y_infile.is_open()) {
    // Jump to offset and read X
    x_infile.seekg(x_offset);
    x_infile.read(reinterpret_cast<char *>(X.data()),
                  num_features * sizeof(double));

    // Jump to offset and read y
    y_infile.seekg(y_offset);
    y_infile.read(reinterpret_cast<char *>(&Y), sizeof(double));

    x_infile.close();
    y_infile.close();
  }
  return;
}

// ReLU
double relu(double z) { return std::max(0.0, z); }

// Helper for ReLU Derivative
double relu_derivative(double z) { return (z > 0.0) ? 1.0 : 0.0; }
