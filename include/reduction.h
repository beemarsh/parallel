#pragma once
#include <ctime>
#include <mpi.h>
#include <string>
#include <vector>

void generate_input(int size, int num_features,
                    std::string filename = "input.bin",
                    std::string label_filename = "labels.bin");

void read_input(std::vector<double> &X, double &Y, int rank,
                int num_features, std::string filename = "input.bin",
                std::string label_filename = "labels.bin");

double relu(double z);
double relu_derivative(double z); 
