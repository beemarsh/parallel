cmake_minimum_required(VERSION 3.16)
project(parallel LANGUAGES CXX)

find_package(MPI REQUIRED)

include_directories(include)

add_library(logger src/logger.cpp)
target_link_libraries(logger PUBLIC MPI::MPI_CXX)

add_executable(blocking src/mpi_blocking.cpp)
target_link_libraries(blocking PRIVATE logger MPI::MPI_CXX)

add_executable(collectives src/mpi_broadcast.cpp)
target_link_libraries(collectives PRIVATE logger MPI::MPI_CXX)
