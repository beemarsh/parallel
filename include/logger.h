#pragma once
#include<string>

extern const int ROOT;
void init_logger(int rank, int size);

void log(const std::string &msg);

void shutdown_logger();
