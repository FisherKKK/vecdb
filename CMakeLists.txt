cmake_minimum_required(VERSION 3.10)
project(VectorSearchDemo)

set(CMAKE_CXX_STANDARD 11)

add_executable(vector_search_demo main.cpp hnsw.cpp ivf.cpp)
