# Vector Search Demo

This repository contains a C++ implementation of two popular Approximate Nearest Neighbor (ANN) search algorithms: HNSW (Hierarchical Navigable Small World) and IVF (Inverted File). These algorithms are commonly used in vector search engines for tasks like image retrieval, recommendation systems, and semantic search.

## Algorithms

*   **HNSW (Hierarchical Navigable Small World):** A graph-based algorithm that builds a multi-layer graph of vectors, allowing for efficient searching by navigating the graph from a high level to a more granular level.
*   **IVF (Inverted File):** A clustering-based algorithm that partitions the vector space into a set of clusters and then searches only a subset of these clusters (the "inverted lists") for a given query.

## Getting Started

### Prerequisites

*   A C++ compiler (supporting C++11 or later)
*   CMake (version 3.10 or later)

### Building and Running

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Build the project:**
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

3.  **Run the demo:**
    ```bash
    ./vector_search_demo
    ```

The demo will generate a random dataset of vectors, build both HNSW and IVF indexes, and then perform a search for a random query vector. The results from both algorithms will be printed to the console.

## Code Structure

*   `main.cpp`: The main entry point of the application. It demonstrates how to use the HNSW and IVF implementations.
*   `hnsw.h`, `hnsw.cpp`: The implementation of the HNSW algorithm.
*   `ivf.h`, `ivf.cpp`: The implementation of the IVF algorithm.
*   `CMakeLists.txt`: The build script for the project.
