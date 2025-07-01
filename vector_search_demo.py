

import numpy as np
import faiss

def create_dummy_data(num_vectors, dim):
    """Creates dummy data for the demonstration."""
    return np.random.random((num_vectors, dim)).astype('float32')

def build_hnsw_index(data):
    """Builds an HNSW index."""
    index = faiss.IndexHNSWFlat(data.shape[1], 32)
    index.add(data)
    return index

def build_ivf_index(data):
    """Builds an IVF index."""
    quantizer = faiss.IndexFlatL2(data.shape[1])
    index = faiss.IndexIVFFlat(quantizer, data.shape[1], 100, faiss.METRIC_L2)
    index.train(data)
    index.add(data)
    return index

def search_index(index, query, k):
    """Searches the index and returns the k nearest neighbors."""
    return index.search(query, k)

def main():
    """Main function to demonstrate HNSW and IVF vector search."""
    num_vectors = 10000
    dim = 128
    k = 5

    # 1. Data Generation
    print("Generating dummy data...")
    data = create_dummy_data(num_vectors, dim)
    query_vector = np.random.random((1, dim)).astype('float32')

    # 2. Build HNSW index
    print("Building HNSW index...")
    hnsw_index = build_hnsw_index(data)

    # 3. Build IVF index
    print("Building IVF index...")
    ivf_index = build_ivf_index(data)

    # 4. Vector Search
    print(f"Searching for {k} nearest neighbors...")
    hnsw_distances, hnsw_indices = search_index(hnsw_index, query_vector, k)
    ivf_distances, ivf_indices = search_index(ivf_index, query_vector, k)

    # 5. Output
    print("\nHNSW Search Results:")
    print("Indices:", hnsw_indices)
    print("Distances:", hnsw_distances)

    print("\nIVF Search Results:")
    print("Indices:", ivf_indices)
    print("Distances:", ivf_distances)

if __name__ == "__main__":
    main()

