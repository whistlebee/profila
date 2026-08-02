"""
Benchmark script for profiling TutteInstitute/evoc (Numba-accelerated clustering).
"""

import time
from evoc.clustering import build_cluster_layers
from evoc.knn_graph import knn_graph
import numpy as np


def main():
    print("Generating synthetic high-dimensional vector dataset for EVOC...")
    np.random.seed(42)
    # Generate 5,000 samples in 64 dimensions
    data = np.random.randn(5000, 64).astype(np.float32)

    print("Warming up EVOC Numba JIT compilation...")
    warmup_data = data[:200]
    _ = build_cluster_layers(warmup_data, base_min_cluster_size=5)

    print("Starting EVOC Numba profiling workload loop...")
    start_time = time.time()
    iterations = 0

    while time.time() - start_time < 3.5:
        layers = build_cluster_layers(data, base_min_cluster_size=15)
        knn_indices, knn_dists = knn_graph(data, n_neighbors=15)
        iterations += 1

    elapsed = time.time() - start_time
    print(f"EVOC Workload completed {iterations} iterations in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
