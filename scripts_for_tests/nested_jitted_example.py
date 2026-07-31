"""
Example script demonstrating Profila profiling nested Numba JIT (@njit) functions.

Hierarchy of nested @njit calls:
  batch_cosine_similarity_search (@njit, parallel=True)
    └── compute_row_similarity (@njit)
          └── vector_dot_product (@njit, fastmath=True)
"""

import time
import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def vector_dot_product(u, v):
    """Leaf JIT kernel: computes dot product of two 1D vectors with SIMD vectorization."""
    s = 0.0
    for i in range(u.shape[0]):
        s += u[i] * v[i]
    return s


@njit(fastmath=True)
def compute_row_similarity(matrix, row_idx, target_vec):
    """Mid-level JIT function: calls leaf JIT kernel vector_dot_product."""
    row = matrix[row_idx]
    dot = vector_dot_product(row, target_vec)
    norm_row = np.sqrt(vector_dot_product(row, row))
    norm_target = np.sqrt(vector_dot_product(target_vec, target_vec))
    if norm_row > 0.0 and norm_target > 0.0:
        return dot / (norm_row * norm_target)
    return 0.0


@njit(fastmath=True, parallel=True)
def batch_cosine_similarity_search(matrix, query_vec):
    """Top-level JIT function: calls mid-level JIT function compute_row_similarity in parallel."""
    n_rows = matrix.shape[0]
    scores = np.zeros(n_rows, dtype=np.float64)
    for i in prange(n_rows):
        scores[i] = compute_row_similarity(matrix, i, query_vec)
    return scores


def main():
    print("Initializing benchmark dataset for nested JIT profiling...")
    np.random.seed(42)
    # 5,000 vectors of dimension 128
    matrix = np.random.randn(5000, 128).astype(np.float64)
    query_vec = np.random.randn(128).astype(np.float64)

    print("Warming up nested Numba JIT compilation...")
    _ = batch_cosine_similarity_search(matrix[:10], query_vec)

    print("Starting profiled benchmark loop for nested JIT functions...")
    start_time = time.time()
    iterations = 0

    while time.time() - start_time < 5.0:
        scores = batch_cosine_similarity_search(matrix, query_vec)
        iterations += 1

    elapsed = time.time() - start_time
    print(f"Benchmark completed {iterations} iterations in {elapsed:.2f} seconds.")
    print(f"Top 3 similarity scores: {np.sort(scores)[-3:]}")


if __name__ == "__main__":
    main()
