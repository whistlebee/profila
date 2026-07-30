"""
Realistic multi-stage Numba benchmark script for line-level profiling and flamegraphs.

Contains 4 distinct computational workloads:
1. Matrix Multiplication (L1/L2 cache & loop unrolling)
2. Pairwise Euclidean Distance Matrix (N^2 distance computations)
3. 2D Stencil Heat Diffusion (Grid neighborhood update)
4. Exponential Moving Average & Signal Filter (Sequential recurrence)
"""

import time
import numpy as np
from numba import njit, float64, int64


@njit
def matrix_multiply(A, B):
    """3D nested loop matrix multiplication (O(N^3))."""
    N, K = A.shape
    K2, M = B.shape
    assert K == K2
    C = np.zeros((N, M), dtype=float64)
    for i in range(N):
        for k in range(K):
            r = A[i, k]
            for j in range(M):
                C[i, j] += r * B[k, j]
    return C


@njit
def pairwise_distances(X):
    """Pairwise Euclidean distance matrix between N points in D dimensions."""
    N, D = X.shape
    dist = np.zeros((N, N), dtype=float64)
    for i in range(N):
        for j in range(i + 1, N):
            # Compute squared Euclidean distance
            d_sq = 0.0
            for k in range(D):
                diff = X[i, k] - X[j, k]
                d_sq += diff * diff
            d = np.sqrt(d_sq)
            dist[i, j] = d
            dist[j, i] = d
    return dist


@njit
def heat_diffusion_stencil(grid, alpha=0.1, steps=20):
    """2D heat diffusion Jacobi stencil update."""
    rows, cols = grid.shape
    curr = grid.copy()
    next_grid = grid.copy()
    
    for s in range(steps):
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # 5-point stencil formula
                laplacian = (
                    curr[r + 1, c] + curr[r - 1, c] +
                    curr[r, c + 1] + curr[r, c - 1] -
                    4.0 * curr[r, c]
                )
                next_grid[r, c] = curr[r, c] + alpha * laplacian
        # Swap grids
        curr, next_grid = next_grid, curr
        
    return curr


@njit
def exponential_moving_average(signal, alpha=0.05):
    """Recursive exponential smoothing filter."""
    n = len(signal)
    smoothed = np.empty(n, dtype=float64)
    smoothed[0] = signal[0]
    for i in range(1, n):
        smoothed[i] = alpha * signal[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def main():
    print("Initializing benchmark data...")
    np.random.seed(42)
    
    # Dataset sizes
    A = np.random.randn(200, 200)
    B = np.random.randn(200, 200)
    points = np.random.randn(600, 30)
    grid = np.random.randn(250, 250)
    signal = np.random.randn(100_000)

    print("Warming up Numba JIT compilation...")
    matrix_multiply(A[:10, :10], B[:10, :10])
    pairwise_distances(points[:10, :5])
    heat_diffusion_stencil(grid[:10, :10], steps=1)
    exponential_moving_average(signal[:100])

    print("Starting profiled workload benchmark loop...")
    start_time = time.time()
    
    # Run loop for ~2.5 seconds to collect thousands of profile samples
    iterations = 0
    while time.time() - start_time < 10.:
        res1 = matrix_multiply(A, B)
        res2 = pairwise_distances(points)
        res3 = heat_diffusion_stencil(grid, alpha=0.1, steps=10)
        res4 = exponential_moving_average(signal, alpha=0.05)
        iterations += 1

    elapsed = time.time() - start_time
    print(f"Benchmark completed {iterations} iterations in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
