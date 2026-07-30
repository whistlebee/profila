"""
Numba test functions for profiling tests.
"""

import numpy as np
from numba import njit


@njit
def expensive_numba_calc(arr):
    n = len(arr)
    result = np.zeros(n)
    for i in range(n):
        # Expensive line:
        result[i] = (arr[i] ** 2 + np.sin(arr[i])) / (1.0 + np.abs(arr[i]))
    for i in range(n):
        # Cheaper line:
        result[i] += 1.0
    return result


@njit
def simple_loop(n):
    total = 0.0
    for i in range(n):
        total += i * 1.5
    return total
