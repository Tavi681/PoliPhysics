"""Small 3-vector helpers (numpy's generic ``cross`` is slow for single vectors)."""

import numpy as np


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
