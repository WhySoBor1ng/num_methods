import numpy as np


def thomas_algorithm(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal linear system using the Thomas algorithm."""
    a = np.array(lower, dtype=float, copy=True)
    b = np.array(diagonal, dtype=float, copy=True)
    c = np.array(upper, dtype=float, copy=True)
    d = np.array(rhs, dtype=float, copy=True)

    n = b.size
    if a.size != n - 1 or c.size != n - 1 or d.size != n:
        raise ValueError("Invalid tridiagonal system dimensions")

    for i in range(1, n):
        if np.isclose(b[i - 1], 0.0):
            raise ValueError("Zero pivot in tridiagonal solver")
        factor = a[i - 1] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    if np.isclose(b[-1], 0.0):
        raise ValueError("Zero pivot in tridiagonal solver")

    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def tridiagonal_to_matrix(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    b = np.array(diagonal, dtype=float)
    matrix = np.diag(b)
    matrix += np.diag(np.array(lower, dtype=float), k=-1)
    matrix += np.diag(np.array(upper, dtype=float), k=1)
    return matrix
