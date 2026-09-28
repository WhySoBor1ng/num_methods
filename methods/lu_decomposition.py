import numpy as np


def lu_decomposition(a: np.ndarray):
    a = np.array(a, dtype=float, copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Non-square matrix")

    n = a.shape[0]
    u = a.copy()
    l = np.zeros((n, n), dtype=float)
    p_idx = np.arange(n)
    swaps_count = 0

    for k in range(n):
        max_row = k
        max_val = abs(u[k, k])
        for i in range(k + 1, n):
            if abs(u[i, k]) > max_val:
                max_row = i
                max_val = abs(u[i, k])

        if np.isclose(max_val, 0.0):
            raise ValueError("Singular matrix")

        if max_row != k:
            u[[k, max_row]] = u[[max_row, k]]

            if k > 0:
                l[[k, max_row], :k] = l[[max_row, k], :k]

            p_idx[[k, max_row]] = p_idx[[max_row, k]]
            swaps_count += 1

        for i in range(k + 1, n):
            l[i, k] = u[i, k] / u[k, k]
            u[i, k:] = u[i, k:] - l[i, k] * u[k, k:]
            u[i, k] = 0.0

    np.fill_diagonal(l, 1.0)
    
    p = np.zeros((n, n), dtype=float)
    for i in range(n):
        p[i, p_idx[i]] = 1.0
    
    return l, u, p, swaps_count


def forward_substitution(l, b):
    l = np.array(l, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(l[i, :i], y[:i])

    return y


def backward_substitution(u, y):
    u = np.array(u, dtype=float)
    y = np.array(y, dtype=float)
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if np.isclose(u[i, i], 0.0):
            raise ValueError("Singular matrix")
        x[i] = (y[i] - np.dot(u[i, i + 1:], x[i + 1:])) / u[i, i]

    return x


def solve_lu(l, u, p, b):
    pb = p @ b

    y = forward_substitution(l, pb)
    x = backward_substitution(u, y)

    return x


def determinant_from_lu(u, swaps_count):
    return float((-1) ** swaps_count * np.prod(np.diag(u)))


def inverse_from_lu(l, u, p):
    n = l.shape[0]
    inverse = np.zeros((n, n))

    i = np.eye(n)

    for j in range(n):
        inverse[:, j] = solve_lu(
            l, u, p, i[:, j]
        )

    return inverse
