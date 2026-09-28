import numpy as np


def _off_diagonal_norm(a: np.ndarray) -> float:
    return float(np.sqrt(np.sum(a**2) - np.sum(np.diag(a) ** 2)))


def _max_off_diagonal_index(a: np.ndarray) -> tuple[int, int]:
    upper = np.triu(np.abs(a), k=1)
    index = int(np.argmax(upper))
    return np.unravel_index(index, a.shape)


def jacobi_rotation_eigen(
    a: np.ndarray,
    eps: float = 1e-8,
    max_iterations: int = 10000,
) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
    """Find eigenvalues and eigenvectors of a symmetric matrix."""
    a = np.array(a, dtype=float, copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square")
    if not np.allclose(a, a.T):
        raise ValueError("Jacobi rotation method requires a symmetric matrix")

    n = a.shape[0]
    eigenvectors = np.eye(n)
    history: list[float] = []

    for iteration in range(1, max_iterations + 1):
        error = _off_diagonal_norm(a)
        history.append(error)

        if error < eps:
            return np.diag(a), eigenvectors, iteration - 1, history

        i, j = _max_off_diagonal_index(a)

        if np.isclose(a[i, i], a[j, j]):
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan2(2 * a[i, j], a[i, i] - a[j, j])

        c = np.cos(phi)
        s = np.sin(phi)
        rotation = np.eye(n)
        rotation[i, i] = c
        rotation[j, j] = c
        rotation[i, j] = -s
        rotation[j, i] = s

        a = rotation.T @ a @ rotation
        eigenvectors = eigenvectors @ rotation

    raise RuntimeError("Jacobi rotation method did not converge")
