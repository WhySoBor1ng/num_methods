import numpy as np


def OffDiagonalNorm(matrix: np.ndarray):
    upper = np.triu(np.array(matrix, dtype=float), k=1)
    return float(np.sqrt(np.sum(upper**2)))


def _max_off_diagonal_index(a: np.ndarray):
    upper = np.triu(np.abs(a), k=1)
    index = int(np.argmax(upper))
    return np.unravel_index(index, a.shape)


def RotationMatrix(matrix: np.ndarray, i: int, j: int):
    matrix = np.array(matrix, dtype=float)
    rotationMatrix = np.eye(matrix.shape[0])
    if np.isclose(matrix[i, i], matrix[j, j]):
        phi = np.pi / 4
    else:
        phi = 0.5 * np.arctan(2 * matrix[i, j] / (matrix[i, i] - matrix[j, j]))

    cosPhi = np.cos(phi)
    sinPhi = np.sin(phi)
    rotationMatrix[i, i] = cosPhi
    rotationMatrix[j, j] = cosPhi
    rotationMatrix[i, j] = -sinPhi
    rotationMatrix[j, i] = sinPhi
    return rotationMatrix


def JacobiRotationMethod(
    a: np.ndarray,
    eps: float = 1e-8,
    max_iterations: int = 10000,
):
    a = np.array(a, dtype=float, copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square")
    if not np.allclose(a, a.T):
        raise ValueError("Jacobi rotation method requires a symmetric matrix")

    n = a.shape[0]
    eigenvectors = np.eye(n)
    history: list[float] = []

    for iteration in range(1, max_iterations + 1):
        error = OffDiagonalNorm(a)
        history.append(error)

        if error < eps:
            return np.diag(a), eigenvectors, iteration - 1, history

        i, j = _max_off_diagonal_index(a)

        rotation = RotationMatrix(a, i, j)
        a = rotation.T @ a @ rotation
        eigenvectors = eigenvectors @ rotation

    raise RuntimeError("Jacobi rotation method did not converge")


jacobi_rotation_eigen = JacobiRotationMethod
