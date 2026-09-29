import numpy as np


def readTridiagonalMatrixFromFile(path: str):
    fullMatrix = np.loadtxt(path, dtype=float)
    if fullMatrix.ndim != 2 or fullMatrix.shape[1] != fullMatrix.shape[0] + 1:
        raise ValueError("Expected an augmented matrix [A | b]")

    matrix = fullMatrix[:, :-1]
    if not np.allclose(matrix, np.triu(np.tril(matrix, 1), -1)):
        raise ValueError("Matrix must be tridiagonal")

    return (
        np.diag(matrix, 1),
        np.diag(matrix),
        np.diag(matrix, -1),
        fullMatrix[:, -1],
    )


def solveTridiagonal(
    upperDiag: np.ndarray,
    mainDiag: np.ndarray,
    lowerDiag: np.ndarray,
    ansCol: np.ndarray,
):
    upper = np.array(upperDiag, dtype=float, copy=True)
    main = np.array(mainDiag, dtype=float, copy=True)
    lower = np.array(lowerDiag, dtype=float, copy=True)
    answer = np.array(ansCol, dtype=float, copy=True)

    n = answer.size
    if main.size != n:
        raise ValueError("Invalid tridiagonal system dimensions")
    if upper.size == n - 1:
        upper = np.append(upper, 0.0)
    if lower.size == n - 1:
        lower = np.insert(lower, 0, 0.0)
    if upper.size != n or lower.size != n:
        raise ValueError("Invalid tridiagonal system dimensions")

    p = np.zeros(n + 1)
    q = np.zeros(n + 1)
    x = np.zeros(n)

    for i in range(n):
        denominator = -main[i] - lower[i] * p[i]
        if np.isclose(denominator, 0.0):
            raise ValueError("Division by zero")
        p[i + 1] = upper[i] / denominator
        q[i + 1] = (lower[i] * q[i] - answer[i]) / denominator

    xNext = 0.0
    for i in range(n - 1, -1, -1):
        x[i] = p[i + 1] * xNext + q[i + 1]
        xNext = x[i]

    return p, q, x


def tridiagonalToMatrix(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
):
    b = np.array(diagonal, dtype=float)
    matrix = np.diag(b)
    matrix += np.diag(np.array(lower, dtype=float), k=-1)
    matrix += np.diag(np.array(upper, dtype=float), k=1)
    return matrix


def thomas_algorithm(lower, diagonal, upper, rhs):
    return solveTridiagonal(upper, diagonal, lower, rhs)[2]


tridiagonal_to_matrix = tridiagonalToMatrix
