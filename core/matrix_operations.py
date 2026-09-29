import numpy as np


def readMatrixFromFile(filepath: str):
    return np.loadtxt(filepath, dtype=float)


def splitAugmentedMatrix(matrix: np.ndarray):
    matrix = np.array(matrix, dtype=float)
    return matrix[:, :-1], matrix[:, -1]


def multiplyMatrices(lhs: np.ndarray, rhs: np.ndarray):
    lhs = np.array(lhs, dtype=float)
    rhs = np.array(rhs, dtype=float)
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[0]:
        raise ValueError("Invalid size of matrices")
    return lhs @ rhs


def transpose(matrix: np.ndarray):
    return np.array(matrix, dtype=float).T


def residualNorm(a: np.ndarray, x: np.ndarray, b: np.ndarray):
    return float(np.linalg.norm(a @ x - b, ord=np.inf))


def isSquare(a: np.ndarray):
    return a.ndim == 2 and a.shape[0] == a.shape[1]


# Compatibility with the names used by the first version of the lab.
def read_matrix(filepath: str):
    return splitAugmentedMatrix(readMatrixFromFile(filepath))


residual_norm = residualNorm
is_square = isSquare
