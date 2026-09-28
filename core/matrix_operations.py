import numpy as np


def read_matrix(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.loadtxt(filepath)
    a = matrix[:, :-1]
    b = matrix[:, -1]
    return a, b


def residual_norm(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a @ x - b, ord=np.inf))


def is_square(a: np.ndarray) -> bool:
    return a.ndim == 2 and a.shape[0] == a.shape[1]
