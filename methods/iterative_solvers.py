import numpy as np


def _validate_system(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square")
    if b.shape[0] != a.shape[0]:
        raise ValueError("Vector size must match matrix size")
    if np.any(np.isclose(np.diag(a), 0.0)):
        raise ValueError("Zero diagonal element")

    return a, b


def simple_iteration(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-6,
    max_iterations: int = 10000,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, int, list[float]]:
    a, b = _validate_system(a, b)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    diagonal = np.diag(a)
    remainder = a - np.diagflat(diagonal)
    history: list[float] = []

    for iteration in range(1, max_iterations + 1):
        x_new = (b - remainder @ x) / diagonal
        error = float(np.linalg.norm(x_new - x, ord=np.inf))
        history.append(error)

        if error < eps:
            return x_new, iteration, history

        x = x_new

    raise RuntimeError("Simple iteration method did not converge")


def seidel_method(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-6,
    max_iterations: int = 10000,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, int, list[float]]:
    a, b = _validate_system(a, b)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    history: list[float] = []

    for iteration in range(1, max_iterations + 1):
        x_old = x.copy()

        for i in range(a.shape[0]):
            left = np.dot(a[i, :i], x[:i])
            right = np.dot(a[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - left - right) / a[i, i]

        error = float(np.linalg.norm(x - x_old, ord=np.inf))
        history.append(error)

        if error < eps:
            return x, iteration, history

    raise RuntimeError("Seidel method did not converge")
