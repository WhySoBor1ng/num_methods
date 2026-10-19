import numpy as np


def _validateSystem(a: np.ndarray, b: np.ndarray):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square")
    if b.shape[0] != a.shape[0]:
        raise ValueError("Vector size must match matrix size")
    if np.any(np.isclose(np.diag(a), 0.0)):
        raise ValueError("Zero diagonal element")

    return a, b


def getQ(a: np.ndarray):
    a, _ = _validateSystem(a, np.zeros(len(a)))
    q = 0.0
    for i in range(a.shape[0]):
        rowSum = np.sum(np.abs(a[i])) - abs(a[i, i])
        q = max(q, float(rowSum / abs(a[i, i])))
    return q


def solveIteration(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-6,
    max_iterations: int = 10000,
    x0: np.ndarray | None = None,
):
    a, b = _validateSystem(a, b)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    diagonal = np.diag(a)
    remainder = a - np.diagflat(diagonal)
    history: list[float] = []
    q = getQ(a)

    for iteration in range(1, max_iterations + 1):
        x_new = (b - remainder @ x) / diagonal
        maxDiff = float(np.linalg.norm(x_new - x, ord=np.inf))
        error = q / (1.0 - q) * maxDiff if q < 1.0 else maxDiff
        history.append(error)

        if error < eps:
            return x_new, iteration, history

        x = x_new

    raise RuntimeError("Simple iteration method did not converge")


def solveSeidel(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-6,
    max_iterations: int = 10000,
    x0: np.ndarray | None = None,
):
    a, b = _validateSystem(a, b)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    history: list[float] = []
    q = getQ(a)

    for iteration in range(1, max_iterations + 1):
        x_old = x.copy()

        for i in range(a.shape[0]):
            left = np.dot(a[i, :i], x[:i])
            right = np.dot(a[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - left - right) / a[i, i]

        maxDiff = float(np.linalg.norm(x - x_old, ord=np.inf))
        error = q / (1.0 - q) * maxDiff if q < 1.0 else maxDiff
        history.append(error)

        if error < eps:
            return x, iteration, history

    raise RuntimeError("Seidel method did not converge")
