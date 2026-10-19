import numpy as np


def QRDecomposition(a: np.ndarray):
    a = np.array(a, dtype=float, copy=True)
    m, n = a.shape
    q = np.eye(m)
    r = a.copy()

    for k in range(min(m - 1, n)):
        x = r[k:, k]
        norm_x = np.linalg.norm(x)
        if np.isclose(norm_x, 0.0):
            continue

        sign = 1.0 if x[0] >= 0 else -1.0
        u = x.copy()
        u[0] += sign * norm_x
        v = u / np.linalg.norm(u)

        h = np.eye(m)
        h[k:, k:] -= 2.0 * np.outer(v, v)

        r = h @ r
        q = q @ h.T

    r[np.isclose(r, 0.0)] = 0.0
    return q, r


def VectorNormFrom(matrix: np.ndarray, column: int, startRow: int):
    return float(np.linalg.norm(matrix[startRow:, column]))


def _quasiTriangularError(a: np.ndarray):
    if a.shape[0] <= 2:
        return 0.0
    return float(np.linalg.norm(np.tril(a, k=-2), ord="fro"))


def _hasOnly1x1And2x2Blocks(a: np.ndarray, eps: float):
    i = 0
    n = a.shape[0]
    while i < n - 1:
        if abs(a[i + 1, i]) < eps:
            i += 1
        else:
            if i + 2 < n and abs(a[i + 2, i + 1]) >= eps:
                return False
            i += 2
    return True


def Eigenvalues2x2(a: float, b: float, c: float, d: float):
    trace = a + d
    determinant = a * d - b * c
    discriminant = trace**2 - 4 * determinant
    root = np.sqrt(complex(discriminant))
    return (trace + root) / 2, (trace - root) / 2


def ExtractEigenvalues(a: np.ndarray, eps: float):
    eigenvalues: list[complex] = []
    i = 0
    n = a.shape[0]

    while i < n:
        if i < n - 1 and abs(a[i + 1, i]) >= eps:
            block = a[i : i + 2, i : i + 2]
            eigenvalues.extend(Eigenvalues2x2(*block.ravel()))
            i += 2
        else:
            eigenvalues.append(complex(a[i, i]))
            i += 1

    result = np.array(eigenvalues)
    if np.allclose(result.imag, 0.0):
        return result.real
    return result


def IsQRIterationFinished(current: np.ndarray, previous: np.ndarray, eps: float):
    i = 0
    while i < current.shape[0] - 1:
        if VectorNormFrom(current, i, i + 1) <= eps:
            i += 1
            continue
        currentPair = Eigenvalues2x2(*current[i : i + 2, i : i + 2].ravel())
        previousPair = Eigenvalues2x2(*previous[i : i + 2, i : i + 2].ravel())
        if max(abs(currentPair[0] - previousPair[0]), abs(currentPair[1] - previousPair[1])) > eps:
            return False
        i += 2
    return True


def FindEigenvaluesQR(
    a: np.ndarray,
    eps: float = 1e-8,
    max_iterations: int = 10000,
):
    current = np.array(a, dtype=float, copy=True)
    if current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("Matrix must be square")

    history: list[float] = []
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        error = _quasiTriangularError(current)
        history.append(error)

        if iterations > 1 and IsQRIterationFinished(current, previous, eps):
            break

        previous = current.copy()
        q, r = QRDecomposition(current)
        current = r @ q
    else:
        raise RuntimeError("QR algorithm did not converge")

    return ExtractEigenvalues(current, eps), iterations, history
