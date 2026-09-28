import numpy as np


def qr_decomposition(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute QR decomposition with Householder reflections."""
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


def _quasi_triangular_error(a: np.ndarray) -> float:
    if a.shape[0] <= 2:
        return 0.0
    return float(np.linalg.norm(np.tril(a, k=-2), ord="fro"))


def _has_only_1x1_and_2x2_blocks(a: np.ndarray, eps: float) -> bool:
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


def _eigenvalues_from_quasi_triangular(a: np.ndarray, eps: float) -> np.ndarray:
    eigenvalues: list[complex] = []
    i = 0
    n = a.shape[0]

    while i < n:
        if i < n - 1 and abs(a[i + 1, i]) >= eps:
            block = a[i : i + 2, i : i + 2]
            trace = block[0, 0] + block[1, 1]
            determinant = block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0]
            discriminant = trace**2 - 4 * determinant
            root = np.sqrt(complex(discriminant))
            eigenvalues.extend([(trace + root) / 2, (trace - root) / 2])
            i += 2
        else:
            eigenvalues.append(complex(a[i, i]))
            i += 1

    result = np.array(eigenvalues)
    if np.allclose(result.imag, 0.0):
        return result.real
    return result


def qr_eigenvalues(
    a: np.ndarray,
    eps: float = 1e-8,
    max_iterations: int = 10000,
) -> tuple[np.ndarray, int, list[float]]:
    """Find eigenvalues using the QR algorithm."""
    current = np.array(a, dtype=float, copy=True)
    if current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("Matrix must be square")

    history: list[float] = []
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        error = _quasi_triangular_error(current)
        history.append(error)

        if error < eps and _has_only_1x1_and_2x2_blocks(current, eps):
            break

        shift = current[-1, -1]
        q, r = qr_decomposition(current - shift * np.eye(current.shape[0]))
        current = r @ q + shift * np.eye(current.shape[0])
    else:
        raise RuntimeError("QR algorithm did not converge")

    return _eigenvalues_from_quasi_triangular(current, eps), iterations, history
