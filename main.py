import numpy as np

from core.matrix_operations import read_matrix, residual_norm
from methods.iterative_solvers import seidel_method, simple_iteration
from methods.jacobi_eigen import jacobi_rotation_eigen
from methods.lu_decomposition import (
    determinant_from_lu,
    inverse_from_lu,
    lu_decomposition,
    solve_lu,
)
from methods.qr_decomposition import qr_decomposition, qr_eigenvalues
from methods.tridiagonal import thomas_algorithm, tridiagonal_to_matrix


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_lu_demo() -> None:
    print_section("1.1 LU-разложение")
    a, b = read_matrix("data/test.txt")

    l, u, p, swaps = lu_decomposition(a)
    x = solve_lu(l, u, p, b)
    determinant = determinant_from_lu(u, swaps)
    inverse = inverse_from_lu(l, u, p)

    print("A:")
    print(a)
    print("b:", b)
    print("L:")
    print(l)
    print("U:")
    print(u)
    print("P:")
    print(p)
    print("PA = LU:", np.allclose(p @ a, l @ u))
    print("x:", x)
    print("||Ax - b||:", residual_norm(a, x, b))
    print("det(A):", determinant)
    print("A^(-1):")
    print(inverse)
    print("A @ A^(-1) = I:", np.allclose(a @ inverse, np.eye(a.shape[0])))


def run_tridiagonal_demo() -> None:
    print_section("1.2 Метод прогонки")
    lower = np.array([1.0, 1.0, 1.0])
    diagonal = np.array([4.0, 4.0, 4.0, 4.0])
    upper = np.array([1.0, 1.0, 1.0])
    b = np.array([5.0, 6.0, 6.0, 5.0])

    a = tridiagonal_to_matrix(lower, diagonal, upper)
    x = thomas_algorithm(lower, diagonal, upper, b)

    print("A:")
    print(a)
    print("b:", b)
    print("x:", x)
    print("||Ax - b||:", residual_norm(a, x, b))


def run_iterative_demo() -> None:
    print_section("1.3 Метод простых итераций и метод Зейделя")
    a = np.array(
        [
            [10.0, 1.0, 1.0],
            [2.0, 10.0, 1.0],
            [2.0, 2.0, 10.0],
        ]
    )
    b = np.array([12.0, 13.0, 14.0])
    eps = 1e-8

    jacobi_x, jacobi_iterations, jacobi_history = simple_iteration(a, b, eps)
    seidel_x, seidel_iterations, seidel_history = seidel_method(a, b, eps)

    print("A:")
    print(a)
    print("b:", b)
    print("Точность:", eps)
    print("Метод простых итераций:")
    print("x:", jacobi_x)
    print("итераций:", jacobi_iterations)
    print("последняя оценка ошибки:", jacobi_history[-1])
    print("Метод Зейделя:")
    print("x:", seidel_x)
    print("итераций:", seidel_iterations)
    print("последняя оценка ошибки:", seidel_history[-1])


def run_jacobi_eigen_demo() -> None:
    print_section("1.4 Метод вращений Якоби")
    a = np.array(
        [
            [4.0, 1.0, 1.0],
            [1.0, 3.0, 0.0],
            [1.0, 0.0, 2.0],
        ]
    )
    eps = 1e-8

    eigenvalues, eigenvectors, iterations, history = jacobi_rotation_eigen(a, eps)

    print("A:")
    print(a)
    print("Собственные значения:", eigenvalues)
    print("Собственные векторы:")
    print(eigenvectors)
    print("итераций:", iterations)
    print("финальная погрешность:", history[-1])
    print("Проверка A V = V Lambda:", np.allclose(a @ eigenvectors, eigenvectors @ np.diag(eigenvalues)))


def run_qr_demo() -> None:
    print_section("1.5 QR-разложение и QR-алгоритм")
    a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
        ]
    )
    eps = 1e-8

    q, r = qr_decomposition(a)
    eigenvalues, iterations, history = qr_eigenvalues(a, eps)

    print("A:")
    print(a)
    print("Q:")
    print(q)
    print("R:")
    print(r)
    print("Q @ R = A:", np.allclose(q @ r, a))
    print("Q^T @ Q = I:", np.allclose(q.T @ q, np.eye(a.shape[0])))
    print("Собственные значения:", eigenvalues)
    print("итераций:", iterations)
    print("финальная погрешность:", history[-1])


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    run_lu_demo()
    run_tridiagonal_demo()
    run_iterative_demo()
    run_jacobi_eigen_demo()
    run_qr_demo()


if __name__ == "__main__":
    main()
