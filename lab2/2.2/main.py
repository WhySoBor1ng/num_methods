from math import cos, exp, sin

import numpy as np


def f1(x: float, y: float):
    return 3.0 * x - cos(y)


def f2(x: float, y: float):
    return 3.0 * y - exp(x)


def phi1(y: float):
    return cos(y) / 3.0


def phi2(x: float):
    return exp(x) / 3.0


def simple_iteration(x: float, y: float, eps: float, max_iterations: int = 1000):
    errors = []

    for iteration in range(1, max_iterations + 1):
        next_x = phi1(y)
        next_y = phi2(x)
        error = np.linalg.norm([next_x - x, next_y - y], ord=np.inf)
        errors.append(float(error))

        residual = np.linalg.norm([f1(next_x, next_y), f2(next_x, next_y)], ord=np.inf)
        if error <= eps and residual <= eps:
            return next_x, next_y, iteration, errors

        x = next_x
        y = next_y

    raise RuntimeError("Simple iteration method did not converge")


def newton(x: float, y: float, eps: float, max_iterations: int = 1000):
    errors = []

    for iteration in range(1, max_iterations + 1):
        jacobian = np.array(
            [
                [3.0, sin(y)],
                [-exp(x), 3.0],
            ]
        )
        values = np.array([f1(x, y), f2(x, y)])
        dx, dy = np.linalg.solve(jacobian, -values)

        x += dx
        y += dy

        error = np.linalg.norm([dx, dy], ord=np.inf)
        errors.append(float(error))

        residual = np.linalg.norm([f1(x, y), f2(x, y)], ord=np.inf)
        if error <= eps and residual <= eps:
            return x, y, iteration, errors

    raise RuntimeError("Newton method did not converge")


def print_errors(title: str, errors: list[float]):
    print(title)
    print("k    error")
    for k, error in enumerate(errors, start=1):
        print(f"{k:<4} {error:.6f}")
    print()


def read_eps():
    eps = float(input("Eps: "))
    if eps <= 0.0:
        raise ValueError("Eps must be positive")
    return eps


def main():
    print("3x - cos(y) = 0")
    print("3y - e^x = 0")
    eps = read_eps()

    x0 = 0.33
    y0 = 0.47
    print("\ninitial approximation:")
    print(f"x0 = {x0:.6f}, y0 = {y0:.6f}")
    print("phi1(y) = cos(y) / 3")
    print("phi2(x) = e^x / 3\n")

    iteration_x, iteration_y, iteration_count, iteration_errors = simple_iteration(
        x0, y0, eps
    )
    newton_x, newton_y, newton_count, newton_errors = newton(x0, y0, eps)

    print("simple iterations:")
    print(f"x = {iteration_x:.6f}")
    print(f"y = {iteration_y:.6f}")
    print(f"f1(x, y) = {f1(iteration_x, iteration_y):.6f}")
    print(f"f2(x, y) = {f2(iteration_x, iteration_y):.6f}")
    print(f"iterations = {iteration_count}\n")

    print("newton:")
    print(f"x = {newton_x:.6f}")
    print(f"y = {newton_y:.6f}")
    print(f"f1(x, y) = {f1(newton_x, newton_y):.6f}")
    print(f"f2(x, y) = {f2(newton_x, newton_y):.6f}")
    print(f"iterations = {newton_count}\n")

    print_errors("simple iterations errors:", iteration_errors)
    print_errors("newton errors:", newton_errors)


if __name__ == "__main__":
    main()
