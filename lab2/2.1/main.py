from math import acos, cos, sin


def f(x: float):
    return cos(x) + 0.25 * x - 0.5


def df(x: float):
    return -sin(x) + 0.25


def phi(x: float):
    return acos(0.5 - 0.25 * x)


def dphi(x: float):
    value = 0.5 - 0.25 * x
    return 0.25 / (1.0 - value**2) ** 0.5


def simple_iteration(x: float, eps: float, max_iterations: int = 1000):
    for iteration in range(1, max_iterations + 1):
        next_x = phi(x)

        if abs(next_x - x) <= eps and abs(f(next_x)) <= eps:
            return next_x, iteration

        x = next_x

    raise RuntimeError("Simple iteration method did not converge")


def newton(x: float, eps: float, max_iterations: int = 1000):
    for iteration in range(1, max_iterations + 1):
        derivative = df(x)
        if abs(derivative) < 1e-15:
            raise ZeroDivisionError("Derivative is zero")

        next_x = x - f(x) / derivative

        if abs(next_x - x) <= eps and abs(f(next_x)) <= eps:
            return next_x, iteration

        x = next_x

    raise RuntimeError("Newton method did not converge")


def read_eps():
    eps = float(input("Eps: "))
    if eps <= 0.0:
        raise ValueError("Eps must be positive")
    return eps


def main():
    print("cos(x) + 0.25x - 0.5 = 0")
    eps = read_eps()

    x0 = 1.4
    print("phi(x) = arccos(0.5 - 0.25x)")
    print(f"|phi'(x0)| = {abs(dphi(x0)):.6f} < 1\n")

    iteration_root, iteration_count = simple_iteration(x0, eps)
    newton_root, newton_count = newton(x0, eps)

    print("simple iterations:")
    print(f"x = {iteration_root:.6f}")
    print(f"f(x) = {f(iteration_root):.6f}")
    print(f"iterations = {iteration_count}\n")

    print("newton:")
    print(f"x = {newton_root:.6f}")
    print(f"f(x) = {f(newton_root):.6f}")
    print(f"iterations = {newton_count}")


if __name__ == "__main__":
    main()
