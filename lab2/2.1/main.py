from math import log


def f(x: float):
    return 4.0**x - 5.0 * x - 2.0


def df(x: float):
    return log(4.0) * 4.0**x - 5.0


def phi(x: float):
    return log(5.0 * x + 2.0) / log(4.0)


def dphi(x: float):
    return 5.0 / ((5.0 * x + 2.0) * log(4.0))


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
    print("4^x - 5x - 2 = 0")
    eps = read_eps()

    x0 = 1.68
    print("phi(x) = log_4(5x + 2)")
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
