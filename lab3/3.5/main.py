from math import atan, log


LEFT = -2.0
RIGHT = 2.0
H1 = 1.0
H2 = 0.5


def f(x: float):
    return 1.0 / (256.0 - x**4)


def exact_integral():
    return log(3.0) / 128.0 + atan(0.5) / 64.0


def interval_count(left: float, right: float, h: float):
    if h <= 0.0:
        raise ValueError("Step must be positive")
    n = round((right - left) / h)
    if n < 1 or abs(n * h - (right - left)) > 1e-10:
        raise ValueError("Step must divide the interval into an integer number of parts")
    return n


def integrate(method: str, left: float, right: float, h: float):
    n = interval_count(left, right, h)
    if method == "Simpson" and n % 2:
        raise ValueError("Simpson's method requires an even number of parts")
    if method == "left rectangles":
        return h * sum(f(left + i * h) for i in range(n))
    if method == "right rectangles":
        return h * sum(f(left + (i + 1) * h) for i in range(n))
    if method == "middle rectangles":
        return h * sum(f(left + (i + 0.5) * h) for i in range(n))
    if method == "trapezoids":
        return h * (0.5 * f(left) + sum(f(left + i * h) for i in range(1, n)) + 0.5 * f(right))
    if method == "Simpson":
        return h / 3.0 * (f(left) + f(right) + sum(
            (4 if i % 2 else 2) * f(left + i * h) for i in range(1, n)
        ))
    raise ValueError("Unknown method")


def main():
    methods = [
        ("left rectangles", 1),
        ("right rectangles", 1),
        ("middle rectangles", 2),
        ("trapezoids", 2),
        ("Simpson", 4),
    ]
    exact = exact_integral()
    print("integral of 1/(256-x^4) from -2 to 2")
    print(f"h1 = {H1:.9f}; h2 = {H2:.9f}; exact value = {exact:.9f}")
    print("method, I(h1), I(h2), p, Runge-Romberg, error estimate, |I(h2)-Iexact|")
    results = []
    for name, order in methods:
        coarse = integrate(name, LEFT, RIGHT, H1)
        fine = integrate(name, LEFT, RIGHT, H2)
        correction = (fine - coarse) / ((H1 / H2)**order - 1.0)
        refined = fine + correction
        estimate = abs(correction)
        actual = abs(fine - exact)
        results.append((actual, name))
        print(f"{name}, {coarse:.9f}, {fine:.9f}, {order}, {refined:.9f}, {estimate:.9f}, {actual:.9f}")
    best_error, best_name = min(results)
    print(f"most accurate at h2: {best_name}, absolute error {best_error:.9f}")


if __name__ == "__main__":
    main()
