from bisect import bisect_right

import numpy as np

from plot_init import save_plot


X = [-2.0, -1.0, 0.0, 1.0, 2.0]
Y = [-1.8647, -0.63212, 1.0, 3.7183, 9.3891]
X_STAR = -0.5


def spline_coefficients(x: list[float], y: list[float]):
    n = len(x) - 1
    h = np.diff(x)
    matrix = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    matrix[0, 0] = matrix[n, n] = 1.0
    for i in range(1, n):
        matrix[i, i - 1] = h[i - 1]
        matrix[i, i] = 2.0 * (h[i - 1] + h[i])
        matrix[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    second = np.linalg.solve(matrix, rhs)
    result = []
    for i in range(n):
        a = y[i]
        b = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * second[i] + second[i + 1]) / 6.0
        c = second[i] / 2.0
        d = (second[i + 1] - second[i]) / (6.0 * h[i])
        result.append((a, b, c, d))
    return result


def interval_index(x: list[float], point: float):
    if point < x[0] or point > x[-1]:
        raise ValueError("Point is outside the spline interval")
    return min(bisect_right(x, point) - 1, len(x) - 2)


def spline_value(point: float, x: list[float], coefficients):
    i = interval_index(x, point)
    a, b, c, d = coefficients[i]
    dx = point - x[i]
    return a + dx * (b + dx * (c + dx * d))


def main():
    coefficients = spline_coefficients(X, Y)
    print("natural cubic spline")
    print("i, x_i, y_i")
    for i, (x, y) in enumerate(zip(X, Y)):
        print(f"{i}, {x:.9f}, {y:.9f}")
    print("\ncoefficients: S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3")
    print("i, interval, a_i, b_i, c_i, d_i")
    for i, (a, b, c, d) in enumerate(coefficients):
        print(f"{i + 1}, [{X[i]:.9f}, {X[i + 1]:.9f}], {a:.9f}, {b:.9f}, {c:.9f}, {d:.9f}")
    i = interval_index(X, X_STAR)
    value = spline_value(X_STAR, X, coefficients)
    print(f"\nx* = {X_STAR:.9f}")
    print(f"interval {i + 1}: [{X[i]:.9f}, {X[i + 1]:.9f}]")
    print(f"S(x*) = {value:.9f}")
    print("spline at nodes:")
    for j, (x, y) in enumerate(zip(X, Y)):
        node_value = spline_value(x, X, coefficients)
        print(f"x_{j} = {x:.9f}: S(x_{j}) = {node_value:.9f}, y_{j} = {y:.9f}, |difference| = {abs(node_value - y):.9f}")
    left_second = 2.0 * coefficients[0][2]
    right_h = X[-1] - X[-2]
    right_second = 2.0 * coefficients[-1][2] + 6.0 * coefficients[-1][3] * right_h
    print(f"S''({X[0]:.9f}) = {left_second:.9f}")
    print(f"S''({X[-1]:.9f}) = {right_second:.9f}")
    save_plot(X, Y, coefficients, X_STAR, spline_value)


if __name__ == "__main__":
    main()
