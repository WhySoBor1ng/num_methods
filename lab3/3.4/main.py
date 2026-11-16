X = [-0.2, 0.0, 0.2, 0.4, 0.6]
Y = [-0.40136, 0.0, 0.40136, 0.81152, 1.2435]
X_STAR = 0.2


def derivatives(x: list[float], y: list[float], point: float):
    index = x.index(point)
    if index == 0 or index == len(x) - 1:
        raise ValueError("Central differences require adjacent nodes")
    used = [index - 1, index, index + 1]
    h_left = x[index] - x[index - 1]
    h_right = x[index + 1] - x[index]
    if abs(h_left - h_right) > 1e-12:
        raise ValueError("The three nodes must be equally spaced")
    first = (y[index + 1] - y[index - 1]) / (2.0 * h_left)
    second = (y[index + 1] - 2.0 * y[index] + y[index - 1]) / h_left**2
    return first, second, used


def main():
    print("numerical differentiation")
    print("i, x_i, y_i")
    for i, (x, y) in enumerate(zip(X, Y)):
        print(f"{i}, {x:.9f}, {y:.9f}")
    first, second, used = derivatives(X, Y, X_STAR)
    print(f"\nx* = {X_STAR:.9f}")
    print("nodes used:")
    for i in used:
        print(f"i = {i}, x_i = {X[i]:.9f}, y_i = {Y[i]:.9f}")
    print(f"f'(x*) = {first:.9f}")
    print(f"f''(x*) = {second:.9f}")


if __name__ == "__main__":
    main()
