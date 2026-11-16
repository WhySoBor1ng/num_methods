import numpy as np

from plot_init import save_plot


X = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
Y = np.array([-2.9502, -1.8647, -0.63212, 1.0, 3.7183, 9.3891])


def polynomial_string(coefficients: np.ndarray):
    degree = len(coefficients) - 1
    terms = []
    for i, coefficient in enumerate(coefficients):
        power = degree - i
        factor = "" if power == 0 else "x" if power == 1 else f"x^{power}"
        terms.append(f"({coefficient:.9f}){factor}")
    return " + ".join(terms)


def main():
    print("least squares method")
    print("i, x_i, y_i")
    for i, (x, y) in enumerate(zip(X, Y)):
        print(f"{i}, {x:.9f}, {y:.9f}")
    polynomials = []
    errors = []
    for degree in (1, 2, 3):
        coefficients = np.polyfit(X, Y, degree)
        polynomial = np.poly1d(coefficients)
        mse = float(np.mean((polynomial(X) - Y) ** 2))
        polynomials.append(polynomial)
        errors.append(mse)
        print(f"\nP_{degree} coefficients:")
        print(", ".join(f"{value:.9f}" for value in coefficients))
        print(f"P_{degree}(x) = {polynomial_string(coefficients)}")
        print(f"MSE{degree} = {mse:.9f}")
    best = int(np.argmin(errors)) + 1
    order = sorted(range(1, 4), key=lambda degree: errors[degree - 1])
    print("\ncomparison: " + " < ".join(f"MSE{degree}" for degree in order))
    print(f"best fit: P_{best}(x)")
    save_plot(X, Y, polynomials)


if __name__ == "__main__":
    main()
