from math import exp

from plot_init import save_plot


X_STAR = -0.5
NODE_SETS = [[-2.0, -1.0, 0.0, 1.0], [-2.0, -1.0, 0.2, 1.0]]


def f(x: float):
    return exp(x) + x


def lagrange(x: float, nodes: list[float], values: list[float]):
    result = 0.0
    for i, value in enumerate(values):
        term = value
        for j, node in enumerate(nodes):
            if i != j:
                term *= (x - node) / (nodes[i] - node)
        result += term
    return result


def divided_differences(nodes: list[float], values: list[float]):
    table = [values.copy()]
    for order in range(1, len(nodes)):
        previous = table[-1]
        table.append([
            (previous[i + 1] - previous[i]) / (nodes[i + order] - nodes[i])
            for i in range(len(nodes) - order)
        ])
    return table


def newton(x: float, nodes: list[float], table: list[list[float]]):
    result = table[-1][0]
    for i in range(len(nodes) - 2, -1, -1):
        result = result * (x - nodes[i]) + table[i][0]
    return result


def lagrange_polynomial(nodes: list[float], values: list[float]):
    terms = []
    for i, value in enumerate(values):
        denominator = 1.0
        factors = []
        for j, node in enumerate(nodes):
            if i != j:
                denominator *= nodes[i] - node
                factors.append(f"(x - ({node:.9f}))")
        terms.append(f"({value / denominator:.9f})*" + "*".join(factors))
    return " + ".join(terms)


def newton_polynomial(nodes: list[float], table: list[list[float]]):
    terms = []
    for order in range(len(nodes)):
        factors = "".join(f"*(x - ({nodes[i]:.9f}))" for i in range(order))
        terms.append(f"({table[order][0]:.9f}){factors}")
    return " + ".join(terms)


def main():
    print("f(x) = e^x + x")
    for number, nodes in enumerate(NODE_SETS, start=1):
        values = [f(x) for x in nodes]
        table = divided_differences(nodes, values)
        print(f"\nnode set {number}:")
        print("i, x_i, y_i")
        for i, (x, value) in enumerate(zip(nodes, values)):
            print(f"{i}, {x:.9f}, {value:.9f}")
        print("L(x) =", lagrange_polynomial(nodes, values))
        print("divided differences:")
        print("i, " + ", ".join(f"order {i}" for i in range(len(nodes))))
        for i in range(len(nodes)):
            print(f"{i}, " + ", ".join(
                f"{table[order][i]:.9f}" if i < len(table[order]) else "-"
                for order in range(len(nodes))
            ))
        print("N(x) =", newton_polynomial(nodes, table))
        lagrange_value = lagrange(X_STAR, nodes, values)
        newton_value = newton(X_STAR, nodes, table)
        exact = f(X_STAR)
        print(f"x* = {X_STAR:.9f}")
        print(f"L(x*) = {lagrange_value:.9f}")
        print(f"N(x*) = {newton_value:.9f}")
        print(f"f(x*) = {exact:.9f}")
        print(f"|L(x*) - f(x*)| = {abs(lagrange_value - exact):.9f}")
        print(f"|N(x*) - f(x*)| = {abs(newton_value - exact):.9f}")
        print(f"|L(x*) - N(x*)| = {abs(lagrange_value - newton_value):.9f}")
        save_plot(nodes, values, table, X_STAR, number, f, lagrange, newton)


if __name__ == "__main__":
    main()
