from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_plot(nodes, values, table, x_star, number, f, lagrange, newton):
    x = np.linspace(nodes[0], nodes[-1], 500)
    path = Path(__file__).with_name(f"graphic_{number}.png")
    plt.figure(figsize=(11, 7))
    plt.plot(x, [f(value) for value in x], label="f(x) = e^x + x")
    plt.plot(x, [lagrange(value, nodes, values) for value in x], label="Lagrange")
    plt.plot(x, [newton(value, nodes, table) for value in x], "--", label="Newton")
    plt.scatter(nodes, values, color="black", zorder=3, label="nodes")
    plt.scatter([x_star], [f(x_star)], color="red", zorder=4, label="x*")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
