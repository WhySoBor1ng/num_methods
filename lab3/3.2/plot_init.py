from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path(__file__).with_name("graphic.png")


def save_plot(nodes, values, coefficients, x_star, spline_value):
    plt.figure(figsize=(11, 7))
    for i in range(len(coefficients)):
        x = np.linspace(nodes[i], nodes[i + 1], 100)
        y = [spline_value(point, nodes, coefficients) for point in x]
        plt.plot(x, y, label=f"S_{i + 1}(x)")
    plt.scatter(nodes, values, color="black", zorder=3, label="nodes")
    plt.scatter([x_star], [spline_value(x_star, nodes, coefficients)], color="red", zorder=4, label="(x*, S(x*))")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    plt.close()
