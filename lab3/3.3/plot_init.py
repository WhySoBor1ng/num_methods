from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path(__file__).with_name("graphic.png")


def save_plot(nodes, values, polynomials):
    x = np.linspace(nodes[0], nodes[-1], 500)
    plt.figure(figsize=(11, 7))
    plt.scatter(nodes, values, color="black", zorder=3, label="data")
    for degree, polynomial in enumerate(polynomials, 1):
        plt.plot(x, polynomial(x), label=f"P_{degree}(x)")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    plt.close()
