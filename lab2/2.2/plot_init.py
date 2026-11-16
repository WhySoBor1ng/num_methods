from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path(__file__).with_name("graphic.png")


def main():
    x = np.linspace(0.2, 0.8, 500)
    first_equation_y = np.arccos(x - 1.0)
    second_equation_y = 2 + np.log10(x + 1.0)

    plt.figure(figsize=(11, 7))
    plt.plot(x, first_equation_y, label="x1 - cos(x2) = 1")
    plt.plot(x, second_equation_y, label="x2 - lg(x1 + 1) = 2")
    plt.xlim(0.2, 0.8)
    plt.ylim(1.8, 2.5)
    plt.grid(True)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
