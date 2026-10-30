from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path(__file__).with_name("graphic.png")


def main():
    x = np.linspace(1.0, 1.8, 500)
    y1 = np.cos(x) + 0.25 * x
    y2 = np.full_like(x, 0.5)

    plt.figure(figsize=(11, 7))
    plt.plot(x, y1, label="y = cos(x) + 0.25x")
    plt.plot(x, y2, label="y = 0.5")
    plt.xlim(1.0, 1.8)
    plt.ylim(0.15, 0.9)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
