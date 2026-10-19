from os import PathLike

import numpy as np


def readMatrixFromFile(filepath: str | PathLike[str]):
    return np.loadtxt(filepath, dtype=float)


def splitAugmentedMatrix(matrix: np.ndarray):
    matrix = np.array(matrix, dtype=float)
    return matrix[:, :-1], matrix[:, -1]
