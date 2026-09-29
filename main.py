from pathlib import Path

import numpy as np

from core.matrix_operations import (
    multiplyMatrices,
    readMatrixFromFile,
    residualNorm,
    splitAugmentedMatrix,
    transpose,
)
from methods.iterative_solvers import solveGauss, solveIteration, solveSeidel
from methods.jacobi_eigen import JacobiRotationMethod
from methods.lu_decomposition import (
    LUDecomposition,
    LUSolve,
    calculateInverse,
    determinantLU,
)
from methods.qr_decomposition import FindEigenvaluesQR, QRDecomposition
from methods.tridiagonal import (
    readTridiagonalMatrixFromFile,
    solveTridiagonal,
    tridiagonalToMatrix,
)


DATA_DIR = Path(__file__).parent / "data"


def printSection(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def runLUExample():
    printSection("1.1 LU-разложение")
    fullMatrix = readMatrixFromFile(DATA_DIR / "lab1_1.txt")
    matrix, b = splitAugmentedMatrix(fullMatrix)

    l, u, p, permutationsCount = LUDecomposition(matrix)
    x = LUSolve(l, u, p, b)
    inverse = calculateInverse(l, u, p)

    print("Расширенная матрица [A | b]:")
    print(fullMatrix)
    print("L:")
    print(l)
    print("U:")
    print(u)
    print("Решение Ax = b:", x)
    print("Определитель A:", determinantLU(u, permutationsCount))
    print("Обратная матрица A:")
    print(inverse)
    print("Проверка P*A = L*U:", np.allclose(p @ matrix, l @ u))
    print("Проверка A*A^(-1) = I:", np.allclose(matrix @ inverse, np.eye(matrix.shape[0])))
    print("Норма невязки:", residualNorm(matrix, x, b))


def runTridiagonalExample():
    printSection("1.2 Метод прогонки")
    fullMatrix = readMatrixFromFile(DATA_DIR / "lab1_2.txt")
    upperDiag, mainDiag, lowerDiag, ansCol = readTridiagonalMatrixFromFile(
        DATA_DIR / "lab1_2.txt"
    )
    p, q, x = solveTridiagonal(upperDiag, mainDiag, lowerDiag, ansCol)
    matrix = tridiagonalToMatrix(lowerDiag, mainDiag, upperDiag)

    print("Расширенная матрица [A | b]:")
    print(fullMatrix)
    print("Коэффициенты прогонки:")
    for i in range(len(p)):
        print(f"P={p[i]:.6f} Q={q[i]:.6f}")
    print("Решение Ax = b:", x)
    print("Норма невязки:", residualNorm(matrix, x, ansCol))


def runIterationExample():
    printSection("1.3 Метод простых итераций и метод Зейделя")
    fullMatrix = readMatrixFromFile(DATA_DIR / "lab1_3.txt")
    matrix, b = splitAugmentedMatrix(fullMatrix)
    eps = 0.01

    transposed = transpose(matrix)
    symmetricMatrix = multiplyMatrices(transposed, matrix)
    symmetricB = transposed @ b

    exact = solveGauss(matrix, b)
    iterationX, iterationCount, iterationHistory = solveIteration(
        symmetricMatrix, symmetricB, eps
    )
    seidelX, seidelCount, seidelHistory = solveSeidel(
        symmetricMatrix, symmetricB, eps
    )

    print("Расширенная матрица [A | b]:")
    print(fullMatrix)
    print("Точность:", eps)
    print("Эталонное решение:", exact)
    print("Метод простых итераций:", iterationX)
    print("Количество итераций:", iterationCount)
    print("Последняя оценка ошибки:", iterationHistory[-1])
    print("Метод Зейделя:", seidelX)
    print("Количество итераций:", seidelCount)
    print("Последняя оценка ошибки:", seidelHistory[-1])
    if seidelCount < iterationCount:
        print("Метод Зейделя быстрее на", iterationCount - seidelCount, "итераций")
    elif iterationCount < seidelCount:
        print("Метод простых итераций быстрее на", seidelCount - iterationCount, "итераций")
    else:
        print("Методы выполнили одинаковое количество итераций")


def runJacobiExample():
    printSection("1.4 Метод вращений Якоби")
    matrix = readMatrixFromFile(DATA_DIR / "lab1_4.txt")
    eps = 0.01
    eigenvalues, eigenvectors, iterations, history = JacobiRotationMethod(matrix, eps)

    print("Матрица A из файла:")
    print(matrix)
    print("Точность:", eps)
    print("Собственные значения:", eigenvalues)
    print("Собственные векторы:")
    print(eigenvectors)
    print("Количество итераций:", iterations)
    print("Последняя оценка ошибки:", history[-1])
    print(
        "Проверка A*V = V*Lambda:",
        np.allclose(matrix @ eigenvectors, eigenvectors @ np.diag(eigenvalues), atol=eps),
    )


def runQRExample():
    printSection("1.5 QR-разложение и QR-алгоритм")
    matrix = readMatrixFromFile(DATA_DIR / "lab1_5.txt")
    complexExample = readMatrixFromFile(DATA_DIR / "lab1_5_complex.txt")
    eps = 0.01

    q, r = QRDecomposition(matrix)
    eigenvalues, iterations, history = FindEigenvaluesQR(matrix, eps)
    complexEigenvalues, _, _ = FindEigenvaluesQR(complexExample, eps)

    print("Матрица A из файла:")
    print(matrix)
    print("Q:")
    print(q)
    print("R:")
    print(r)
    print("Проверка Q*R = A:", np.allclose(q @ r, matrix))
    print("Собственные значения:", eigenvalues)
    print("Количество QR-итераций:", iterations)
    print("Последняя оценка ошибки:", history[-1])
    print("Матрица с комплексными собственными значениями из файла:")
    print(complexExample)
    print("Собственные значения:", complexEigenvalues)


def main():
    np.set_printoptions(precision=6, suppress=True)
    runLUExample()
    runTridiagonalExample()
    runIterationExample()
    runJacobiExample()
    runQRExample()


if __name__ == "__main__":
    main()
