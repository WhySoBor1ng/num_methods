# Numeric methods lab 1

Реализация всех пунктов первой лабораторной работы в стиле предоставленного
примера. Основные методы названы аналогично примеру:

1. `LUDecomposition`, `LUSolve`, `determinantLU`, `calculateInverse`.
2. `readTridiagonalMatrixFromFile`, `solveTridiagonal`.
3. `solveGauss`, `getQ`, `solveIteration`, `solveSeidel`.
4. `RotationMatrix`, `OffDiagonalNorm`, `JacobiRotationMethod`.
5. `QRDecomposition`, `Eigenvalues2x2`, `FindEigenvaluesQR`.

Матрицы для каждого пункта находятся в каталоге `data`. Для пунктов 1.1-1.3
из файлов считываются расширенные матрицы `[A | b]`; для пунктов 1.4-1.5 -
квадратные матрицы задач на собственные значения.

## Запуск

```bash
python main.py
```

Зависимость:

```bash
pip install numpy
```
