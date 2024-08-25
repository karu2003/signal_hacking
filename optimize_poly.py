import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.signal import correlate

# Загрузка данных из CSV файла с использованием numpy
data = np.loadtxt("instantaneous_frequency.csv", delimiter=",")

# Переменная для хранения мгновенной частоты
y = data
x = np.arange(len(y))

# Список для хранения корреляций и аппроксимированных значений
correlations = []
approximations = []
r = range(1, 33)

# Перебираем степени полинома от 20 до 33
for degree in r:
    # Строим полином степени `degree`
    p = Polynomial.fit(x, y, degree)

    # Оцениваем значения полинома
    y_poly_pred = p(x)
    approximations.append(y_poly_pred)

    # Вычисляем корреляцию между оригинальным сигналом и аппроксимированным
    corr = correlate(y, y_poly_pred, mode="valid")
    correlation = np.max(corr)
    correlations.append(correlation)

# Находим максимальную корреляцию и соответствующую ей степень полинома
max_correlation = max(correlations)
best_degree = correlations.index(max_correlation) + 1

# Построим график зависимости корреляции от степени полинома
plt.figure(figsize=(10, 6))
plt.plot(r, correlations, marker="o")
plt.title("Зависимость корреляции от степени полинома")
plt.xlabel("Степень полинома")
plt.ylabel("Корреляция")
plt.grid(True)
# plt.show()

# Визуализация всех полученных кривых на одном графике
plt.figure(figsize=(12, 8))
plt.plot(x, y, label="Исходная мгновенная частота", color="black", linewidth=2)

for i, y_poly_pred in enumerate(approximations):
    plt.plot(x, y_poly_pred, label=f"Полином {r[i]} степени")

plt.title("Аппроксимация мгновенной частоты полиномами различных степеней")
plt.xlabel("Индекс")
plt.ylabel("Частота (Гц)")
# plt.legend()
plt.grid(True)


print(
    f"Лучшая степень полинома: {best_degree}, Максимальная корреляция: {max_correlation:.4f}"
)

plt.show()
