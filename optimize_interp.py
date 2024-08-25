import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import json

# Загрузка данных из CSV файла с использованием numpy
data = np.loadtxt("instantaneous_frequency.csv", delimiter=",")

# Переменная для хранения мгновенной частоты
y = data
x = np.arange(len(y))

# Список для хранения корреляций и аппроксимаций
correlations = []
approximations = []

# Перебираем количество узлов интерполяции от 3 до 32
r = range(3, 33)
for num_knots in r:
    # Выбираем узлы для линейной интерполяции
    knots = np.linspace(x.min(), x.max(), num=num_knots)
    
    # Создаем линейную интерполяцию на основе узлов
    y_linear_pred = np.interp(x, knots, np.interp(knots, x, y))
    
    approximations.append(y_linear_pred)

    # Вычисляем корреляцию между оригинальным сигналом и интерполированным
    corr = correlate(y, y_linear_pred, mode="valid")
    correlation = np.max(corr)
    correlations.append(correlation)

# Находим максимальную корреляцию и соответствующее ей количество узлов
max_correlation = max(correlations)
best_knots = r[correlations.index(max_correlation)]
best_approximation = approximations[correlations.index(max_correlation)]

# Сохранение параметров линейной интерполяции в файл
interpolation_params = {
    "knots": np.linspace(x.min(), x.max(), num=best_knots).tolist(),
    # Коэффициенты не применимы для линейной интерполяции
}

with open("interpolation_params.json", "w") as f:
    json.dump(interpolation_params, f)

# Построим график зависимости корреляции от количества узлов интерполяции
plt.figure(figsize=(10, 6))
plt.plot(r, correlations, marker="o")
plt.title("Зависимость корреляции от количества узлов интерполяции")
plt.xlabel("Количество узлов")
plt.ylabel("Корреляция")
plt.grid(True)

# Визуализация всех полученных кривых на одном графике
plt.figure(figsize=(12, 8))
plt.plot(x, y, label="Исходная мгновенная частота", color="black", linewidth=2)

plt.plot(x, best_approximation, label=f"Линейная интерполяция с {best_knots} узлами", color="red", linestyle="--")

plt.title("Аппроксимация мгновенной частоты линейной интерполяцией с наилучшим количеством узлов")

plt.xlabel("Индекс")
plt.ylabel("Частота")
plt.legend()
plt.grid(True)
plt.show()

# Вывод параметров лучшей интерполяции
print(f"Оптимальное количество узлов: {best_knots}")
print(f"Максимальная корреляция: {max_correlation:.4f}")
print(f"Позиции узлов: {np.linspace(x.min(), x.max(), num=best_knots)}")
