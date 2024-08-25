import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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

# Перебираем количество узлов сплайна от 3 до 32
r = range(3, 33)
for num_knots in r:
    # Выбираем узлы для сплайн-интерполяции
    knots = np.linspace(x.min(), x.max(), num=num_knots)
    
    # Создаем сплайн-интерполяцию на основе узлов
    cs = CubicSpline(knots, np.interp(knots, x, y))
    # cs = CubicSpline(x, y)
    
    # Оцениваем значения сплайна на всем диапазоне x
    y_spline_pred = cs(x)
    approximations.append(y_spline_pred)

    # Вычисляем корреляцию между оригинальным сигналом и интерполированным
    corr = correlate(y, y_spline_pred, mode="valid")
    correlation = np.max(corr)
    correlations.append(correlation)

# Находим максимальную корреляцию и соответствующее ей количество узлов
max_correlation = max(correlations)
best_knots = r[correlations.index(max_correlation)]
best_approximation = approximations[correlations.index(max_correlation)]

# Получаем параметры лучшего сплайна
best_knots_positions = np.linspace(x.min(), x.max(), num=best_knots)
best_spline = CubicSpline(best_knots_positions, np.interp(best_knots_positions, x, y))

# Сохранение параметров сплайна в файл
spline_params = {
    "knots": best_knots_positions.tolist(),
    "coefficients": best_spline.c.tolist()
}

with open("spline_params.json", "w") as f:
    json.dump(spline_params, f)


# Построим график зависимости корреляции от количества узлов сплайна
plt.figure(figsize=(10, 6))
plt.plot(r, correlations, marker="o")
plt.title("Зависимость корреляции от количества узлов сплайна")
plt.xlabel("Количество узлов")
plt.ylabel("Корреляция")
plt.grid(True)

# Визуализация всех полученных кривых на одном графике
plt.figure(figsize=(12, 8))
plt.plot(x, y, label="Исходная мгновенная частота", color="black", linewidth=2)

# for i, y_spline_pred in enumerate(approximations):
#     plt.plot(x, y_spline_pred, label=f"{r[i]} узлов")
# plt.title("Аппроксимация мгновенной частоты сплайнами с различным количеством узлов")
plt.plot(x, best_approximation, label=f"Сплайн с {best_knots} узлами", color="red", linestyle="--")

plt.title("Аппроксимация мгновенной частоты сплайном с наилучшим количеством узлов")

plt.xlabel("Индекс")
plt.ylabel("Частота")
plt.legend()
plt.grid(True)
plt.show()

# Вывод параметров лучшего сплайна
print(f"Оптимальное количество узлов: {best_knots}")
print(f"Максимальная корреляция: {max_correlation:.4f}")
print(f"Позиции узлов: {best_knots_positions}")
print(f"Коэффициенты сплайна: {best_spline.c}")
