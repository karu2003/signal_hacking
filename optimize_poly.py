import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.io import wavfile
import signal_helper as sh
import re
import json
from signal_type import signal_type

# signal_type = "1834cs1"
# signal_type = "1707cs1"
match = re.search(r"(\d{2})(\d{2})", signal_type)
f1 = int(match.group(1)) * 1000  # Первая часть числа
f0 = int(match.group(2)) * 1000  # Вторая часть числа
print(f"f1 = {f1} Гц, f0 = {f0} Гц")

# Загрузка данных из CSV файла с использованием numpy
data = np.loadtxt(f"instan/{signal_type}_map_instantaneous_freq.csv", delimiter=",")
sample_rate, signal = wavfile.read(f"wav/{signal_type}_first_frame.wav")
signal = sh.normalize(signal)

print(f"Частота дискретизации: {sample_rate}")
print(f"Длина сигнала: {len(signal)}")
print(f"Длина Instantaneous: {len(data)}")

with open(f"params/{signal_type}_signal_params.json", "r") as json_file:
    params = json.load(json_file)

initial_phase = params.get("initial_phase", 180)

# Переменная для хранения мгновенной частоты
y = data
x = np.linspace(0, 1, len(data))
# initial_phase = 180
polynomial_type = "polynomial"

# Список для хранения корреляций и аппроксимированных значений
correlations = []
approximations = []
chirps = []
r = range(1, 33)

for degree in r:
    p = Polynomial.fit(x, y, degree)
    y_poly_pred = p(x)
    approximations.append(y_poly_pred)
    mapped_poly_freq = sh.map_values_tb(y_poly_pred, f0, f1, reverse=True)
    synthesized_chirp = sh.freq_to_chirp(mapped_poly_freq, sample_rate, initial_phase)
    chirps.append(synthesized_chirp)
    max_corr, correlation, signal_lendata = sh.compute_correlation(
        signal, synthesized_chirp
    )
    correlations.append(max_corr)

# Находим максимальную корреляцию и соответствующую ей степень полинома
max_correlation = max(correlations)
best_degree = correlations.index(max_correlation) + 1
best_chirp_index = np.argmax(correlations)

# Сохраняем коэффициенты лучшего полинома в файл JSON
best_polynomial = Polynomial.fit(x, y, best_degree)
coefficients = best_polynomial.convert().coef.tolist()  # Преобразуем коэффициенты в список

spline_params = {
    "type": polynomial_type,
    "degree": best_degree,
    "coefficients": coefficients
}

with open(f"poly/{signal_type}_params.json", "w") as f:
    json.dump(spline_params, f)

print(
    f"Лучшая степень полинома: {best_degree}, Максимальная корреляция: {max_correlation:.4f}"
)

# Создаем фигуру и оси для нескольких графиков
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Построим график зависимости корреляции от степени полинома
axs[0].plot(r, correlations, marker="o")
axs[0].set_title("Зависимость корреляции от степени полинома")
axs[0].set_xlabel("Степень полинома")
axs[0].set_ylabel("Корреляция")
axs[0].grid(True)

# Построим график аппроксимации мгновенной частоты
axs[1].plot(x, y, label="Исходная мгновенная частота", color="black", linewidth=2)
axs[1].plot(
    x,
    approximations[best_degree - 1],
    label="Лучшая аппроксимация",
    linestyle="--",
    color="red",
)
axs[1].set_title("Аппроксимация мгновенной частоты полиномом")
axs[1].set_xlabel("Время (с)")
axs[1].set_ylabel("Частота (Гц)")
axs[1].grid(True)
axs[1].legend()

# Построим график оригинального и синтезированного сигналов
axs[2].plot(signal, label="Оригинальный сигнал", color="black")
axs[2].plot(
    chirps[best_chirp_index], label="Синтезированный сигнал", linestyle="--", color="red"
)
axs[2].set_title("Оригинальный и синтезированный сигналы")
axs[2].set_xlabel("Время (с)")
axs[2].set_ylabel("Амплитуда")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()