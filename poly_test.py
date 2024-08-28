import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.interpolate import interp1d

# Параметры
fs = 1000  # частота дискретизации
t = np.linspace(0, 10, fs * 10)  # временная шкала
signal = np.sin(2 * np.pi * 5 * t)  # пример сигнала

# Определение полиномиальной модели 6-й степени
def polynomial_model6(t, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5 + a6 * t**6

# Коэффициенты полинома (пример)
coefficients = [0, 1, -0.1, 0.01, -0.001, 0.0001, -0.00001]

# Генерация полиномиальной частоты
f_poly = polynomial_model6(t, *coefficients)

# Построение графиков
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Чирп-сигнал')

plt.subplot(2, 1, 2)
plt.plot(t, f_poly / 1000)  # Частота в кГц для удобства
plt.title('Полиномиальная частота по времени')
plt.xlabel('Время [s]')
plt.ylabel('Частота [kHz]')

plt.tight_layout()
plt.show()

# Применение STFT для извлечения частоты
f, t_stft, Zxx = stft(signal, fs=fs, nperseg=2048)
frequencies = np.mean(np.abs(Zxx), axis=1)

# Проверка длины массивов
len_t_stft = len(t_stft)
len_frequencies = len(frequencies)

# Обрезка массивов до одинаковой длины
min_length = min(len_t_stft, len_frequencies)
t_stft = t_stft[:min_length]
frequencies = frequencies[:min_length]

# Интерполяция частоты
interp_func = interp1d(t_stft, frequencies, kind='linear', fill_value="extrapolate")
interpolated_freq = interp_func(t)

# Построение графиков
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Чирп-сигнал')

plt.subplot(2, 1, 2)
plt.plot(t, interpolated_freq / 1000)  # Частота в кГц для удобства
plt.title('Интерполированная частота по времени')
plt.xlabel('Время [s]')
plt.ylabel('Частота [kHz]')

plt.tight_layout()
plt.show()

from numpy.polynomial.polynomial import Polynomial

# Применение полиномиальной регрессии
degree = 6  # Степень полинома, можно настроить
poly_fit = Polynomial.fit(t, interpolated_freq, degree)

# Построение графиков
plt.figure(figsize=(12, 6))
plt.plot(t, interpolated_freq / 1000, label='Извлеченная частота')
plt.plot(t, poly_fit(t) / 1000, label=f'Полиномиальная подгонка (степень {degree})', linestyle='--')
plt.title('Полиномиальная подгонка частоты по времени')
plt.xlabel('Время [s]')
plt.ylabel('Частота [kHz]')
plt.legend()
plt.show()

# Коэффициенты полинома
print("Коэффициенты полинома:", poly_fit.convert().coef)
