import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Параметры для генерации чирп-сигнала
duration = 1.0  # Длительность сигнала в секундах
sample_rate = 1e6  # Частота дискретизации
t = np.linspace(0, duration, int(sample_rate * duration))
from scipy.optimize import curve_fit

# Полиномиальные коэффициенты для закона изменения частоты
# Пример: f(t) = a0 + a1*t + a2*t^2 + ... + an*t^n
# Здесь мы хотим, чтобы частота изменялась от 17000 Гц до 7000 Гц
# Используем полином 32-го порядка
degree = 32
poly_coefficients = np.random.uniform(-1, 1, degree + 1)  # Пример случайных коэффициентов

# Функция для генерации частоты в зависимости от времени
def polynomial_frequency(t, coefficients):
    return Polynomial(coefficients)(t)

# Генерация частоты в зависимости от времени
instantaneous_frequency = polynomial_frequency(t, poly_coefficients)

# Нормализация частоты для изменения от 17000 Гц до 7000 Гц
instantaneous_frequency = 17000 + (7000 - 17000) * (instantaneous_frequency - np.min(instantaneous_frequency)) / (np.max(instantaneous_frequency) - np.min(instantaneous_frequency))

# Генерация чирп-сигнала
chirp_signal = np.sin(2 * np.pi * np.cumsum(instantaneous_frequency) / sample_rate)

# Полиномиальная аппроксимация
degree = 32  # Степень полинома
poly_fit = Polynomial.fit(t, instantaneous_frequency, degree)
poly_coefficients = poly_fit.convert().coef

# Восстановленная функция мгновенной частоты
def restored_frequency(t, coefficients):
    return Polynomial(coefficients)(t)

restored_instantaneous_frequency = restored_frequency(t, poly_coefficients)

# Визуализация исходной и восстановленной мгновенной частоты
plt.figure(figsize=(12, 6))
plt.plot(t, instantaneous_frequency, label='Исходная мгновенная частота', color='blue')
plt.plot(t, restored_instantaneous_frequency, label='Восстановленная мгновенная частота', linestyle='--', color='red')
plt.title('Сравнение исходной и восстановленной мгновенной частоты чирп-сигнала')
plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц)')
plt.legend()
plt.grid(True)
plt.show()