import os
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import minimize
import matplotlib.pyplot as plt

filename = 'poly_coefficients.txt'
script_path = os.path.dirname(os.path.realpath(__file__))

# Загрузка коэффициентов полинома из файла
poly_coefficients = np.loadtxt(os.path.join(script_path, filename))

# Создание полинома с загруженными коэффициентами
polynomial = Polynomial(poly_coefficients)


# Функция для оценки разницы между исходным полиномом и чирп-моделью
def chirp_residuals(params, polynomial):
    f_0, f_1, T = params
    t = np.linspace(0, T, 16402)  # временная шкала от 0 до T
    chirp_model = f_0 + (f_1 - f_0) * t / T
    poly_values = polynomial(t)
    residuals = poly_values - chirp_model
    return np.sum(residuals**2)

# Начальные предположения для f_0, f_1 и T
fs = 1e6
f_0_init = polynomial(0)
f_1_init = polynomial(1)
T_init = 0.0165  # Предположительная длительность чирпа
length_s = 16402

time = time = np.arange(length_s) / fs


# Оптимизация параметров чирпа
result = minimize(chirp_residuals, [f_0_init, f_1_init, T_init], args=(polynomial,))

# Извлечение оптимальных параметров
f_0_opt, f_1_opt, T_opt = result.x

print(f"Оптимальная начальная частота f_0: {f_0_opt:.2f} Hz")
print(f"Оптимальная конечная частота f_1: {f_1_opt:.2f} Hz")
print(f"Оптимальная длительность чирпа T: {T_opt:.4f} секунд")

# Новый полином чирпа
chirp_polynomial = Polynomial([f_0_opt, (f_1_opt - f_0_opt) / T_opt])

# Создаем временную шкалу для построения графиков
t = np.linspace(0, T_opt, length_s)

# Вычисляем значения полинома и чирпа на этой временной шкале
original_poly_values = polynomial(t)
chirp_values = chirp_polynomial(t)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(t, original_poly_values, label='Оригинальный полином', color='blue')
plt.plot(t, chirp_values, label='Оптимизированный полином чирпа', color='red', linestyle='--')
plt.title('Сравнение оригинального полинома и оптимизированного полинома чирпа')
plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц)')
plt.legend()
plt.grid(True)
plt.show()
