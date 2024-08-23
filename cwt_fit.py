import numpy as np
import matplotlib.pyplot as plt
import fcwt
from scipy.optimize import curve_fit
from scipy.signal import chirp, hilbert

# Генерация линейного чирп-сигнала
fs = 96000 # частота дискретизации
t = np.linspace(0, 1, fs)
f0 = 7000  # начальная частота
f1 = 17000  # конечная частота
fn = 200  # количество частот
k = 200  # скорость изменения частоты (Hz/s)

signal = chirp(t, f0=f0, f1=f1, t1=t[-1], method='quadratic')
# signal = np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2))

# Применение CWT с использованием Морле-вейвлета
# scales = np.arange(1, 256)
# coef, freqs = pywt.cwt(signal, scales, 'cmor', sampling_period=1/fs)
freqs, coef = fcwt.cwt(signal, int(fs), f0, f1, fn)

# Найти частоты с максимальной мощностью на каждом временном шаге
power = np.abs(coef) ** 2
max_power_indices = np.argmax(power, axis=0)
instantaneous_frequency = freqs[max_power_indices]

# Подгонка линейной и полиномиальной модели к мгновенной частоте
def linear_model(t, f0, k):
    return f0 + k * t

def polynomial_model(t, a0, a1, a2):
    return a0 + a1 * t + a2 * t**2

# Применение модели
popt_linear, _ = curve_fit(linear_model, t, instantaneous_frequency)
popt_poly, _ = curve_fit(polynomial_model, t, instantaneous_frequency)

# Визуализация
plt.figure(figsize=(12, 6))

plt.plot(t, instantaneous_frequency, label="Мгновенная частота", color="blue")
plt.plot(t, linear_model(t, *popt_linear), label="Линейная модель", linestyle="--", color="red")
plt.plot(t, polynomial_model(t, *popt_poly), label="Полиномиальная модель", linestyle="--", color="green")

plt.title("Закон изменения частоты в чирп-сигнале (CWT-анализ)")
plt.xlabel("Время (с)")
plt.ylabel("Частота (Гц)")
plt.legend()
plt.grid(True)
plt.show()

print("Линейная модель: f(t) = {:.2f} + {:.2f} * t".format(popt_linear[0], popt_linear[1]))
print("Полиномиальная модель: f(t) = {:.2f} + {:.2f} * t + {:.2f} * t^2".format(popt_poly[0], popt_poly[1], popt_poly[2]))
