import numpy as np
import matplotlib.pyplot as plt
import fcwt
from scipy.optimize import curve_fit
from scipy.signal import chirp, hilbert
import signal_helper as sh

# Генерация линейного чирп-сигнала
fs = 96000 # частота дискретизации
t = np.linspace(0, 1, fs)
f0 = 7000  # начальная частота
f1 = 17000  # конечная частота
fn = 200  # количество частот
k = 200  # скорость изменения частоты (Hz/s)

signal = chirp(t, f0=f0, f1=f1, t1=t[-1], method='quadratic')
freqs, coef = fcwt.cwt(signal, int(fs), f0, f1, fn)

instantaneous_frequency = sh.cwt_instantaneous_frequency(coef, freqs)

# Применение модели
popt_linear, _ = curve_fit(sh.linear_model, t, instantaneous_frequency)
popt_poly, _ = curve_fit(sh.polynomial_model, t, instantaneous_frequency)

# Визуализация
plt.figure(figsize=(12, 6))

plt.plot(t, instantaneous_frequency, label="Мгновенная частота", color="blue")
plt.plot(t, sh.linear_model(t, *popt_linear), label="Линейная модель", linestyle="--", color="red")
plt.plot(t, sh.polynomial_model(t, *popt_poly), label="Полиномиальная модель", linestyle="--", color="green")

plt.title("Закон изменения частоты в чирп-сигнале (CWT-анализ)")
plt.xlabel("Время (с)")
plt.ylabel("Частота (Гц)")
plt.legend()
plt.grid(True)
plt.show()

print("Линейная модель: f(t) = {:.2f} + {:.2f} * t".format(popt_linear[0], popt_linear[1]))
print("Полиномиальная модель: f(t) = {:.2f} + {:.2f} * t + {:.2f} * t^2".format(popt_poly[0], popt_poly[1], popt_poly[2]))
