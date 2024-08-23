import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, hilbert
from scipy.optimize import curve_fit
import signal_helper as sh

# Генерация чирп-сигнала (нелинейного)
fs = 96000  # частота дискретизации
t = np.linspace(0, 1, fs)
f0 = 7000  # начальная частота
f1 = 17000  # конечная частота
signal = chirp(t, f0=f0, f1=f1, t1=t[-1], method="quadratic")


instantaneous_frequency = sh.hilbert_instantaneous_frequency(signal, t)

# Применение модели
popt_linear, _ = curve_fit(sh.linear_model, t[:-1], instantaneous_frequency)
popt_poly, _ = curve_fit(sh.polynomial_model, t[:-1], instantaneous_frequency)

# Визуализация
plt.figure(figsize=(12, 6))

plt.plot(t[:-1], instantaneous_frequency, label="Мгновенная частота", color="blue")
plt.plot(
    t[:-1],
    sh.linear_model(t[:-1], *popt_linear),
    label="Линейная модель",
    linestyle="--",
    color="red",
)
plt.plot(
    t[:-1],
    sh.polynomial_model(t[:-1], *popt_poly),
    label="Полиномиальная модель",
    linestyle="--",
    color="green",
)

plt.title("Закон изменения частоты в чирп-сигнале")
plt.xlabel("Время (с)")
plt.ylabel("Частота (Гц)")
plt.legend()
plt.grid(True)
plt.show()

print(
    "Линейная модель: f(t) = {:.2f} + {:.2f} * t".format(popt_linear[0], popt_linear[1])
)
print(
    "Полиномиальная модель: f(t) = {:.2f} + {:.2f} * t + {:.2f} * t^2".format(
        popt_poly[0], popt_poly[1], popt_poly[2]
    )
)
