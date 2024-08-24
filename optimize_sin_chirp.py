import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, sweep_poly, correlate
from barbutils import load_barb
import fcwt
import signal_helper as sh
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d

# Пример использования
script_path = os.path.dirname(os.path.realpath(__file__))
# filename = "1834cs1.barb"
filename = "barb/1707cs1.barb"

f0 = 7000  # Начальная частота
f1 = 17000  # Конечная частота
fn = 200  # Количество коэффициентов
threshold = 0.6  # Пороговое значение для огибающей CWT
threshold_p = 0.2
num_sine_periods = 7.7
amp_reduction_factor = 0.25

try:
    with open(os.path.join(script_path, filename), "rb") as f:
        barb = f.read()
except FileNotFoundError:
    print(f"Файл {filename} не найден.")
    exit()

sample_rate, signal_data = load_barb(barb)
sample_rate = 1e6  # Sampling frequency

# Найти индексы, где сигнал переходит через ноль
zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
# Определить начало первого пакета (первый переход через ноль)
start_index = zero_crossings[0]

# Определить конец первого пакета (последний переход через ноль перед паузой)
# Предполагаем, что пауза определяется отсутствием переходов через ноль в течение определенного времени
for i in range(1, len(zero_crossings)):
    if (
        zero_crossings[i] - zero_crossings[i - 1] > 90
    ):  # 10 - это пример порога, определяющего паузу
        end_index = zero_crossings[i - 1]
        break
else:
    end_index = zero_crossings[-1]  # Если паузы нет, берем последний пере1ход

signal_width = end_index - start_index
first_frame = signal_data[start_index:end_index]
time = np.arange(len(first_frame)) / sample_rate
print("Длина первого пакета:", len(first_frame))
t1 = time[-1]
freqs, first_cwt = fcwt.cwt(first_frame, int(sample_rate), f0, f1, fn)
magnitude = np.abs(first_cwt)

synthesized_chirp = sh.generate_sin_chirp(
    f1, f0, t1, sample_rate, num_sine_periods, amp_reduction_factor
)

max_corr, correlation, c_with = sh.compute_correlation(first_frame, synthesized_chirp)
lag = np.arange(-c_with + 1, c_with)

max_corr = np.max(correlation)
# print(f"Максимальная корреляция: {max_corr:.4f}")

steps = 100

# Диапазоны значений для поиска оптимальных параметров
num_sine_periods_range = np.linspace(6.5, 10, steps)
amp_reduction_factor_range = np.linspace(0.05, 0.4, steps)

max_correlation = -np.inf
best_num_sine_periods = None
best_amp_reduction_factor = None

for num_sine_periods in num_sine_periods_range:
    for amp_reduction_factor in amp_reduction_factor_range:
        # Генерация синусоидального чирпа с текущими параметрами
        synthesized_chirp = sh.generate_sin_chirp(
            f1, f0, t1, sample_rate, num_sine_periods, amp_reduction_factor
        )

        # Вычисление корреляции
        max_corr, correlation, c_with = sh.compute_correlation(
            first_frame, synthesized_chirp
        )

        # Если текущая корреляция выше предыдущей максимальной, обновляем параметры
        if max_corr > max_correlation:
            max_correlation = max_corr
            best_num_sine_periods = num_sine_periods
            best_amp_reduction_factor = amp_reduction_factor

print(f"Лучшее значение num_sine_periods: {best_num_sine_periods}")
print(f"Лучшее значение amp_reduction_factor: {best_amp_reduction_factor}")
print(f"Максимальная корреляция: {max_correlation:.4f}")

synthesized_chirp = sh.generate_sin_chirp(
    f1, f0, t1, sample_rate, best_num_sine_periods, best_amp_reduction_factor
)

fig, ax = plt.subplots(4, 1, figsize=(12, 10))

ax[0].plot(first_frame, label="Сигнал")
ax[0].plot(
    synthesized_chirp[: len(first_frame)],
    label="Синтезированный сигнал",
    linestyle="--",
)
ax[0].set_title(f"Сигнал из файла {filename}")
ax[0].set_xlabel("Время (с)")
ax[0].set_ylabel("Амплитуда сигнала")
ax[0].grid(True)
ax[0].legend(loc="upper right")
ax[1].imshow(
    magnitude, aspect="auto", extent=[0, len(first_frame), freqs[0], freqs[-1]]
)
ax[1].set_title("Вейвлет-преобразование")

ax[2].plot(lag, correlation, label="Корреляция", color="purple")
ax[2].set_title("Корреляция между оригинальным и синтезированным сигналами")
ax[2].set_xlabel("Задержка (samples)")
ax[2].grid(True)

freqs, first_cwt = fcwt.cwt(synthesized_chirp, int(sample_rate), f0, f1, fn)
magnitude = np.abs(first_cwt)

ax[3].imshow(
    magnitude, aspect="auto", extent=[0, len(first_frame), freqs[0], freqs[-1]]
)
ax[3].set_title("Вейвлет-преобразование синтезированного сигнала")

plt.tight_layout()
plt.show()
