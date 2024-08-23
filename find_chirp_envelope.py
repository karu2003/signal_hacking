import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from barbutils import load_barb
import fcwt

import signal_helper as sh

# Пример использования
script_path = os.path.dirname(os.path.realpath(__file__))
# filename = "1834cs1.barb"
filename = "1707cs1.barb"

f0 = 7000 # Начальная частота
f1 = 17000  # Конечная частота
fn = 200  # Количество коэффициентов
threshold = 0.6  # Пороговое значение для огибающей CWT
threshold_p = 0.2

try:
    with open(os.path.join(script_path, filename), "rb") as f:
        barb = f.read()
except FileNotFoundError:
    print(f"Файл {filename} не найден.")
    exit()

sample_rate, signal_data = load_barb(barb)
sample_rate = 1e6  # Sampling frequency
print(f"Частота дискретизации: {sample_rate}")
print(f"Длина сигнала: {len(signal_data)}")

# Вычислить огибающую CWT
cwt_envelope = sh.calculate_cwt_envelope(signal_data, sample_rate, f0, f1, fn)

pauses = sh.find_pauses(cwt_envelope, sample_rate, threshold_p)
print(f"Количество пауз: {len(pauses)}")
print(f"Паузы: {pauses}")

# Найти ширину импульсов огибающей CWT
intervals, pulse_widths = sh.find_pulse_widths(cwt_envelope, sample_rate, threshold)
print(f"Количество интервалов: {len(intervals)}")
print(f"Ширина импульсов: {[f'{width:.4f}' for width in pulse_widths]}")


fig, ax1 = plt.subplots(figsize=(10, 4))
time = np.arange(len(signal_data)) / sample_rate
max_signal = 2
lightred = "#FF7F7F"
lightblue = "#ADD8E6"
lightgreen = "#90EE90"

# Построить график сигнала с интервалами и огибающей CWT
ax1.plot(time, signal_data, label="Сигнал")
ax1.plot(time, cwt_envelope, color="orange", linestyle="--", label="Огибающая CWT")
for (
    start,
    end,
) in intervals:
    ax1.axvline(start, color="red", linestyle="--")
    ax1.axvline(end, color="green", linestyle="--")
    ax1.fill_betweenx(
        y=[-max_signal, max_signal], x1=start, x2=end, color=lightred, alpha=0.3
    )
for start, end, _ in pauses:
    ax1.fill_betweenx(
        y=[-max_signal, max_signal], x1=start, x2=end, color=lightgreen, alpha=0.3
    )
ax1.set_title(f"Сигнал из файла {filename}")
ax1.set_xlabel("Время (с)")
ax1.set_ylabel("Амплитуда сигнала")
ax1.grid(True)
ax1.legend(loc="upper right")

plt.tight_layout()
plt.show()
