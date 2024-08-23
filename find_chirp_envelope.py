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

# Найти индексы, где сигнал переходит через ноль
zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]

print(f"Количество переходов через ноль: {len(zero_crossings)}")

# Определить начало первого пакета (первый переход через ноль)
start_index = zero_crossings[0]

# Определить конец первого пакета (последний переход через ноль перед паузой)
# Предполагаем, что пауза определяется отсутствием переходов через ноль в течение определенного времени
for i in range(1, len(zero_crossings)):
    if zero_crossings[i] - zero_crossings[i-1] > 90:  # 10 - это пример порога, определяющего паузу
        end_index = zero_crossings[i-1]
        break
else:
    end_index = zero_crossings[-1]  # Если паузы нет, берем последний пере1ход

# Вычисляем ширину первого пакета
signal_width = end_index - start_index

print(f"Индекс начала первого пакета: {start_index/sample_rate}")
print(f"Индекс конца первого пакета: {end_index/sample_rate}")
print(f"Ширина первого пакета: {signal_width/sample_rate} секунд")

# Вычислить огибающую CWT
cwt_envelope = sh.calculate_cwt_envelope(signal_data, sample_rate, f0, f1, fn)

pauses = sh.find_pauses(cwt_envelope, sample_rate, threshold_p)
print(f"Количество пауз: {len(pauses)}")
print(f"Паузы: {pauses}")

# Найти ширину импульсов огибающей CWT
intervals, pulse_widths = sh.find_pulse_widths(cwt_envelope, sample_rate, threshold)
print(f"Количество интервалов: {len(intervals)}")
print(f"Ширина импульсов: {[f'{width:.4f}' for width in pulse_widths]}")


# fig, ax1 = plt.subplots(figsize=(10, 4))
fig, ax = plt.subplots(4, 1, figsize=(12, 8))

time = np.arange(len(signal_data)) / sample_rate
max_signal = 2
lightred = "#FF7F7F"
lightblue = "#ADD8E6"
lightgreen = "#90EE90"

# Построить график сигнала с интервалами и огибающей CWT
ax[0].plot(time, signal_data, label="Сигнал")
ax[0].plot(time, cwt_envelope, color="orange", linestyle="--", label="Огибающая CWT")
for (
    start,
    end,
) in intervals:
    ax[0].axvline(start, color="red", linestyle="--")
    ax[0].axvline(end, color="green", linestyle="--")
    ax[0].fill_betweenx(
        y=[-max_signal, max_signal], x1=start, x2=end, color=lightred, alpha=0.3
    )
for start, end, _ in pauses:
    ax[0].fill_betweenx(
        y=[-max_signal, max_signal], x1=start, x2=end, color=lightgreen, alpha=0.3
    )
ax[0].set_title(f"Сигнал из файла {filename}")
ax[0].set_xlabel("Время (с)")
ax[0].set_ylabel("Амплитуда сигнала")
ax[0].grid(True)
ax[0].legend(loc="upper right")

first_frame = signal_data[start_index:end_index]

ax[1].plot(first_frame, label="1 Сигнал")
ax[1].set_title("Первый пакет")
ax[1].grid(True)

# Построить график вейвлет-преобразования
freqs, out = fcwt.cwt(first_frame, int(sample_rate), f0, f1, fn)
magnitude = np.abs(out)
ax[2].imshow(magnitude, aspect="auto", extent=[0, len(first_frame), f0, f1])
ax[2].set_title("Вейвлет-преобразование")

max_indexs, out_s = sh.fill_max2one(magnitude)
print("Shape of out_s:", out_s.shape)

# ax[3].imshow(np.abs(out_s), aspect="auto", extent=[0, len(first_frame), f0, f1])
# ax[3].plot(out_s, label="Максимумы")
ax[3].plot(max_indexs, label="Максимумы")

plt.tight_layout()
plt.show()
