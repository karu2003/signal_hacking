import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from barbutils import load_barb
import fcwt


def calculate_cwt_envelope(signal, fs, f0, f1, fn):
    # Вычислить CWT с помощью fcwt
    freqs, cwt_matrix = fcwt.cwt(signal, int(fs), f0, f1, fn)
    # Рассчитать огибающую как максимумы амплитуд по частотам
    cwt_envelope = np.max(np.abs(cwt_matrix), axis=0)
    return cwt_envelope


def find_pulse_widths(envelope, fs, threshold):
    # Преобразовать огибающую в бинарный вид по порогу
    binary_envelope = envelope > threshold
    # Найти изменения (переключения) в бинарной огибающей
    changes = np.diff(binary_envelope.astype(int))
    # Найти начала и окончания импульсов
    pulse_starts = np.where(changes == 1)[0] + 1
    pulse_ends = np.where(changes == -1)[0] + 1
    if binary_envelope[0]:
        pulse_starts = np.insert(pulse_starts, 0, 0)
    if binary_envelope[-1]:
        pulse_ends = np.append(pulse_ends, len(envelope))
    # Рассчитать ширину импульсов
    pulse_widths = (pulse_ends - pulse_starts) / fs
    intervals = list(zip(pulse_starts / fs, pulse_ends / fs))
    return intervals, pulse_widths


def find_pauses(cwt_envelope, fs, threshold):
    """
    Находит паузы в сигнале по уровню cwt_envelope.

    Parameters:
    cwt_envelope (numpy.ndarray): Массив значений cwt_envelope.
    fs (float): Частота дискретизации.
    threshold (float): Пороговое значение для определения пауз.

    Returns:
    list: Список кортежей, где каждый кортеж содержит начало, конец и длину паузы в секундах.
    """
    pauses = []
    in_pause = False
    start = 0

    for i, value in enumerate(cwt_envelope):
        if value < threshold and not in_pause:
            in_pause = True
            start = i
        elif value >= threshold and in_pause:
            in_pause = False
            end = i
            pause_start_time = start / fs
            pause_end_time = end / fs
            pause_length = pause_end_time - pause_start_time
            pauses.append((pause_start_time, pause_end_time, pause_length))

    # Если сигнал заканчивается на паузе
    if in_pause:
        end = len(cwt_envelope)
        pause_start_time = start / fs
        pause_end_time = end / fs
        pause_length = pause_end_time - pause_start_time
        pauses.append((pause_start_time, pause_end_time, pause_length))

    return pauses


# Пример использования
script_path = os.path.dirname(os.path.realpath(__file__))
filename = "1834cs1.barb"

f0 = 18000  # Начальная частота
f1 = 34000  # Конечная частота
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
cwt_envelope = calculate_cwt_envelope(signal_data, sample_rate, f0, f1, fn)

pauses = find_pauses(cwt_envelope, sample_rate, threshold_p)
print(f"Количество пауз: {len(pauses)}")
print(f"Паузы: {pauses}")

# Найти ширину импульсов огибающей CWT
intervals, pulse_widths = find_pulse_widths(cwt_envelope, sample_rate, threshold)
print(f"Количество интервалов: {len(intervals)}")
print(f"Ширина импульсов: {[f'{width:.3f}' for width in pulse_widths]}")


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
