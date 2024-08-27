import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, sweep_poly, correlate
from scipy.io import wavfile
from barbutils import load_barb
import fcwt
import signal_helper as sh
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.io import wavfile
import re
import json

script_path = os.path.dirname(os.path.realpath(__file__))
filename = "barb/1834cs1.barb"
filename = "barb/1707cs1.barb"
# filename = "wav/1707cs1_resampled.wav"

f0 = 7000  # Начальная частота
f1 = 17000  # Конечная частота
fn = 200  # Количество коэффициентов
threshold = 0.6  # Пороговое значение для огибающей CWT
threshold_p = 0.2

type_f = None

file_path = os.path.join(script_path, filename)

barb_filename = os.path.basename(filename)
frequency_str_parts = re.split(r"[._]", barb_filename)

match = re.search(r"(\d{2})(\d{2})", frequency_str_parts[0])
if match:
    f1 = int(match.group(1)) * 1000  # Первая часть числа
    f0 = int(match.group(2)) * 1000  # Вторая часть числа
    print(f"f1 = {f1} Гц, f0 = {f0} Гц")
else:
    print("Не удалось извлечь частоты из имени файла.")
    exit(1)

wav_filename = f"wav/{frequency_str_parts[0]}_first_frame.wav"

if filename.endswith(".barb"):
    try:
        with open(file_path, "rb") as f:
            barb = f.read()
        sample_rate, signal_data = load_barb(barb)
        sample_rate = 1e6  # Частота дискретизации
        type_f = "barb"
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        exit()
elif filename.endswith(".wav"):
    try:
        sample_rate, signal_data = wavfile.read(file_path)
        signal_data = sh.normalize(signal_data)
        type_f = "wav"
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        exit()
else:
    print(f"Неподдерживаемый формат файла: {filename}")
    exit()

# Найти индексы, где сигнал переходит через ноль
zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]

threshold_amplitude = 0.01  # Пороговое значение амплитуды для определения паузы

for i in range(len(signal_data)):
    if np.abs(signal_data[[i]]) > threshold_amplitude:
        start_index = i
        break

# start_index = zero_crossings[0]
# print(f"Начальный индекс сигнала после исключения паузы: {start_index}")

cross_zero = int((sample_rate / max(f0, f1)) + 5)

for i in range(1, len(zero_crossings)):
    if zero_crossings[i] - zero_crossings[i - 1] > cross_zero:
        end_index = zero_crossings[i - 1]
        break
else:
    end_index = zero_crossings[-1]  # Если паузы нет, берем последний переход

# Вычисляем ширину первого пакета
signal_width = end_index - start_index
cwt_envelope = sh.calculate_cwt_envelope(signal_data, sample_rate, f0, f1, fn)
pauses = sh.find_pauses(cwt_envelope, sample_rate, threshold_p)
intervals, pulse_widths = sh.find_pulse_widths(cwt_envelope, sample_rate, threshold)

freqs_inst, out_ints = fcwt.cwt(signal_data, int(sample_rate), f0, f1, fn)
instantaneous_frequency_full = sh.cwt_instantaneous_frequency(out_ints, freqs_inst)
chirp_directions = sh.determine_chirp_direction(intervals, sample_rate, instantaneous_frequency_full)
# print(f"Направления чирпа: {chirp_directions}")

print(f"Частота дискретизации: {sample_rate}")
print(f"Длина сигнала: {len(signal_data)}")
print(f"Количество переходов через ноль: {len(zero_crossings)}")
print(f"Индекс начала первого пакета: {start_index/sample_rate}")
print(f"Индекс конца первого пакета: {end_index/sample_rate}")
print(f"Ширина первого пакета: {signal_width/sample_rate} секунд")
print(f"Количество пауз: {len(pauses)}")
# print(f"Паузы: {pauses}")
print(f"Количество пакетов: {len(intervals)}")
# print(f"Интервалы: {intervals}")
# print(f"Ширина импульсов: {[f'{width:.4f}' for width in pulse_widths]}")

first_frame = signal_data[start_index:end_index]

# wavfile.write(wav_filename, int(sample_rate), first_frame.astype(np.int16))
wavfile.write(wav_filename, int(sample_rate), first_frame)
print(f"Первый фрейм сохранен в файл: {wav_filename}")
print(f"Длина первого фрейма: {len(first_frame)}")

freqs, out = fcwt.cwt(first_frame, int(sample_rate), f0, f1, fn)
magnitude = np.abs(out)


t = np.linspace(0, pulse_widths[0], len(first_frame))
t1 = np.linspace(0, 1, len(first_frame))

instantaneous_frequency = sh.cwt_instantaneous_frequency(out, freqs)
map_instantaneous_frequency = sh.map_values_tb(instantaneous_frequency, f0, f1)

filename_instan = f"instan/{frequency_str_parts[0]}_instantaneous_freq.csv"
filename_map_instan = f"instan/{frequency_str_parts[0]}_map_instantaneous_freq.csv"
np.savetxt(filename_instan, instantaneous_frequency, delimiter=",")
np.savetxt(filename_map_instan, map_instantaneous_frequency, delimiter=",")
# print(f"Файлы с мгновенной частотой сохранены: {filename_instan}, {filename_map_instan}")

degree = 31
poly_fit_norm = Polynomial.fit(t1, map_instantaneous_frequency, degree)
poly_freq = poly_fit_norm(t1)
mapped_poly_freq = sh.map_values_tb(poly_freq, f0, f1, reverse=True)

# Ограничим частоту диапазоном от 7 до 17 кГц
# freq_t = np.clip(freq_t, 7e3, 17e3)

if f0 < f1:
    initial_phase = 180
else:
    initial_phase = 0

synthesized_chirp = sh.freq_to_chirp(mapped_poly_freq, sample_rate, initial_phase)

# poly_fit = Polynomial.fit(t, instantaneous_frequency, degree)
# poly_coefficients = poly_fit.convert().coef
# poly_file = f"{int(sample_rate)}_poly_coefficients.txt"
# np.savetxt(poly_file, poly_coefficients)

max_corr, correlation, len_signal = sh.compute_correlation(first_frame, synthesized_chirp)
lag = np.arange(-len_signal + 1, len_signal)
print(f"Максимальная нормализованная корреляция: {max_corr:.4f}")

phase = np.angle(out)

input_signal_params = {
    "f0": f0,
    "f1": f1,
    "sample_rate": sample_rate,
    "chirp_directions": chirp_directions,
    "pulse_widths": [round(pw, 3) for pw in pulse_widths],
    "pauses": pauses,
    "intervals": len(intervals),
    "initial_phase": initial_phase,
}

params_filename = f"params/{frequency_str_parts[0]}_signal_params.json"
with open(params_filename, "w") as json_file:
    json.dump(input_signal_params, json_file, indent=4)
# print(f"Параметры входного сигнала сохранены в файл: {params_filename}")

# Нормализуем корреляцию
# correlation = correlation / np.max(np.abs(correlation))

# CWT анализ синтезированного чирпа
freqs, cwt_out = fcwt.cwt(synthesized_chirp, int(sample_rate), f0, f1, fn)
cwt_magnitude = np.abs(cwt_out)

fig, ax = plt.subplots(5, 2, figsize=(18, 12))

time = np.arange(len(signal_data)) / sample_rate
max_signal = 2
lightred = "#FF7F7F"
lightblue = "#ADD8E6"
lightgreen = "#90EE90"

# Построить график сигнала с интервалами и огибающей CWT
ax[0][0].plot(time, signal_data, label="Сигнал")
ax[0][0].plot(time, cwt_envelope, color="orange", linestyle="--", label="Огибающая CWT")
for (
    start,
    end,
) in intervals:
    ax[0][0].axvline(start, color="red", linestyle="--")
    ax[0][0].axvline(end, color="green", linestyle="--")
    ax[0][0].fill_betweenx(
        y=[-max_signal, max_signal], x1=start, x2=end, color=lightred, alpha=0.3
    )
for start, end, _ in pauses:
    ax[0][0].fill_betweenx(
        y=[-max_signal, max_signal], x1=start, x2=end, color=lightgreen, alpha=0.3
    )
ax[0][0].set_title(f"Сигнал из файла {filename}")
ax[0][0].set_xlabel("Время (с)")
ax[0][0].set_ylabel("Амплитуда сигнала")
ax[0][0].grid(True)
ax[0][0].legend(loc="upper right")

ax[1][0].plot(first_frame, label="1 Сигнал")
ax[1][0].set_title("Первый пакет")
ax[1][0].grid(True)

ax[2][0].imshow(magnitude, aspect="auto", extent=[0, len(first_frame), f0, f1])
ax[2][0].set_title("Mагнитуда частот")

ax[3][0].plot(t, instantaneous_frequency, label="Мгновенная частота", color="blue")
ax[3][0].plot(
    t,
    mapped_poly_freq,
    label="Полиномиальная модель",
    linestyle="--",
    color="green",
)

ax[3][0].set_title("Закон изменения частоты в чирп-сигнале")
ax[3][0].set_xlabel("Время (с)")
ax[3][0].set_ylabel("Частота (Гц)")
ax[3][0].legend()
ax[3][0].grid(True)

ax[4][0].plot(t1, synthesized_chirp)

# 1. Корреляция
ax[0][1].plot(lag, correlation, label="Корреляция", color="purple")
ax[0][1].set_title("Корреляция между оригинальным и синтезированным сигналами")
ax[0][1].set_xlabel("Задержка (samples)")
ax[0][1].set_ylabel("Нормализованная корреляция")
ax[0][1].grid(True)

# 2. CWT анализ оригинального сигнала
ax[1][1].imshow(cwt_magnitude, aspect="auto", extent=[0, pulse_widths[0], f0, f1])
ax[1][1].set_title("Mагнитуда частот синтезированного chirp")
ax[1][1].set_xlabel("Время (samples)")
ax[1][1].set_ylabel("Частота (Hz)")
ax[1][1].invert_yaxis()

# 3. Сравнение сигналов
time = np.arange(len(first_frame)) / sample_rate
ax[2][1].plot(first_frame, label="Оригинальный сигнал")
ax[2][1].plot(
    synthesized_chirp[: len(first_frame)],
    label="Синтезированный сигнал",
    linestyle="--",
    color="orange",
)
ax[2][1].set_title("Сравнение оригинального и синтезированного сигналов")
ax[2][1].set_xlabel("Время (с)")
ax[2][1].set_ylabel("Амплитуда")
ax[2][1].legend()
ax[2][1].grid(True)

ax[3][1].plot(instantaneous_frequency_full, label="Мгновенная частота", color="blue")

ax[4][1].imshow(phase, extent=[0, len(first_frame), f0, f1],
           aspect='auto', cmap='jet')
ax[4][1].set_title("Фаза")

plt.tight_layout()
plt.show()
