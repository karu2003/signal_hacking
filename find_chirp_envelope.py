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

# Пример использования
script_path = os.path.dirname(os.path.realpath(__file__))
# filename = "1834cs1.barb"
# filename = "barb/1707cs1.barb"
filename = "resampled_signal.wav"

f0 = 7000  # Начальная частота
f1 = 17000  # Конечная частота
fn = 200  # Количество коэффициентов
threshold = 0.6  # Пороговое значение для огибающей CWT
threshold_p = 0.2

type_f = None

file_path = os.path.join(script_path, filename)

# Проверка расширения файла и загрузка данных
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

print(f"Частота дискретизации: {sample_rate}")
print(f"Длина сигнала: {len(signal_data)}")

# Найти индексы, где сигнал переходит через ноль
zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]

print(f"Количество переходов через ноль: {len(zero_crossings)}")

# Определить начало первого пакета (первый переход через ноль)
# if type_f == "wav":
threshold_amplitude = 0.01  # Пороговое значение амплитуды для определения паузы
#     start_index = 0

for i in range(len(signal_data)):
    if np.abs(signal_data[[i]]) > threshold_amplitude:
        start_index = i
        break

# else:
#     start_index = zero_crossings[0]

print(f"Начальный индекс сигнала после исключения паузы: {start_index}")

# Определить конец первого пакета (последний переход через ноль перед паузой)
# Предполагаем, что пауза определяется отсутствием переходов через ноль в течение определенного времени
if type_f == "barb":
    cross_zero = 90
else:
    cross_zero = 60

for i in range(1, len(zero_crossings)):
    if (
        zero_crossings[i] - zero_crossings[i - 1] > cross_zero
    ):  # 10 - это пример порога, определяющего паузу
        end_index = zero_crossings[i - 1]
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
fig, ax = plt.subplots(5, 1, figsize=(12, 10))

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


# phase = np.angle(out)
# ax[3].imshow(phase, extent=[0, len(first_frame), f0, f1],
#            aspect='auto', cmap='jet')
# ax[3].set_title("Фаза вейвлет-преобразования")

t = np.linspace(0, pulse_widths[0], len(first_frame))
initial_guess = [1, 1, 0, 1, 2, 0, 0]

instantaneous_frequency = sh.cwt_instantaneous_frequency(out, freqs)
popt_linear, _ = curve_fit(sh.linear_model, t, instantaneous_frequency)
# popt_poly, _ = curve_fit(sh.polynomial_model, t, instantaneous_frequency)
# popt_poly4, _ = curve_fit(sh.polynomial_model4, t, instantaneous_frequency)
# popt_poly6, _ = curve_fit(sh.polynomial_model6, t, instantaneous_frequency)
# params, popt_sin = curve_fit(
#     sh.sinusoidal_model_2nd_order,
#     t,
#     instantaneous_frequency,
#     p0=initial_guess,
# )

degree = 32
poly_fit = Polynomial.fit(t, instantaneous_frequency, degree)
poly_coefficients = poly_fit.convert().coef

poly_file = f"{int(sample_rate)}_poly_coefficients.txt"
np.savetxt(poly_file, poly_coefficients)


ax[3].plot(t, instantaneous_frequency, label="Мгновенная частота", color="blue")
# ax[3].plot(
#     t,
#     sh.sinusoidal_model_2nd_order(t, *popt_sin),ispol
#     label="Sinus модель",
#     linestyle="--",
#     color="red",
# )
ax[3].plot(
    t,
    # sh.polynomial_model6(t, *popt_poly6),
    poly_fit(t),
    label="Полиномиальная модель",
    linestyle="--",
    color="green",
)

ax[3].set_title("Закон изменения частоты в чирп-сигнале (CWT-анализ)")
ax[3].set_xlabel("Время (с)")
ax[3].set_ylabel("Частота (Гц)")
ax[3].legend()
ax[3].grid(True)

# print(
#     "Линейная модель: f(t) = {:.2f} + {:.2f} * t".format(popt_linear[0], popt_linear[1])
# )
# print(
#     "Полиномиальная модель: f(t) = {:.2f} + {:.2f} * t + {:.2f} * t^2".format(
#         popt_poly[0], popt_poly[1], popt_poly[2]
#     )
# )

# print(
#     "Полиномиальная модель 4: f(t) = {:.2f} + {:.2f} * t + {:.2f} * t^2 + {:.2f} * t^3 + {:.2f} * t^4".format(
#         popt_poly4[0], popt_poly4[1], popt_poly4[2], popt_poly4[3], popt_poly4[4]
#     )
# )
# print(
#     "Полиномиальная модель 6: f(t) = {:.2f} + {:.2f} * t + {:.2f} * t^2 + {:.2f} * t^3 + {:.2f} * t^4 + {:.2f} * t^5 + {:.2f} * t^6".format(
#         popt_poly6[0],
#         popt_poly6[1],
#         popt_poly6[2],
#         popt_poly6[3],
#         popt_poly6[4],
#         popt_poly6[5],
#         popt_poly6[6],
#     )
# )

# print("Коэффициенты полинома:", poly_fit.convert().coef)

T = pulse_widths[0]  # Длительность сигнала, секунды
t = np.linspace(0, T, int(T * sample_rate))  # Временная шкала
# popt_poly = [15121.45, -385326.34, 402286.82]
# freq_t = sh.polynomial_model(t, *popt_poly)
# freq_t = sh.polynomial_model4(t, *popt_poly4)
# freq_t = sh.polynomial_model6(t, *popt_poly6)
freq_t = poly_fit(t)

# Ограничим частоту диапазоном от 7 до 17 кГц
# freq_t = np.clip(freq_t, 7e3, 17e3)

# Задать начальную фазу
initial_phase = sh.convert_phase_to_radians(180)  # Например, 0 радиан

# Вычисляем фазу сигнала как интеграл от частоты с учетом начальной фазы
phase_t = 2 * np.pi * np.cumsum(freq_t) / sample_rate + initial_phase

# Генерация чирп-сигнала с использованием фазы
synthesized_chirp = np.sin(phase_t)

ax[4].plot(t, synthesized_chirp)

print(f"Длина сигнала: {len(synthesized_chirp)}")
print(f"Длина первого пакета: {len(first_frame)}")

# Проверка длины массивов
len_first_frame = len(first_frame)
len_synthesized_chirp = len(synthesized_chirp)

# Обрезка массивов до одинаковой длины
min_length = min(len_first_frame, len_synthesized_chirp)
first_frame = first_frame[:min_length]
synthesized_chirp = synthesized_chirp[:min_length]

# Корреляция между оригинальным и синтезированным сигналами
correlation = correlate(first_frame, synthesized_chirp[: len(first_frame)], mode="full")
lag = np.arange(-len(first_frame) + 1, len(first_frame))

# Нормализуем корреляцию
# correlation = correlation / np.max(np.abs(correlation))

# CWT анализ синтезированного чирпа
freqs, cwt_out = fcwt.cwt(synthesized_chirp, int(sample_rate), f0, f1, fn)
cwt_magnitude = np.abs(cwt_out)

fig1, axx = plt.subplots(3, 1, figsize=(12, 10))

# 1. Корреляция
axx[0].plot(lag, correlation, label="Корреляция", color="purple")
axx[0].set_title("Корреляция между оригинальным и синтезированным сигналами")
axx[0].set_xlabel("Задержка (samples)")
axx[0].set_ylabel("Нормализованная корреляция")
axx[0].grid(True)

# 2. CWT анализ оригинального сигнала
axx[1].imshow(cwt_magnitude, aspect="auto", extent=[0, T, f0, f1])  # , cmap="jet")
axx[1].set_title("CWT анализ синтезированного чирп-сигнала")
axx[1].set_xlabel("Время (samples)")
axx[1].set_ylabel("Частота (Hz)")
axx[1].invert_yaxis()

# 3. Сравнение сигналов
time = np.arange(len(first_frame)) / sample_rate
axx[2].plot(first_frame, label="Оригинальный сигнал")
axx[2].plot(
    synthesized_chirp[: len(first_frame)],
    label="Синтезированный сигнал",
    linestyle="--",
    color="orange",
)
axx[2].set_title("Сравнение оригинального и синтезированного сигналов")
axx[2].set_xlabel("Время (с)")
axx[2].set_ylabel("Амплитуда")
axx[2].legend()
axx[2].grid(True)

# Вывод максимальной корреляции
max_corr = np.max(correlation)
print(f"Максимальная нормализованная корреляция: {max_corr:.4f}")


plt.tight_layout()
plt.show()
