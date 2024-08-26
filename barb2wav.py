import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import os
from barbutils import load_barb
from scipy.io.wavfile import write

# Пример использования
script_path = os.path.dirname(os.path.realpath(__file__))
filename = "barb/1707cs1.barb"

# Исходные параметрыs
original_fs = 1e6  # 1 МГц
target_fs = 750e3  # 750 кГц
# target_fs = 75000 # 75 кГц

try:
    with open(os.path.join(script_path, filename), "rb") as f:
        barb = f.read()
except FileNotFoundError:
    print(f"Файл {filename} не найден.")
    exit()

# Загрузка данных
try:
    sample_rate, signal = load_barb(barb)
    sample_rate = 1e6  # Частота дискретизации

    if sample_rate != original_fs:
        raise ValueError(
            f"Ожидаемая частота дискретизации {original_fs}, но загружена частота {sample_rate}."
        )
    if not isinstance(signal, np.ndarray):
        raise ValueError("Загруженный сигнал не является массивом NumPy.")

except (ValueError, TypeError) as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# Рассчитываем новое количество точек
num_samples = int(len(signal) * (target_fs / original_fs))

# Ресамплинг
try:
    resampled_signal = resample(signal, num_samples)
except Exception as e:
    print(f"Ошибка при ресамплинге: {e}")
    exit()


def normalize(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    else:
        return signal


# Нормализация сигналов
normalized_signal = normalize(signal)
normalized_resampled_signal = normalize(resampled_signal)

# Сохранение в WAV файлы
original_wav_file = os.path.join(script_path, "original_signal.wav")
resampled_wav_file = os.path.join(script_path, "resampled_signal.wav")

# Запись оригинального сигнала в WAV файл
write(original_wav_file, int(original_fs), (normalized_signal * 32767).astype(np.int16))
print(f"Оригинальный сигнал сохранен в {original_wav_file}")

# Запись ресамплированного сигнала в WAV файл
write(
    resampled_wav_file,
    int(target_fs),
    (normalized_resampled_signal * 32767).astype(np.int16),
)
print(f"Ресамплированный сигнал сохранен в {resampled_wav_file}")

# Временные оси для графиков
original_time = np.arange(len(signal)) / original_fs
resampled_time = np.arange(len(resampled_signal)) / target_fs

# Визуализация сигналов
plt.figure(figsize=(12, 6))

# Оригинальный сигнал
plt.subplot(2, 1, 1)
plt.plot(original_time, normalized_signal, label="Оригинальный сигнал")
plt.title("Оригинальный сигнал")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)

# Ресамплированный сигнал
plt.subplot(2, 1, 2)
plt.plot(
    resampled_time,
    normalized_resampled_signal,
    label="Ресамплированный сигнал",
    color="orange",
)
plt.title("Ресамплированный сигнал")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.grid(True)

# Показать графики
plt.tight_layout()
plt.show()