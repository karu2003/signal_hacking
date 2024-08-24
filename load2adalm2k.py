import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import os
from barbutils import load_barb

# Пример использования
script_path = os.path.dirname(os.path.realpath(__file__))
filename = "barb/1707cs1.barb"

# Исходные параметры
original_fs = 1e6  # 1 МГц
target_fs = 750e3  # 750 кГц

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
        raise ValueError(f"Ожидаемая частота дискретизации {original_fs}, но загружена частота {sample_rate}.")
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

# Временные оси для графиков
original_time = np.arange(len(signal)) / original_fs
resampled_time = np.arange(len(resampled_signal)) / target_fs

# Визуализация сигналов
plt.figure(figsize=(12, 6))

# Оригинальный сигнал
plt.subplot(2, 1, 1)
plt.plot(original_time, signal, label='Оригинальный сигнал')
plt.title('Оригинальный сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# Ресамплированный сигнал
plt.subplot(2, 1, 2)
plt.plot(resampled_time, resampled_signal, label='Ресамплированный сигнал', color='orange')
plt.title('Ресамплированный сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)

# Показать графики
plt.tight_layout()
plt.show()
