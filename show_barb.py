import os
import matplotlib.pyplot as plt
from barbutils import generate_barb, load_barb
import numpy as np
import fcwt
import sys

# sys.path.insert(1, os.path.join(sys.path[0], ".."))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import signal_helper as sh
script_path = os.path.dirname(os.path.realpath(__file__))

# Чтение данных из файла
filename = "1834cs1.barb"

f0 = 18000  # lowest frequency
f1 = 34000  # highest frequency
fn = 200  # number of frequencies
window_duration = 0.0002
threshold_avg = 0.01  # Threshold value for detection
threshold = 0.3  # Threshold value for noise

with open(os.path.join(script_path, filename), "rb") as f:
    barb = f.read()

sample_rate, signal_data = load_barb(barb)
sample_rate = 1e6  # sample_rate
print(f"Sample rate: {sample_rate}")
print(f"Lenght of signal: {len(signal_data)}")

intervals = sh.find_chirp_intervals(
    signal_data, int(sample_rate), f0, f1, window_duration, fn, threshold_avg, threshold
)

freqs, out = fcwt.cwt(signal_data, int(sample_rate), f0, f1, fn)

# print(intervals)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
time = np.arange(len(signal_data)) / sample_rate
max_signal = 2
lightred = "#FF7F7F"

print(f"Number of intervals: {len(intervals)}")
print(f"Intervals: {intervals[-1]}")

# Построение графика сигнала с вертикальными линиями интервалов
ax1.plot(time, signal_data)
# for interval in intervals:
#     ax1.axvline(x=interval, color='r', linestyle='--')
for start, end in intervals:
    ax1.axvline(start, color="red", linestyle="--")
    ax1.axvline(end, color="green", linestyle="--")
    ax1.fill_betweenx(y=[-max_signal, max_signal], x1=start, x2=end, color=lightred, alpha=0.3)
ax1.set_title(f"Signal from {filename} file")
ax1.set_xlabel("Sample Number")
ax1.set_ylabel("Signal Amplitude")
ax1.grid(True)

# Построение графика вейвлет-преобразования
ax2.imshow(np.abs(out), aspect="auto", extent=[0, len(signal_data), f0, f1])
ax2.set_title("Wavelet Transform")
ax2.set_xlabel("Sample Number")
ax2.set_ylabel("Frequency (Hz)")
fig.colorbar(
    ax2.imshow(np.abs(out), aspect="auto", extent=[0, len(signal_data), f0, f1]), ax=ax2
)

# Отображение графиков
plt.tight_layout()
plt.show()
