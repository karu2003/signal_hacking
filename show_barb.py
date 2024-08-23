import os
import matplotlib.pyplot as plt
from barbutils import generate_barb, load_barb
import numpy as np
import fcwt


# Helper functions
def threshold_2Darray_Level(in_array, threshold):
    in_array[in_array < threshold] = 0
    return in_array


def detect_chirp_in_window(
    window, fs, f0, f1, num_coeffs=100, threshold_avg=0.02, threshold=0.1
):
    freqs, cwt_matrix = fcwt.cwt(window, fs, f0, f1, num_coeffs)
    cwt_matrix = threshold_2Darray_Level(cwt_matrix, threshold)
    magnitude = np.abs(cwt_matrix) ** 2
    avg_magnitude = np.average(magnitude, axis=0)
    indices = np.where(avg_magnitude > threshold_avg)[0]
    if len(indices) > 20:
        indices = [indices[0], indices[-1]]
    else:
        indices = []
    chirp_detected = len(indices) > 0

    # plt.figure(figsize=(10, 4))
    # plt.plot(np.abs(magnitude), label='Signal Magnitude')
    # plt.show()

    return chirp_detected, indices, avg_magnitude


def find_chirp_intervals(
    signal,
    fs,
    f0,
    f1,
    window_duration=0.004,
    num_coeffs=100,
    threshold_avg=0.02,
    threshold=0.9,
):
    window_size = int(window_duration * fs)
    intervals = []
    in_chirp = False
    chirp_start = None
    for i in range(0, len(signal) - window_size + 1, window_size // 2):
        window = signal[i : i + window_size]
        detected, indices, _ = detect_chirp_in_window(
            window, fs, f0, f1, num_coeffs, threshold_avg, threshold
        )
        if detected:
            if not in_chirp:
                chirp_start = (i + indices[0]) / fs
                in_chirp = True
            chirp_end = (i + indices[-1] + 1) / fs
        else:
            if in_chirp:
                intervals.append((chirp_start, chirp_end))
                in_chirp = False
    if in_chirp:
        intervals.append((chirp_start, chirp_end))
    return intervals


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

intervals = find_chirp_intervals(
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
