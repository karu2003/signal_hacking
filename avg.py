import libm2k
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
import signal_helper as sh
import json
import re
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import fcwt
from scipy.io import wavfile
from signal_type import signal_type


# Function to generate the signal
def generate_signal(params, polynomial_data, fs):
    f0 = params.get("f0")
    f1 = params.get("f1")
    dir = params.get("chirp_directions", -1)
    pulse_widths = params.get("pulse_widths")
    pauses = params.get("pauses")
    intervals = params.get("intervals")
    initial_phase = params.get("initial_phase")

    print(f"pulses = {pauses}")

    polynomial_type = polynomial_data.get("type", "Unknown")

    if polynomial_type == "Unknown":
        print("Unknown polynomial type")
        return

    x = np.linspace(0, 1, int(pulse_widths[0] * fs))

    # signal.extend(np.zeros(int(fs * pauses[0][2])))

    if polynomial_type == "polynomial":
        poly = Polynomial(polynomial_data.get("coefficients"))
        y_poly_pred = poly(x)
        # plt.plot(x, y_poly_pred)
        # plt.show()
    elif polynomial_type == "spline":
        y_poly_pred = sh.create_cubic_spline(polynomial_data, x)
    elif polynomial_type == "linear":
        y_poly_pred
    elif polynomial_type == "quadratic":
        y_poly_pred
    elif polynomial_type == "cubic":
        y_poly_pred
    elif polynomial_type == "hermite":
        y_poly_pred = sh.create_hermite_spline(polynomial_data, x)

    signal = []
    # y_poly_pred = poly(x)
    mapped_poly_freq = sh.map_values_tb(y_poly_pred, f0, f1, reverse=True)
    synthesized_chirp = sh.freq_to_chirp(mapped_poly_freq, fs, initial_phase)

    signal.extend(synthesized_chirp)
    signal.extend(np.zeros(int(fs * pauses[0])))

    # Second and third chirps
    for _ in range(2):
        signal.extend(sh.generate_chirp(f1, f0, pulse_widths[1], fs))
        signal.extend(np.zeros(int(fs * pauses[1])))

    for _ in range(intervals - 3):
        signal.extend(sh.generate_chirp(f1, f0, pulse_widths[3]/(intervals - 3), fs))

    # signal.extend(np.zeros(int(fs * pauses[2])))

    return np.array(signal)


# sampling rate (must be 750, 7500, 75000, 750000, 7500000, 75000000)
fs = 750000.0  # Sampling frequency
fn = 200
window = 100

# signal_type = "1834cs1"
# signal_type = "1707cs1"
match = re.search(r"(\d{2})(\d{2})", signal_type)
f1 = int(match.group(1)) * 1000  # Первая часть числа
f0 = int(match.group(2)) * 1000  # Вторая часть числа
print(f"f1 = {f1} Гц, f0 = {f0} Гц")

with open(f"params/{signal_type}_signal_params.json", "r") as json_file:
    params = json.load(json_file)

# Загрузка параметров полинома из файла JSON
with open(f"poly/{signal_type}_params.json", "r") as f:
    polynomial_data = json.load(f)

if polynomial_data.get("type") == "polynomial":
    degree = polynomial_data.get("degree")
    coefficients = polynomial_data.get("coefficients")
    if len(coefficients) != degree + 1:
        raise ValueError(
            "Количество коэффициентов не соответствует заявленной степени."
        )


resampled_filename = f"wav/{signal_type}_resampled.wav"
org_fs, org_signal = wavfile.read(resampled_filename)

org_signal = org_signal / np.max(np.abs(org_signal))
signal = generate_signal(params, polynomial_data, fs)

# pauses = params.get("pauses")

# jump = int((pauses[0][0] * fs) - window / 2)

# org_signal = org_signal[0: window]
# signal = signal[0: window]

# if len(org_signal) != len(signal):
#     min_length = min(len(org_signal), len(signal))
#     org_signal = org_signal[:min_length]
#     signal = signal[:min_length]

# freqs, out = fcwt.cwt(signal, int(fs), f0, f1, fn)
# out_magnitude = np.abs(out)
# fig, axs = plt.subplots(2, 1, figsize=(12, 8))
# t = np.arange(len(signal)) / fs
# axs[0].plot(signal, label="Synthesized signal")
# axs[0].plot(org_signal, linestyle="--", color="orange", label="Resampled signal")
# axs[1].imshow(out_magnitude, extent=[0, len(signal) / fs, f0, f1], aspect="auto")
# plt.show()
# exit()

ctx = libm2k.m2kOpen()
if ctx is None:
    print("Connection Error: No ADALM2000 device available/connected to your PC.")
    exit(1)

ain = ctx.getAnalogIn()
aout = ctx.getAnalogOut()
trig = ain.getTrigger()

# Prevent bad initial config for ADC and DAC
ain.reset()
aout.reset()

ctx.calibrateADC()
ctx.calibrateDAC()

# Normalize the signal to the range [-1, 1]
buffer2 = signal / np.max(np.abs(signal))
samples = len(signal)

# Вычисление длительности сигнала
duration = samples / fs
print(f"Длительность сигнала: {duration} секунд")

aout.setSampleRate(0, fs)
aout.setSampleRate(1, fs)
aout.enableChannel(0, True)
aout.enableChannel(1, True)

time_array = np.linspace(0, 1, samples)

rms_value = np.sqrt(np.mean(buffer2**2))
target_rms = 0.045  # 100 мВ
scaling_factor = target_rms / rms_value
buffer2 *= scaling_factor


aout.setCyclic(True)
aout.push(0, buffer2)

# libm2k.contextClose(ctx)
