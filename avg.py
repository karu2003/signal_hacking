import libm2k
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
import signal_helper as sh
import json


# Function to generate the signal
def generate_signal(
    fs, f0, f1, t1, t2, pause, num_sine_periods=8, amp_reduction_factor=0.2
):
    signal = []

    # First chirp
    # signal.extend(generate_chirp(f0, f1, t1, fs))
    signal.extend(
        sh.generate_sin_chirp(f1, f0, t1, fs, num_sine_periods, amp_reduction_factor)
    )
    signal.extend(np.zeros(int(fs * pause)))

    # Second and third chirps
    for _ in range(2):
        signal.extend(sh.generate_chirp(f1, f0, t1, fs))
        signal.extend(np.zeros(int(fs * pause)))

    # Remaining chirps 34-18 kHz, 2 ms, no pause
    for _ in range(129):
        signal.extend(sh.generate_chirp(f1, f0, t2, fs))

    return np.array(signal)


# sampling rate (must be 750, 7500, 75000, 750000, 7500000, 75000000)
fs = 750000.0  # Sampling frequency
t1 = 0.0165  # Chirp duration
t2 = 0.002  # Chirp duration
pause = 0.0078  # Pause  ms
f0 = 17000  # Start frequency
f1 = 7000  # End frequency
num_sine_periods = 7.19
amp_reduction_factor = 0.33


# Загрузка параметров полинома из файла JSON
with open("params.json", "r") as f:
    polynomial_data = json.load(f)

polynomial_type = polynomial_data.get("type", "Unknown")
coefficients = polynomial_data.get("coefficients", [])
knots = polynomial_data.get("knots", None)

print(f"Загружен полином типа: {polynomial_type}")

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

signal = generate_signal(
    fs, f0, f1, t1, t2, pause, num_sine_periods, amp_reduction_factor
)

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
target_rms = 0.1  # 100 мВ
scaling_factor = target_rms / rms_value
buffer2 *= scaling_factor


aout.setCyclic(True)
aout.push(0, buffer2)

# libm2k.contextClose(ctx)
