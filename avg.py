import libm2k
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.signal import chirp
import fcwt

# Signal parameters
# sampling rate (must be 750, 7500, 75000, 750000, 7500000, 75000000)
fs = 750000.0  # Sampling frequency
t1 = 0.0165  # Chirp duration 4 ms
t2 = 0.002  # Chirp duration 2 ms
pause = 0.008  # Pause 8 ms
f0 = 17000  # Start frequency 18 kHz
f1 = 7000  # End frequency 34 kHz
num_sine_periods = 7.19
amp_reduction_factor = 0.33
import signal_helper as sh


# Function to generate the signal
def generate_signal(
    fs, f0, f1, t1, t2, pause, num_sine_periods=8, amp_reduction_factor=0.2
):
    signal = []

    # First chirp 18-34 kHz, 4 ms, pause 8 ms
    # signal.extend(generate_chirp(f0, f1, t1, fs))
    signal.extend(
        sh.generate_sin_chirp(f1, f0, t1, fs, num_sine_periods, amp_reduction_factor, 128)
    )
    signal.extend(np.zeros(int(fs * pause)))

    # Second and third chirps 34-18 kHz, 4 ms, pause 8 ms
    for _ in range(2):
        signal.extend(sh.generate_chirp(f1, f0, t1, fs))
        signal.extend(np.zeros(int(fs * pause)))

    # Remaining chirps 34-18 kHz, 2 ms, no pause
    for _ in range(129):
        signal.extend(sh.generate_chirp(f1, f0, t2, fs))

    return np.array(signal)


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

# ctx.calibrateADC()
ctx.calibrateDAC()

# ain.enableChannel(0, True)
# ain.enableChannel(1, True)
# ain.setSampleRate(100000)
# ain.setRange(0, -10, 10)

# Генерация синусоидального сигнала
# frequency = 34000  # Частота сигнала, Гц
# amplitude = 0.1  # Амплитуда сигнала
# offset = 0.0  # Смещение сигнала
# samples = 1024    # Количество образцов
num_sine_periods = 8

# Создание массива данных для генерации сигнала
# time_array = np.linspace(0, 1, samples)
# signal = amplitude * np.sin(2 * np.pi * frequency * time_array) + offset

signal = generate_signal(
    fs, f0, f1, t1, t2, pause, num_sine_periods, amp_reduction_factor
)

# Normalize the signal to the range [-1, 1]
buffer2 = signal / np.max(np.abs(signal))
samples = len(signal)
print(samples)
# Вычисление длительности сигнала
duration = samples / fs
print(f"Длительность сигнала: {duration} секунд")


### uncomment the following block to enable triggering
# trig.setAnalogSource(0) # Channel 0 as source
# trig.setAnalogCondition(0,libm2k.RISING_EDGE_ANALOG)
# trig.setAnalogLevel(0,0.5)  # Set trigger level at 0.5
# trig.setAnalogDelay(0) # Trigger is centered
# trig.setAnalogMode(1, libm2k.ANALOG)

aout.setSampleRate(0, fs)
# aout.setSampleRate(1, fs)
aout.enableChannel(0, True)
# aout.enableChannel(1, True)
print("SampleRate: ", aout.getSampleRate(0))

# x=np.linspace(-np.pi,np.pi,1024)
buffer1 = np.linspace(-2.0, 2.00, samples)
# buffer2=np.sin(x)

time_array = np.linspace(0, 1, samples)
# buffer2 = amplitude * np.sin(2 * np.pi * frequency * time_array) + offset

# buffer = [buffer1, buffer2]

chirp1 = sh.generate_chirp(f0, f1, t1, fs)
chirp2 = sh.generate_chirp(f1, f0, t1, fs)
chirp3 = sh.generate_chirp(f1, f0, t2, fs)

rms_value = np.sqrt(np.mean(buffer2**2))
target_rms = 0.1  # 100 мВ
scaling_factor = target_rms / rms_value
buffer2 *= scaling_factor


aout.setCyclic(True)
aout.push(0, buffer2)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

freqs, cwt_matrix = fcwt.cwt(buffer2, int(fs), f0, f1, 200)

ax[1].imshow(
    np.abs(cwt_matrix),
    aspect="auto",
    interpolation="none",
    extent=[0, len(buffer2), freqs[0], freqs[-1]],
)
ax[1].set_title("Амплитуда вейвлет-преобразования")
ax[1].set_xlabel("Время (с)")
ax[1].set_ylabel("Частота (Гц)")
plt.show()
# libm2k.contextClose(ctx)
