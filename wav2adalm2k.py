import libm2k
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.signal import chirp
from scipy.io import wavfile
import signal_helper as sh
import fcwt
from signal_type import signal_type

# sampling rate (must be 750, 7500, 75000, 750000, 7500000, 75000000)
fs = 75000.0  # Sampling frequency
target_rms = 0.1/2  # 100 мВ

resampled_filename = f"wav/{signal_type}_resampled.wav"
sample_rate, data = wavfile.read(resampled_filename)


if sample_rate != fs:
    print(f"Error: Sample rate of the WAV file ({sample_rate} Hz) does not match the device sample rate ({fs} Hz).")
    exit(1)

print(f"Data length: {len(data)}")
data = data.astype(np.float64)
rms_value = np.sqrt(np.mean(data**2))
scaling_factor = target_rms / rms_value
data *= scaling_factor

ctx = libm2k.m2kOpen()
if ctx is None:
    print("Connection Error: No ADALM2000 device available/connected to your PC.")
    exit(1)

ain = ctx.getAnalogIn()
aout = ctx.getAnalogOut()
trig = ain.getTrigger()

ain.reset()
aout.reset()

ctx.calibrateADC()
ctx.calibrateDAC()

aout.setSampleRate(0, fs)
aout.setSampleRate(1, fs)
aout.enableChannel(0, True)
aout.enableChannel(1, True)

buffer = data.tolist()

aout.setCyclic(True)
aout.push(0, buffer)
aout.push(1, buffer)

# # Остановка генерации сигнала
# aout.stopChannel(0)

# # Закрытие соединения
# libm2k.contextClose(ctx)
