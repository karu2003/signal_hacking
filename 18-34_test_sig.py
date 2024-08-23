import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.io.wavfile import write
from nptdms import TdmsWriter, ChannelObject
import fcwt

# Signal parameters
fs = 96000  # Sampling frequency
t1 = 0.004  # Chirp duration 4 ms
t2 = 0.002  # Chirp duration 2 ms
pause = 0.008  # Pause 8 ms


# Chirp generation
def generate_chirp(f0, f1, t, fs):
    t = np.linspace(0, t, int(fs * t))
    return chirp(t, f0=f0, f1=f1, t1=t[-1], method="linear")


# Function to generate the signal
def generate_signal(fs, t1, t2, pause):
    signal = []

    # First chirp 18-34 kHz, 4 ms, pause 8 ms
    signal.extend(generate_chirp(18000, 34000, t1, fs))
    signal.extend(np.zeros(int(fs * pause)))

    # Second and third chirps 34-18 kHz, 4 ms, pause 8 ms
    for _ in range(2):
        signal.extend(generate_chirp(34000, 18000, t1, fs))
        signal.extend(np.zeros(int(fs * pause)))

    # Remaining chirps 34-18 kHz, 2 ms, no pause
    for _ in range(129):
        signal.extend(generate_chirp(34000, 18000, t2, fs))

    return np.array(signal)


if __name__ == "__main__":

    f0 = 18000
    f1 = 34000

    # Signal generation
    signal = generate_signal(fs, t1, t2, pause)

    fn = 128

    # Normalize the signal to the range [-1, 1]
    signal = signal / np.max(np.abs(signal))
    # write("1834_cs1.wav", fs, signal.astype(np.float32))

    signal = signal * 8192
    signal = signal.astype(np.int16)
    # Save the signal as a WAV file
    # write("1834_cs1.wav", fs, signal)

    # with TdmsWriter("1834_cs1.tdms") as tdms_writer:
    #     channel = ChannelObject("Group", "Signal", signal)
    #     tdms_writer.write_segment([channel])

        # Calculate the duration of the signal
    duration = len(signal) / fs
    print(f"Total duration of the signal: {duration} seconds")
    print(len(signal))

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # Display the resulting signal
    ax[0].plot(signal)
    ax[0].set_title("Resulting Signal")
    ax[0].set_xlabel("Sample")
    ax[0].set_ylabel("Amplitude")

    # Calculate the Continuous Wavelet Transform (CWT) of the signal
    freqs, cwt_matrix = fcwt.cwt(signal, fs, f0, f1, fn)

    ax[1].imshow(np.abs(cwt_matrix), extent=[0, len(signal), 1, 128], cmap='PRGn', aspect='auto',
                vmax=abs(cwt_matrix).max(), vmin=-abs(cwt_matrix).max())
    ax[1].set_title("CWT Matrix")
    ax[1].set_xlabel("Sample")
    ax[1].set_ylabel("Scale")


    plt.show()
