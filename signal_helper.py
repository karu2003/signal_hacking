# Helper functions
import numpy as np
import fcwt

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

def calculate_cwt_envelope(signal, fs, f0, f1, fn):
    # Вычислить CWT с помощью fcwt
    freqs, cwt_matrix = fcwt.cwt(signal, int(fs), f0, f1, fn)
    # Рассчитать огибающую как максимумы амплитуд по частотам
    cwt_envelope = np.max(np.abs(cwt_matrix), axis=0)
    return cwt_envelope

def find_pulse_widths(envelope, fs, threshold):
    # Преобразовать огибающую в бинарный вид по порогу
    binary_envelope = envelope > threshold
    # Найти изменения (переключения) в бинарной огибающей
    changes = np.diff(binary_envelope.astype(int))
    # Найти начала и окончания импульсов
    pulse_starts = np.where(changes == 1)[0] + 1
    pulse_ends = np.where(changes == -1)[0] + 1
    if binary_envelope[0]:
        pulse_starts = np.insert(pulse_starts, 0, 0)
    if binary_envelope[-1]:
        pulse_ends = np.append(pulse_ends, len(envelope))
    # Рассчитать ширину импульсов
    pulse_widths = (pulse_ends - pulse_starts) / fs
    intervals = list(zip(pulse_starts / fs, pulse_ends / fs))
    return intervals, pulse_widths

def find_pauses(cwt_envelope, fs, threshold):
    """
    Находит паузы в сигнале по уровню cwt_envelope.

    Parameters:
    cwt_envelope (numpy.ndarray): Массив значений cwt_envelope.
    fs (float): Частота дискретизации.
    threshold (float): Пороговое значение для определения пауз.

    Returns:
    list: Список кортежей, где каждый кортеж содержит начало, конец и длину паузы в секундах.
    """
    pauses = []
    in_pause = False
    start = 0

    for i, value in enumerate(cwt_envelope):
        if value < threshold and not in_pause:
            in_pause = True
            start = i
        elif value >= threshold and in_pause:
            in_pause = False
            end = i
            pause_start_time = start / fs
            pause_end_time = end / fs
            pause_length = pause_end_time - pause_start_time
            pauses.append((pause_start_time, pause_end_time, pause_length))

    # Если сигнал заканчивается на паузе
    if in_pause:
        end = len(cwt_envelope)
        pause_start_time = start / fs
        pause_end_time = end / fs
        pause_length = pause_end_time - pause_start_time
        pauses.append((pause_start_time, pause_end_time, pause_length))

    return pauses

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

def fill_max2one(in_array):
    out_s = np.zeros_like(in_array)
    max_indexs = []
    for count, m in enumerate(in_array):
        max_index = np.argmax(m)
        max_indexs.append(max_index)
        # f1D.append(out_s[count][max_index])
        in_array[count].fill(0)

    for count, m in enumerate(max_indexs):
        out_s[count][m] = 1.0

    return max_indexs, out_s