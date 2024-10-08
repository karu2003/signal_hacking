# Helper functions
import numpy as np
import fcwt
from scipy.signal import chirp, hilbert, sweep_poly, correlate
from scipy.optimize import curve_fit
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, butter, filtfilt
from scipy.signal import firwin, lfilter, resample_poly


def convert_phase_to_radians(initial_phase_degrees):
    # Преобразование фазы из градусов в радианы
    initial_phase_radians = np.radians(initial_phase_degrees)
    return initial_phase_radians


def convert_phase_to_degrees(initial_phase):
    # Преобразование фазы из радиан в градусы
    initial_phase_degrees = np.degrees(initial_phase)
    return initial_phase_degrees


def hilbert_instantaneous_frequency(signal, t):
    # Вычисление мгновенной частоты через преобразование Гильберта
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * np.diff(t))
    return instantaneous_frequency


def cwt_instantaneous_frequency(coef, freqs):
    # Найти частоты с максимальной мощностью на каждом временном шаге
    power = np.abs(coef) ** 2
    max_power_indices = np.argmax(power, axis=0)
    instantaneous_frequency = freqs[max_power_indices]
    return instantaneous_frequency


def linear_model(t, f0, k):
    return f0 + k * t


def polynomial_model(t, a0, a1, a2):
    return a0 + a1 * t + a2 * t**2


def polynomial_model4(t, a0, a1, a2, a3, a4):
    return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4


def polynomial_model6(t, a0, a1, a2, a3, a4, a5, a6):
    return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5 + a6 * t**6


def sinusoidal_model(x, A, B, C, D):
    return A * np.sin(B * x + C) + D


def sinusoidal_model_2nd_order(x, A1, B1, C1, A2, B2, C2, D):
    return A1 * np.sin(B1 * x + C1) + A2 * np.sin(B2 * x + C2) + D


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


# Chirp generation
def generate_chirp(f0, f1, t, fs):
    t = np.linspace(0, t, int(fs * t))
    return chirp(t, f0=f0, f1=f1, t1=t[-1], method="linear")


def freq_to_chirp(freq_t, sample_rate, initial_phase_degrees=180):
    """
    Генерирует чирп-сигнал на основе временного ряда частот.

    :param freq_t: Временной ряд частот.
    :param sample_rate: Частота дискретизации.
    :param initial_phase_degrees: Начальная фаза в градусах (по умолчанию 180).
    :return: Сгенерированный чирп-сигнал.
    """
    # Задать начальную фазу
    initial_phase = convert_phase_to_radians(initial_phase_degrees)

    # Вычисляем фазу сигнала как интеграл от частоты с учетом начальной фазы
    phase_t = 2 * np.pi * np.cumsum(freq_t) / sample_rate + initial_phase

    # Генерация чирп-сигнала с использованием фазы
    synthesized_chirp = np.sin(phase_t)

    return synthesized_chirp


def generate_sin_chirp(
    f_start, f_end, T, fs, num_sine_periods, amp_reduction_factor=0.1
):  # , fn=128):
    """
    Генерирует чирп-сигнал с синусоидальным изменением частоты.

    Parameters:
    - f_start: начальная частота (Гц)
    - f_end: конечная частота (Гц)
    - T: длительность сигнала (с)
    - fs: частота дискретизации (Гц)
    - num_sine_periods: количество периодов синусоидального изменения частоты
    - amp_reduction_factor: коэффициент уменьшения амплитуды синусоидальной модуляции
    - fn: количество частотных бинов для вейвлет-преобразования

    Returns:
    - chirp_signal: сгенерированный чирп-сигнал
    - t: временной вектор
    - cwt_matrix: матрица вейвлет-преобразования
    - freqs: частоты для вейвлет-преобразования
    """
    # Создание временного вектора
    t = np.linspace(0, T, int(T * fs), endpoint=False)

    # Определение частоты синусоиды, чтобы получить указанное количество периодов
    sine_freq = num_sine_periods / T

    # Линейное изменение частоты от начальной до конечной частоты
    freq_linear = np.linspace(f_start, f_end, len(t))

    # Синусоидальное изменение частоты с уменьшенной амплитудой
    amp = (f_start - f_end) / 2 * amp_reduction_factor
    freq_sine = (f_start + f_end) / 2
    freq_sine_modulated = freq_sine + amp * np.cos(2 * np.pi * sine_freq * t)

    # Изменение частоты: линейное изменение с синусоидальным модулем
    freq_t = freq_linear + freq_sine_modulated - freq_sine

    # Комбинированное изменение частоты с ограничением в диапазоне [f_start, f_end]
    # freq_t = np.clip(freq_linear + freq_sine_modulated - freq_sine, f_start, f_end)

    initial_phase = convert_phase_to_radians(180)

    # Интегрируем частоту для получения фазы
    phase = 2 * np.pi * np.cumsum(freq_t) / fs + initial_phase

    # Генерация чирп-сигнала
    chirp_signal = np.sin(phase)

    # Применение вейвлет-преобразования
    # freqs, cwt_matrix = fcwt.cwt(chirp_signal, fs, f_start, f_end, fn)

    return chirp_signal
    # return chirp_signal, t, freqs, cwt_matrix


def compute_correlation(signal_data, synthesized_chirp):
    len_first_frame = len(signal_data)
    len_synthesized_chirp = len(synthesized_chirp)

    # Обрезаем до минимальной длины
    min_length = min(len_first_frame, len_synthesized_chirp)
    signal_data = signal_data[:min_length]
    synthesized_chirp = synthesized_chirp[:min_length]

    # Вычисляем корреляцию
    correlation = correlate(signal_data, synthesized_chirp, mode="full")
    max_corr = np.max(correlation)
    return max_corr, correlation, len(signal_data)


def normalize(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    else:
        return signal


def normalize01(data):
    """Нормализует данные от 0 до 1."""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def map_values(data, new_min, new_max):
    """Мапирует значения массива data в новый диапазон [new_min, new_max]."""
    old_min = np.min(data)
    old_max = np.max(data)
    return new_min + (data - old_min) * (new_max - new_min) / (old_max - old_min)


def map_values_reverse(data, old_min, old_max, new_min, new_max):
    """Мапирует значения массива data из диапазона [old_min, old_max] в диапазон [new_min, new_max] в обратном порядке."""
    return new_max - (data - old_min) * (new_max - new_min) / (old_max - old_min)


def map_values_tb(data, range_min, range_max, reverse=False):
    """Мапирует значения массива data из диапазона [range_min, range_max] в диапазон [0, 1].
    Если reverse=True, мапирует значения из диапазона [0, 1] в диапазон [range_min, range_max].
    """
    if reverse:
        return range_min + data * (range_max - range_min)
    else:
        return (data - range_min) / (range_max - range_min)


def determine_chirp_direction(intervals, sample_rate, instantaneous_frequency):
    """
    Определяет направление чирпа (восходящий или нисходящий) в каждом интервале.

    Параметры:
    intervals: список кортежей (в секундах), содержащих начало и конец каждого интервала.
    sample_rate: частота дискретизации сигнала.
    instantaneous_frequency: массив мгновенных частот, соответствующий сигналу.

    Возвращает:
    chirp_directions: список строк, указывающих направление чирпа для каждого интервала.
    """
    chirp_directions = []

    for start_sec, end_sec in intervals:
        # Перевод интервала из секунд в индексы
        start_idx = int(start_sec * sample_rate)
        end_idx = int(end_sec * sample_rate)

        # Извлечение мгновенной частоты для интервала
        freq_interval = instantaneous_frequency[start_idx:end_idx]

        # Вычисление наклона мгновенной частоты
        if len(freq_interval) > 1:
            slope = np.polyfit(np.arange(len(freq_interval)), freq_interval, 1)[0]
            if slope > 0:
                direction = 1
            else:
                direction = -1
        else:
            direction = 0  # В случае недостаточного количества точек данных

        chirp_directions.append(direction)

    return chirp_directions


def create_hermite_spline(spline_params, x):
    """Создает сплайн Эрмита на основе загруженных параметров."""
    coefficients = np.array(spline_params["knots_positions"])
    y_knots = np.array(spline_params["knots_values"])
    dydx = np.array(spline_params["derivatives"])
    hermite_spline = CubicHermiteSpline(coefficients, y_knots, dydx)
    return hermite_spline(x)


def create_cubic_spline(spline_params, x):
    knots = np.array(spline_params["knots"])
    # Восстанавливаем сплайн с использованием значений y в узлах.
    y_values_at_knots = np.array(spline_params["y_values_at_knots"])
    extrapolate = spline_params["extrapolate"]
    # Создаем кубический сплайн, используя узлы и соответствующие значения y
    cubic_spline = CubicSpline(knots, y_values_at_knots, extrapolate=extrapolate)
    return cubic_spline(x)


# Низкочастотный фильтр Баттерворта
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Частота Найквиста
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def manual_resample_multiplicity(signal, original_fs, target_fs):
    """
    Ресамплинг сигнала вниз с использованием фильтрации и децимации.

    :param signal: Исходный сигнал (массив NumPy)
    :param original_fs: Исходная частота дискретизации
    :param target_fs: Целевая частота дискретизации
    :return: Ресамплированный сигнал
    """
    if original_fs <= target_fs:
        raise ValueError("Исходная частота дискретизации должна быть больше целевой.")

    # Применение фильтра низких частот
    nyquist_rate = original_fs / 2.0
    cutoff_hz = target_fs / 2.0
    numtaps = 101  # Параметр размера фильтра, можно варьировать
    fir_coeff = firwin(numtaps, cutoff_hz / nyquist_rate)
    filtered_signal = lfilter(fir_coeff, 1.0, signal)

    # Вычисление децимационного фактора и проверка его на целочисленность
    decimation_factor = original_fs / target_fs
    if not decimation_factor.is_integer():
        raise ValueError(
            "Частоты дискретизации должны быть кратны для целочисленного децимационного фактора."
        )
    decimation_factor = int(decimation_factor)

    # Децимация - выбор каждого n-го отсчета
    resampled_signal = filtered_signal[::decimation_factor]

    return resampled_signal


def manual_resample(signal, original_fs, target_fs):
    """
    Ресамплинг сигнала с использованием интерполяции и децимации для некратных частот.

    :param signal: Исходный сигнал (массив NumPy)
    :param original_fs: Исходная частота дискретизации
    :param target_fs: Целевая частота дискретизации
    :return: Ресамплированный сигнал
    """
    # Проверка на допустимость частот
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError("Частоты дискретизации должны быть положительными значениями.")

    # Определение дробного коэффициента интерполяции и децимации
    gcd = np.gcd(int(original_fs), int(target_fs))
    up = int(target_fs / gcd)
    down = int(original_fs / gcd)

    # Фильтр низких частот (проектируем фильтр для нового Nyquist)
    nyquist_rate = 0.5 * min(original_fs, target_fs)
    cutoff_hz = nyquist_rate * 0.9  # небольшое снижение для предотвращения искажений
    numtaps = 101  # количество коэффициентов фильтра
    fir_coeff = firwin(numtaps, cutoff_hz / (0.5 * original_fs))

    # Применение фильтрации к сигналу
    filtered_signal = lfilter(fir_coeff, 1.0, signal)

    # Ресэмплинг с помощью интерполяции и децимации
    resampled_signal = resample_poly(filtered_signal, up, down)

    return resampled_signal


def find2power(n):
    m = 0
    m2 = 1 << m  # 2 to the power of m
    while m2 - n < 0:
        m += 1
        m2 <<= 1  # m2 = m2 * 2
    return m

def find_nearest_power_of_two(n):
    """Находит ближайшую степень двойки для заданного числа n."""
    if n <= 0:
        raise ValueError("Число должно быть положительным.")
    
    # Находим ближайшую степень двойки
    power = 1
    while power < n:
        power <<= 1  # Умножаем на 2
    
    # Определяем ближайшую степень двойки
    prev_power = power >> 1  # Делим на 2, чтобы получить предыдущую степень двойки
    if abs(n - prev_power) < abs(n - power):
        return prev_power
    else:
        return power

