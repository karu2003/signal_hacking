import numpy as np
import matplotlib.pyplot as plt
import fcwt


def generate_sin_chirp(
    f_start, f_end, T, fs, num_sine_periods, amp_reduction_factor=1.0, fn=128
):
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
    freq_sine_modulated = freq_sine + amp * np.sin(2 * np.pi * sine_freq * t)

    # Изменение частоты: линейное изменение с синусоидальным модулем
    freq_t = freq_linear + freq_sine_modulated - freq_sine

    # Интегрируем частоту для получения фазы
    phase = 2 * np.pi * np.cumsum(freq_t) / fs

    # Генерация чирп-сигнала
    chirp_signal = np.cos(phase)

    # Применение вейвлет-преобразования
    freqs, cwt_matrix = fcwt.cwt(chirp_signal, fs, f_start, f_end, fn)

    return chirp_signal, t, freqs, cwt_matrix


def plot_results(t, chirp_signal, cwt_matrix, freqs, T):
    """
    Визуализирует чирп-сигнал и результаты вейвлет-преобразования.

    Parameters:
    - t: временной вектор
    - chirp_signal: сгенерированный чирп-сигнал
    - cwt_matrix: матрица вейвлет-преобразования
    - f_start: начальная частота (Гц)
    - f_end: конечная частота (Гц)
    - T: длительность сигнала (с)
    """
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    ax[0].plot(t, chirp_signal)
    ax[0].set_title("Чирп-сигнал с синусоидальным изменением частоты")
    ax[0].set_xlabel("Время (с)")
    ax[0].set_ylabel("Амплитуда")

    ax[1].imshow(
        np.abs(cwt_matrix),
        aspect="auto",
        interpolation="none",
        extent=[0, T, freqs[0], freqs[-1]],
    )
    ax[1].set_title("Амплитуда вейвлет-преобразования")
    ax[1].set_xlabel("Время (с)")
    ax[1].set_ylabel("Частота (Гц)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Пример использования функции
    f_start = 17000  # начальная частота (Гц)
    f_end = 7000  # конечная частота (Гц)
    T = 0.0165  # длительность сигнала (с)
    fs = 96000  # частота дискретизации (Гц)
    num_sine_periods = 8  # количество периодов синусоидального изменения частоты
    amp_reduction_factor = (
        0.2  # коэффициент уменьшения амплитуды синусоидальной модуляции
    )

    chirp_signal, t, freqs, cwt_matrix = generate_sin_chirp(
        f_start, f_end, T, fs, num_sine_periods, amp_reduction_factor
    )
    # freqs = freqs[::-1]
    cwt_matrix = cwt_matrix[::-1, :]
    plot_results(t, chirp_signal, cwt_matrix, freqs, T)
