import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.io import wavfile
import signal_helper as sh
import json
import sys
from scipy.interpolate import CubicHermiteSpline
import re


# Функция для вычисления корреляции с кусочно-линейной интерполяцией Эрмита
def compute_hermite_correlation(num_knots, x, y, signal, sample_rate, initial_phase, f0=7000, f1=17000):
    # Выбираем узлы для сплайна Эрмита
    knots = np.linspace(x.min(), x.max(), num=num_knots)
    y_knots = np.interp(knots, x, y)

    # Приблизим производные численно
    dydx = np.gradient(y_knots, knots)

    # Создаем сплайн Эрмита
    hermite_spline = CubicHermiteSpline(knots, y_knots, dydx)

    # Оцениваем значения сплайна Эрмита на всем диапазоне x
    y_spline_pred = hermite_spline(x)
    mapped_poly_freq = sh.map_values_tb(y_spline_pred, f0, f1, reverse=True)
    synthesized_chirp = sh.freq_to_chirp(mapped_poly_freq, sample_rate, initial_phase)

    max_corr, _, _ = sh.compute_correlation(signal, synthesized_chirp)
    return max_corr, y_spline_pred, synthesized_chirp


# Функция бинарного поиска для оптимизации корреляции
def binary_search_optimization(
    x, y, signal, sample_rate, initial_phase, low, high, tol=1, f0=7000, f1=17000
):
    best_knots = None
    max_correlation = -np.inf
    best_approximation = None
    best_chirp = None

    while low <= high:
        mid = (low + high) // 2
        mid_left = mid - 1
        mid_right = mid + 1

        corr_mid, approx_mid, chirp_mid = compute_hermite_correlation(
            mid, x, y, signal, sample_rate, initial_phase, f0, f1
        )
        corr_left, _, _ = compute_hermite_correlation(
            mid_left, x, y, signal, sample_rate, initial_phase, f0, f1
        )
        corr_right, _, _ = compute_hermite_correlation(
            mid_right, x, y, signal, sample_rate, initial_phase ,f0, f1
        )

        if corr_mid > max(corr_left, corr_right):
            best_knots = mid
            max_correlation = corr_mid
            best_approximation = approx_mid
            best_chirp = chirp_mid
            break
        elif corr_left > corr_mid:
            high = mid_left
        else:
            low = mid_right

    return best_knots, max_correlation, best_approximation, best_chirp


# Основная функция
def main():
    # signal_type = "1834cs1"
    signal_type = "1707cs1"
    match = re.search(r"(\d{2})(\d{2})", signal_type)
    f1 = int(match.group(1)) * 1000  # Первая часть числа
    f0 = int(match.group(2)) * 1000  # Вторая часть числа
    print(f"f1 = {f1} Гц, f0 = {f0} Гц")

    try:
        data = np.loadtxt(f"instan/{signal_type}_map_instantaneous_freq.csv", delimiter=",")
        sample_rate, signal = wavfile.read(f"wav/{signal_type}_first_frame.wav")
        signal = sh.normalize(signal)
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)

    print(f"Частота дискретизации: {sample_rate}")
    print(f"Длина сигнала: {len(signal)}")
    print(f"Длина Instantaneous: {len(data)}")

    y = data
    x = np.linspace(0, 1, len(y))
    print(f"Длина x: {len(y)}")
    print(f"Длина x: {len(x)}")
    initial_phase = 180
    polynomial_type = "hermite"

    # Оптимизация методом бинарного поиска
    low, high = 3, 47  # Начальные границы диапазона количества узлов
    tol = 1  # Допуск для завершения поиска

    best_knots, max_correlation, best_approximation, best_chirp = (
        binary_search_optimization(
            x, y, signal, sample_rate, initial_phase, low, high, tol, f0, f1
        )
    )

    best_knots_positions = np.linspace(x.min(), x.max(), num=best_knots)
    y_knots = np.interp(best_knots_positions, x, y)
    dydx = np.gradient(y_knots, best_knots_positions)


    spline_params = {
        "type": polynomial_type,
        "knots_positions": best_knots_positions.tolist(),
        "knots_values": y_knots.tolist(),
        "derivatives": dydx.tolist(),
    }

    with open(f"poly/{signal_type}_params.json", "w") as f:
        json.dump(spline_params, f)

    # Вывод параметров лучшего сплайна
    print(f"Оптимальное количество узлов: {best_knots}")
    print(f"Максимальная корреляция: {max_correlation:.4f}")

    # Создаем фигуру и оси для нескольких графиков
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    # Построим график зависимости корреляции от количества узлов сплайна Эрмита
    axs[0].plot(
        range(low, high + 1),
        [
            compute_hermite_correlation(i, x, y, signal, sample_rate, initial_phase)[0]
            for i in range(low, high + 1)
        ],
        marker="o",
    )
    axs[0].axvline(best_knots, color="red", linestyle="--")
    axs[0].set_title("Зависимость корреляции от количества узлов сплайна Эрмита")
    axs[0].set_xlabel("Количество узлов")
    axs[0].set_ylabel("Корреляция")
    axs[0].grid(True)

    # Визуализация всех полученных кривых на одном графике
    axs[1].plot(x, y, label="Исходная мгновенная частота", color="black", linewidth=2)
    axs[1].plot(
        x,
        best_approximation,
        label=f"Сплайн Эрмита с {best_knots} узлами",
        color="red",
        linestyle="--",
    )
    axs[1].set_title(
        "Аппроксимация мгновенной частоты сплайном Эрмита с наилучшим количеством узлов"
    )
    axs[1].set_xlabel("Индекс")
    axs[1].set_ylabel("Частота")
    axs[1].legend()
    axs[1].grid(True)

    # Построим график оригинального и синтезированного сигналов
    axs[2].plot(signal, label="Оригинальный сигнал", color="black")
    axs[2].plot(best_chirp, label="Синтезированный сигнал", linestyle="--", color="red")
    axs[2].set_title("Оригинальный и синтезированный сигналы")
    axs[2].set_xlabel("Время (с)")
    axs[2].set_ylabel("Амплитуда")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# Запуск основной функции
if __name__ == "__main__":
    main()
