import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import json
from scipy.io import wavfile
import signal_helper as sh
import sys

# Функция для вычисления корреляции с кусочно-линейной интерполяцией
def compute_spline_correlation(num_knots, x, y, signal, sample_rate, initial_phase):
    # Выбираем узлы для линейной интерполяции
    knots = np.linspace(x.min(), x.max(), num=num_knots)
    
    # Оцениваем значения линейного сплайна на всем диапазоне x
    y_spline_pred = np.interp(x, knots, np.interp(knots, x, y))
    mapped_poly_freq = sh.map_values_reverse(y_spline_pred, 0, 1, 7000, 17000)
    synthesized_chirp = sh.freq_to_chirp(mapped_poly_freq, sample_rate, initial_phase)
    
    max_corr, _, _ = sh.compute_correlation(signal, synthesized_chirp)
    return max_corr, y_spline_pred, synthesized_chirp

# Функция для поиска максимальной корреляции методом золотого сечения
def golden_section_search(x, y, signal, sample_rate, initial_phase, a, b, tol=1):
    gr = (np.sqrt(5) + 1) / 2  # Золотое сечение
    
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while abs(c - d) > tol:
        corr_c, _, _ = compute_spline_correlation(int(c), x, y, signal, sample_rate, initial_phase)
        corr_d, _, _ = compute_spline_correlation(int(d), x, y, signal, sample_rate, initial_phase)

        if corr_c > corr_d:
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    best_knots = int((b + a) / 2)
    max_correlation, best_approximation, best_chirp = compute_spline_correlation(best_knots, x, y, signal, sample_rate, initial_phase)

    return best_knots, max_correlation, best_approximation, best_chirp

# Основная функция
def main():
    try:
        data = np.loadtxt("instantaneous_frequency.csv", delimiter=",")
        sample_rate, signal = wavfile.read("first_frame.wav")
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
    initial_phase = 180
    polynomial_type = "interp"

    # Оптимизация методом золотого сечения
    a, b = 3, 47  # Начальные границы диапазона количества узлов
    tol = 1  # Допуск для завершения поиска

    best_knots, max_correlation, best_approximation, best_chirp = golden_section_search(
        x, y, signal, sample_rate, initial_phase, a, b, tol
    )

    # Сохранение параметров линейного сплайна в файл
    best_knots_positions = np.linspace(x.min(), x.max(), num=best_knots)
    spline_params = {
        "type": polynomial_type,
        "coefficients": best_knots_positions.tolist(),
    }

    with open("params.json", "w") as f:
        json.dump(spline_params, f)

    # Создаем фигуру и оси для нескольких графиков
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Построим график зависимости корреляции от количества узлов сплайна
    axs[0].plot(range(a, b+1), [compute_spline_correlation(i, x, y, signal, sample_rate, initial_phase)[0] for i in range(a, b+1)], marker="o")
    axs[0].axvline(best_knots, color='red', linestyle='--')
    axs[0].set_title("Зависимость корреляции от количества узлов линейного сплайна")
    axs[0].set_xlabel("Количество узлов")
    axs[0].set_ylabel("Корреляция")
    axs[0].grid(True)

    # Визуализация всех полученных кривых на одном графике
    axs[1].plot(x, y, label="Исходная мгновенная частота", color="black", linewidth=2)
    axs[1].plot(x, best_approximation, label=f"Линейный сплайн с {best_knots} узлами", color="red", linestyle="--")
    axs[1].set_title("Аппроксимация мгновенной частоты линейным сплайном с наилучшим количеством узлов")
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

    # Вывод параметров лучшего сплайна
    print(f"Оптимальное количество узлов: {best_knots}")
    print(f"Максимальная корреляция: {max_correlation:.4f}")

    plt.tight_layout()
    plt.show()

# Запуск основной функции
if __name__ == "__main__":
    main()
