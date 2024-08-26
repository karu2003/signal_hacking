import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import correlate
import json
from scipy.io import wavfile
import signal_helper as sh
import multiprocessing as mp
import sys

# Функция для вычисления корреляции, которую можно использовать в параллельной обработке
def compute_spline_correlation(num_knots, x, y, signal, sample_rate, initial_phase):
    # Выбираем узлы для сплайн-интерполяции
    knots = np.linspace(x.min(), x.max(), num=num_knots)
    cs = CubicSpline(knots, np.interp(knots, x, y))
    
    # Оцениваем значения сплайна на всем диапазоне x
    y_spline_pred = cs(x)
    mapped_poly_freq = sh.map_values_reverse(y_spline_pred, 0, 1, 7000, 17000)
    synthesized_chirp = sh.freq_to_chirp(mapped_poly_freq, sample_rate, initial_phase)
    
    max_corr, _, _ = sh.compute_correlation(signal, synthesized_chirp)
    return max_corr, num_knots, y_spline_pred, synthesized_chirp

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

    r = range(3, 48)

    # Параллельное вычисление корреляций для разных узлов сплайна
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(compute_spline_correlation, [(num_knots, x, y, signal, sample_rate, initial_phase) for num_knots in r])

    # Извлекаем результаты и находим наилучшую корреляцию
    correlations, knots_values, approximations, chirps = zip(*results)
    max_correlation = max(correlations)
    best_index = correlations.index(max_correlation)
    best_knots = knots_values[best_index]
    best_approximation = approximations[best_index]

    # Получаем параметры лучшего сплайна
    best_knots_positions = np.linspace(x.min(), x.max(), num=best_knots)
    best_spline = CubicSpline(best_knots_positions, np.interp(best_knots_positions, x, y))

    # Сохранение параметров сплайна в файл
    spline_params = {
        "knots": best_knots_positions.tolist(),
        "coefficients": best_spline.c.tolist()
    }

    with open("spline_params.json", "w") as f:
        json.dump(spline_params, f)

    # Создаем фигуру и оси для нескольких графиков
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Построим график зависимости корреляции от количества узлов сплайна
    axs[0].plot(r, correlations, marker="o")
    axs[0].set_title("Зависимость корреляции от количества узлов сплайна")
    axs[0].set_xlabel("Количество узлов")
    axs[0].set_ylabel("Корреляция")
    axs[0].grid(True)

    # Визуализация всех полученных кривых на одном графике
    axs[1].plot(x, y, label="Исходная мгновенная частота", color="black", linewidth=2)
    axs[1].plot(x, best_approximation, label=f"Сплайн с {best_knots} узлами", color="red", linestyle="--")
    axs[1].set_title("Аппроксимация мгновенной частоты сплайном с наилучшим количеством узлов")
    axs[1].set_xlabel("Индекс")
    axs[1].set_ylabel("Частота")
    axs[1].legend()
    axs[1].grid(True)

    # Построим график оригинального и синтезированного сигналов
    axs[2].plot(signal, label="Оригинальный сигнал", color="black")
    axs[2].plot(chirps[best_index], label="Синтезированный сигнал", linestyle="--", color="red")
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
