import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import correlate
import json
from scipy.io import wavfile
import signal_helper as sh
import multiprocessing as mp
import sys
import re
from signal_type import signal_type

# Функция для вычисления корреляции, которую можно использовать в параллельной обработке
def compute_spline_correlation(
    num_knots, x, y, signal, sample_rate, initial_phase, f0, f1
):
    # Выбираем узлы для сплайн-интерполяции
    knots = np.linspace(x.min(), x.max(), num=num_knots)
    cs = CubicSpline(knots, np.interp(knots, x, y))

    # Оцениваем значения сплайна на всем диапазоне x
    y_spline_pred = cs(x)
    mapped_poly_freq = sh.map_values_tb(y_spline_pred, f0, f1, reverse=True)
    synthesized_chirp = sh.freq_to_chirp(mapped_poly_freq, sample_rate, initial_phase)

    max_corr, _, _ = sh.compute_correlation(signal, synthesized_chirp)
    return max_corr, num_knots, y_spline_pred, synthesized_chirp, cs

# Основная функция
def main():
    # signal_type = "1834cs1"
    # signal_type = "1707cs1"
    match = re.search(r"(\d{2})(\d{2})", signal_type)
    f1 = int(match.group(1)) * 1000  # Первая часть числа
    f0 = int(match.group(2)) * 1000  # Вторая часть числа
    print(f"f1 = {f1} Гц, f0 = {f0} Гц")

    try:
        data = np.loadtxt(
            f"instan/{signal_type}_map_instantaneous_freq.csv", delimiter=","
        )
        sample_rate, signal = wavfile.read(f"wav/{signal_type}_first_frame.wav")
        signal = sh.normalize(signal)
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)

    with open(f"params/{signal_type}_signal_params.json", "r") as json_file:
        params = json.load(json_file)

    initial_phase = params.get("initial_phase", 180)

    print(f"Частота дискретизации: {sample_rate}")
    print(f"Длина сигнала: {len(signal)}")
    print(f"Длина Instantaneous: {len(data)}")

    y = data
    x = np.linspace(0, 1, len(y))
    polynomial_type = "spline"

    r = range(3, 48)

    # Параллельное вычисление корреляций для разных узлов сплайна
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(
            compute_spline_correlation,
            [
                (num_knots, x, y, signal, sample_rate, initial_phase, f0, f1)
                for num_knots in r
            ],
        )

    # Извлекаем результаты и находим наилучшую корреляцию
    correlations, knots_values, approximations, chirps, splines = zip(*results)
    max_correlation = max(correlations)
    best_index = correlations.index(max_correlation)
    best_knots = knots_values[best_index]
    best_spline = splines[best_index]

    # Получаем параметры лучшего сплайна
    best_knots_positions = np.linspace(x.min(), x.max(), num=best_knots)

    spline_params = {
        "type": polynomial_type,
        "knots": best_knots_positions.tolist(),
        "y_values_at_knots": best_spline(best_knots_positions).tolist(),  # Сохраняем значения y в узлах
        "extrapolate": best_spline.extrapolate
    }

    with open(f"poly/{signal_type}_params.json", "w") as f:
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
    axs[1].plot(
        x,
        approximations[best_index],
        label=f"Сплайн с {best_knots} узлами",
        color="red",
        linestyle="--",
    )
    axs[1].set_title(
        "Аппроксимация мгновенной частоты сплайном с наилучшим количеством узлов"
    )
    axs[1].set_xlabel("Индекс")
    axs[1].set_ylabel("Частота")
    axs[1].legend()
    axs[1].grid(True)

    # Построим график оригинального и синтезированного сигналов
    axs[2].plot(signal, label="Оригинальный сигнал", color="black")
    axs[2].plot(
        chirps[best_index], label="Синтезированный сигнал", linestyle="--", color="red"
    )
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
