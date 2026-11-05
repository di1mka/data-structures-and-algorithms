#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 6. Линейный, бинарный и интерполяционный поиск (вариант 7)."""

from __future__ import annotations

import argparse
import bisect
import random
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Алгоритмы поиска (реализуются вручную, без in/index/bisect в основной части)
# ---------------------------------------------------------------------------


def linear_search(a: list[int], key: int) -> tuple[int, int]:
    """Линейный поиск первого вхождения key в отсортированном по неубыванию массиве.

    Возвращает (индекс первого вхождения или -1, число сравнений ключей).
    Ожидаемая сложность: O(n) в среднем и худшем случаях, O(1) в лучшем случае.
    """
    comparisons = 0
    for i, val in enumerate(a):
        comparisons += 1
        if val == key:
            return i, comparisons
        if val > key:
            # Массив отсортирован по неубыванию: дальше элементы только >= val > key
            return -1, comparisons
    return -1, comparisons


def binary_search(a: list[int], key: int) -> tuple[int, int]:
    """Бинарный поиск key в отсортированном массиве.

    Возвращает (индекс любого вхождения или -1, число сравнений ключей).
    Поддерживает инвариант замкнутого отрезка [lo, hi]: если key присутствует
    в массиве, он гарантированно лежит в a[lo..hi]. На каждом шаге длина
    отрезка строго уменьшается (lo = mid + 1 или hi = mid - 1).
    Ожидаемая сложность: O(log n) в среднем и худшем случаях.
    """
    lo = 0
    hi = len(a) - 1
    comparisons = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        comparisons += 1
        val = a[mid]
        if val == key:
            return mid, comparisons
        if val < key:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1, comparisons


def interpolation_search(a: list[int], key: int) -> tuple[int, int]:
    """Интерполяционный поиск key в отсортированном массиве.

    Возвращает (индекс любого вхождения или -1, число сравнений ключей).
    Позиция зонда: mid = lo + (key - a[lo]) * (hi - lo) // (a[hi] - a[lo]).
    Защищён от деления на ноль при a[hi] == a[lo] и от выхода key за [a[lo], a[hi]].
    Средняя сложность на равномерных данных: O(log log n);
    худший случай на сильно неравномерных (экспоненциальных) данных: O(n).
    """
    n = len(a)
    if n == 0:
        return -1, 0

    lo = 0
    hi = n - 1
    comparisons = 0

    while lo <= hi:
        # Проверка попадания ключа в текущий диапазон значений [a[lo], a[hi]]
        comparisons += 1
        if key < a[lo] or key > a[hi]:
            return -1, comparisons

        # Защита от деления на ноль, когда все элементы на отрезке [lo, hi] равны
        if a[hi] == a[lo]:
            comparisons += 1
            if a[lo] == key:
                return lo, comparisons
            return -1, comparisons

        # Вычисление позиции зонда по линейной интерполяции
        mid = lo + (key - a[lo]) * (hi - lo) // (a[hi] - a[lo])

        comparisons += 1
        val = a[mid]
        if val == key:
            return mid, comparisons
        if val < key:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1, comparisons


# ---------------------------------------------------------------------------
# 2. Данные по варианту: два распределения, присутствующие/отсутствующие ключи
# ---------------------------------------------------------------------------


def make_dataset(n: int, distribution: str, rng: random.Random) -> list[int]:
    """Отсортированный массив n ключей: 'uniform' — равномерное распределение,
    'exponential' — неравномерное (экспоненциальное, значения сгущаются у нуля)."""
    if distribution == "uniform":
        return sorted(rng.randint(0, 10 * n) for _ in range(n))
    if distribution == "exponential":
        return sorted(int(rng.expovariate(1.0 / 1000.0)) for _ in range(n))
    raise ValueError(f"неизвестное распределение: {distribution!r}")


def pick_keys(a: list[int], rng: random.Random, count: int, present: bool) -> list[int]:
    """count ключей для серии поисков: присутствующие берутся из массива,
    отсутствующие подбираются вне множества его значений."""
    if present:
        return [a[rng.randrange(len(a))] for _ in range(count)]
    have = set(a)
    hi = (a[-1] if a else 0) + 2
    keys: list[int] = []
    while len(keys) < count:
        k = rng.randint(-hi, hi)
        if k not in have:
            keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# 3. Репрезентативные тесты и инварианты (шаги 1–3 методики верификации)
# ---------------------------------------------------------------------------

ALGORITHMS = [
    ("linear", linear_search),
    ("binary", binary_search),
    ("interpolation", interpolation_search),
]


def self_check() -> None:
    """Граничные случаи + сверка с эталоном (оператор in, list.index, bisect)."""
    for name, fn in ALGORITHMS:
        # Пустой массив: элемент не найден, сравнений нет.
        assert fn([], 1) == (-1, 0), f"{name}: пустой массив"
        # Один элемент: найден / не найден.
        assert fn([7], 7)[0] == 0, f"{name}: один элемент, ключ найден"
        assert fn([7], 3)[0] == -1, f"{name}: один элемент, ключ отсутствует"
        # Массив из одинаковых элементов (проверка защиты от деления на ноль)
        assert fn([5, 5, 5, 5], 5)[0] != -1, f"{name}: одинаковые элементы (найден)"
        assert fn([5, 5, 5, 5], 3)[0] == -1, f"{name}: одинаковые элементы (не найден)"
        # Первый, последний и отсутствующие ключи (внутри и вне диапазона).
        a = [1, 3, 3, 5, 8, 13, 21]
        assert fn(a, 1)[0] == 0, f"{name}: первый элемент"
        assert fn(a, 21)[0] == len(a) - 1, f"{name}: последний элемент"
        assert fn(a, 4)[0] == -1, f"{name}: отсутствующий ключ внутри диапазона"
        assert fn(a, 100)[0] == -1, f"{name}: ключ вне диапазона"

    # Сверка с эталоном стандартной библиотеки на случайных массивах.
    rng = random.Random(0)
    for _ in range(300):
        a = sorted(rng.randint(0, 60) for _ in range(rng.randint(1, 40)))
        key = rng.randint(-5, 65)
        # Эталон присутствия ключа — bisect_left (и, эквивалентно, оператор in).
        j = bisect.bisect_left(a, key)
        present = j < len(a) and a[j] == key
        assert present == (key in a)
        for name, fn in ALGORITHMS:
            idx, cmp_count = fn(a, key)
            assert cmp_count >= 1, f"{name}: счётчик сравнений не ведётся"
            if present:
                assert idx != -1 and a[idx] == key, f"{name}: ключ не найден"
            else:
                assert idx == -1, f"{name}: найден отсутствующий ключ"
        # Линейный поиск обязан возвращать первое вхождение — эталон list.index.
        if present:
            assert linear_search(a, key)[0] == a.index(key)
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 4. Бенчмарк: сравнения и время от n (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5  # число повторов серии замеров (берётся медиана)
N_KEYS = 200  # число ключей в одной серии поисков


def bench(fn, a: list[int], keys: list[int]) -> float:
    """Медиана времени одной серии поисков по REPEATS запускам, с прогревом."""
    for k in keys:  # прогрев — не учитывается
        fn(a, k)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for k in keys:
            fn(a, k)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def mean_comparisons(fn, a: list[int], keys: list[int]) -> float:
    """Среднее число сравнений ключей на один поиск."""
    return statistics.mean(fn(a, k)[1] for k in keys)


def run_benchmarks(seed: int) -> dict:
    rng = random.Random(seed)
    results: dict = {"uniform": {}, "exponential": {}}

    for distribution in ("uniform", "exponential"):
        print(f"\nРаспределение: {distribution}")
        for present in (True, False):
            label = "есть" if present else "нет"
            results[distribution][label] = {name: [] for name, _ in ALGORITHMS}

        for n in SIZES:
            a = make_dataset(n, distribution, rng)
            for present in (True, False):
                keys = pick_keys(a, rng, N_KEYS, present)
                label = "есть" if present else "нет"
                for name, fn in ALGORITHMS:
                    if name == "linear" and n > 30_000:
                        continue  # линейный поиск: большие n занимают много времени
                    mc = mean_comparisons(fn, a, keys)
                    t = bench(fn, a, keys)
                    results[distribution][label][name].append((n, mc, t))
                    print(
                        f"  n={n:>7}  ключ {label:<4}  {name:<13}"
                        f"  сравнений={mc:>10.1f}  t={t:.6f} c"
                    )
    return results


def plot_results(results: dict, out_dir: Path) -> None:
    """Построение графиков среднего числа сравнений и времени от n."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен — график пропущен.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    dist_titles = {
        "uniform": "Равномерное распределение (uniform)",
        "exponential": "Экспоненциальное распределение (exponential)",
    }
    markers = {"linear": "s", "binary": "o", "interpolation": "^"}

    for col, dist in enumerate(("uniform", "exponential")):
        ax_cmp = axes[0, col]
        ax_time = axes[1, col]

        for label, style in (("есть", "-"), ("нет", "--")):
            for name, _ in ALGORITHMS:
                pts = results[dist][label][name]
                if not pts:
                    continue
                ns = [p[0] for p in pts]
                cmps = [p[1] for p in pts]
                ts = [p[2] for p in pts]
                legend_lbl = f"{name} (ключ {label})"
                ax_cmp.plot(
                    ns, cmps, linestyle=style, marker=markers[name], label=legend_lbl
                )
                ax_time.plot(
                    ns, ts, linestyle=style, marker=markers[name], label=legend_lbl
                )

        ax_cmp.set_xscale("log")
        ax_cmp.set_yscale("log")
        ax_cmp.set_title(f"Число сравнений от n — {dist_titles[dist]}")
        ax_cmp.set_xlabel("Размер массива n")
        ax_cmp.set_ylabel("Среднее число сравнений (лог. шкала)")
        ax_cmp.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_cmp.legend(fontsize=8)

        ax_time.set_xscale("log")
        ax_time.set_yscale("log")
        ax_time.set_title(f"Время серии из {N_KEYS} поисков от n — {dist_titles[dist]}")
        ax_time.set_xlabel("Размер массива n")
        ax_time.set_ylabel("Время выполнения t, с (лог. шкала)")
        ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_time.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "lab06_search.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nГрафик сохранён:\n  {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="каталог для сохранения графика",
    )
    args = ap.parse_args()
    seed = 30 + args.variant
    print(f"Запуск ЛР-06: вариант {args.variant}, seed={seed}")
    random.seed(seed)
    self_check()
    results = run_benchmarks(seed)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()
