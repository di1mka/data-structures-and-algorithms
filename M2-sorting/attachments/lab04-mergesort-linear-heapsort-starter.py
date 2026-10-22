#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 4. MergeSort, Counting, Radix и HeapSort: оценка производительности (вариант 7)."""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Сортировки (реализуются вручную; sorted() используется только как эталон)
# ---------------------------------------------------------------------------


def merge(left: list, right: list, key) -> list:
    """Стабильное слияние двух отсортированных списков в новый список.

    Стабильность: при равенстве ключей элемент из left идёт раньше (<=).
    Временная сложность: O(|left| + |right|) = O(n), память: O(n).
    """
    res = []
    i = 0
    j = 0
    n_left = len(left)
    n_right = len(right)

    while i < n_left and j < n_right:
        if key(left[i]) <= key(right[j]):
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1

    if i < n_left:
        res.extend(left[i:])
    if j < n_right:
        res.extend(right[j:])
    return res


def merge_sort(a: list, key=lambda x: x) -> list:
    """MergeSort: возвращает новый отсортированный список.

    Сложность: Theta(n log n) во всех случаях (лучший, средний, худший).
    Дополнительная память: O(n). Алгоритм стабилен.
    """
    n = len(a)
    if n <= 1:
        return list(a)
    mid = n // 2
    left = merge_sort(a[:mid], key=key)
    right = merge_sort(a[mid:], key=key)
    return merge(left, right, key)


def counting_sort(a: list[int], key_max: int) -> list[int]:
    """Устойчивая сортировка подсчётом: ключи — целые из диапазона [0, key_max].

    Схема: массив счётчиков -> префиксные суммы -> расстановка с конца.
    Время и память: Theta(n + k), где k = key_max + 1.
    """
    n = len(a)
    if n == 0:
        return []

    k = key_max + 1
    counts = [0] * k
    for x in a:
        counts[x] += 1

    # Префиксные суммы: counts[v] хранит индекс конца блока для ключа v
    for v in range(1, k):
        counts[v] += counts[v - 1]

    # Проход с конца массива обеспечивает стабильность
    out = [0] * n
    for i in range(n - 1, -1, -1):
        x = a[i]
        counts[x] -= 1
        out[counts[x]] = x
    return out


def radix_sort_lsd(a: list[int]) -> list[int]:
    """RadixSort LSD по десятичным разрядам, неотрицательные целые.

    Поверх устойчивой поразрядной counting-сортировки: от младшего разряда
    к старшему. Сложность: Theta(d * (n + b)), где b = 10, d — число разрядов.
    """
    n = len(a)
    if n == 0:
        return []

    res = list(a)
    max_val = max(res)
    if max_val == 0:
        return res

    exp = 1
    while max_val // exp > 0:
        counts = [0] * 10
        for x in res:
            digit = (x // exp) % 10
            counts[digit] += 1

        for d in range(1, 10):
            counts[d] += counts[d - 1]

        out = [0] * n
        for i in range(n - 1, -1, -1):
            x = res[i]
            digit = (x // exp) % 10
            counts[digit] -= 1
            out[counts[digit]] = x

        res = out
        exp *= 10

    return res


def sift_down(a: list[int], i: int, n: int) -> None:
    """Просейка вниз элемента i в куче a[0:n] (max-heap), на месте за O(log n)."""
    while True:
        left = 2 * i + 1
        if left >= n:
            break
        right = left + 1
        largest = right if (right < n and a[right] > a[left]) else left
        if a[i] >= a[largest]:
            break
        a[i], a[largest] = a[largest], a[i]
        i = largest


def heap_sort(a: list[int]) -> None:
    """HeapSort in-place: сортирует список a на месте, без вспомогательного массива.

    Схема: построение max-кучи просейками вниз (снизу вверх, O(n)),
    затем обмен вершины с последним элементом и просейка. Сложность: Theta(n log n).
    """
    n = len(a)
    if n <= 1:
        return

    # 1. Построение max-heap за O(n)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n)

    # 2. Извлечение максимумов по одному за O(n log n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end)


def quick_sort(a: list[int], rng: random.Random | None = None) -> list[int]:
    """QuickSort с рандомизацией опорного элемента и 3-way разбиением (из ЛР 3)."""
    if rng is None:
        rng = random.Random(37)
    res = list(a)
    n = len(res)
    if n <= 1:
        return res

    def _qsort(lo: int, hi: int) -> None:
        while lo < hi:
            pivot_val = res[rng.randrange(lo, hi + 1)]
            lt = lo
            gt = hi
            i = lo
            while i <= gt:
                val = res[i]
                if val < pivot_val:
                    res[lt], res[i] = res[i], res[lt]
                    lt += 1
                    i += 1
                elif val > pivot_val:
                    res[gt], res[i] = res[i], res[gt]
                    gt -= 1
                else:
                    i += 1

            if (lt - 1 - lo) < (hi - (gt + 1)):
                _qsort(lo, lt - 1)
                lo = gt + 1
            else:
                _qsort(gt + 1, hi)
                hi = lt - 1

    _qsort(0, n - 1)
    return res


# ---------------------------------------------------------------------------
# 2. Верификация: инварианты + сверка с эталоном sorted()
# ---------------------------------------------------------------------------


def self_check() -> None:
    """Граничные и типовые случаи, стабильность, сверка с эталоном sorted()."""
    # Граничные случаи: пустой массив, один элемент, все элементы равны
    assert merge_sort([]) == []
    assert merge_sort([7]) == [7]
    assert merge_sort([5, 5, 5]) == [5, 5, 5]
    assert counting_sort([], key_max=10) == []
    assert counting_sort([5, 5, 5], key_max=5) == [5, 5, 5]
    assert radix_sort_lsd([]) == []
    assert radix_sort_lsd([0]) == [0]
    assert radix_sort_lsd([5, 5, 5]) == [5, 5, 5]
    single = [42]
    heap_sort(single)
    assert single == [42]
    assert quick_sort([]) == []
    assert quick_sort([7]) == [7]

    # Стабильность слияния: пары (ключ, порядковый номер); эталон sorted() стабилен,
    # поэтому при равных ключах порядок номеров обязан совпасть
    rng = random.Random(0)
    pairs = [(rng.randint(0, 5), i) for i in range(100)]
    assert merge_sort(pairs, key=lambda p: p[0]) == sorted(pairs, key=lambda p: p[0])

    # Случайные тесты: инвариант перестановки + сверка с эталоном
    for _ in range(200):
        a = [rng.randint(0, 999) for _ in range(rng.randint(0, 60))]
        expected = sorted(a)
        assert merge_sort(a) == expected
        assert counting_sort(a, key_max=999) == expected
        assert radix_sort_lsd(a) == expected
        assert quick_sort(a) == expected
        buf = list(a)
        heap_sort(buf)
        assert buf == expected
        assert sorted(buf) == expected  # выход — перестановка входа
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 3. Бенчмарк и визуализация (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
SMALL_KEY_MAX = 100        # узкий диапазон: counting уместен
WIDE_KEY_MAX = 10**9       # широкий диапазон: counting теряет смысл


def make_datasets(rng: random.Random, n: int) -> dict[str, list[int]]:
    """Массивы разных распределений и диапазонов ключей для размера n."""
    almost = list(range(n))
    for _ in range(max(1, n // 100)):  # ~1 % случайных обменов
        i, j = rng.randrange(n), rng.randrange(n)
        almost[i], almost[j] = almost[j], almost[i]
    return {
        "random_wide": [rng.randint(0, WIDE_KEY_MAX) for _ in range(n)],
        "random_small": [rng.randint(0, SMALL_KEY_MAX) for _ in range(n)],
        "sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "almost_sorted": almost,
    }


def bench(fn, data: list[int]) -> float:
    """Медиана времени fn на копии data по REPEATS запускам, с прогревом."""
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        buf = list(data)
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def log_log_slope(points: list[tuple[int, float]]) -> float:
    """Наклон прямой регрессии на log-log масштабе."""
    valid = [(n, t) for n, t in points if t > 0]
    if len(valid) < 2:
        return 0.0
    xs = [math.log10(n) for n, _ in valid]
    ys = [math.log10(t) for _, t in valid]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def print_properties_table() -> None:
    """Сводная таблица теоретических свойств шести исследуемых алгоритмов."""
    print("\n--- Сводная таблица свойств алгоритмов сортировки ---")
    header = f"{'Алгоритм':<18} | {'Лучший':<10} | {'Средний':<10} | {'Худший':<10} | {'Память':<10} | {'Стаб.':<5} | {'Ограничения'}"
    print(header)
    print("-" * len(header))
    rows = [
        ("merge_sort", "O(n log n)", "O(n log n)", "O(n log n)", "O(n)", "Да", "Требует доп. буфер O(n)"),
        ("counting_sort", "O(n + k)", "O(n + k)", "O(n + k)", "O(n + k)", "Да", "Целые ключи 0..k, малое k"),
        ("radix_sort_lsd", "O(d(n+b))", "O(d(n+b))", "O(d(n+b))", "O(n + b)", "Да", "Целые >= 0, разрядность d"),
        ("heap_sort", "O(n log n)", "O(n log n)", "O(n log n)", "O(1)", "Нет", "Плохая локальность кэша"),
        ("quick_sort", "O(n log n)", "O(n log n)", "O(n^2)*", "O(log n)", "Нет", "*Рандомизация убирает O(n^2)"),
        ("sorted() (Timsort)", "O(n)", "O(n log n)", "O(n log n)", "O(n)", "Да", "Реализация на C в CPython"),
    ]
    for r in rows:
        print(f"{r[0]:<18} | {r[1]:<10} | {r[2]:<10} | {r[3]:<10} | {r[4]:<10} | {r[5]:<5} | {r[6]}")


def run_benchmarks(seed: int) -> dict:
    rng = random.Random(seed)
    algorithms = [
        ("merge_sort", merge_sort),
        ("counting_sort", lambda a: counting_sort(a, key_max=max(a, default=0))),
        ("radix_sort_lsd", radix_sort_lsd),
        ("heap_sort", heap_sort),
        ("quick_sort (ЛР 3)", quick_sort),
        ("sorted() (эталон)", sorted),
    ]

    dists = ["random_wide", "random_small", "sorted", "reversed", "almost_sorted"]
    results = {d: {name: [] for name, _ in algorithms} for d in dists}

    for n in SIZES:
        datasets = make_datasets(rng, n)
        print(f"\nn = {n}:")
        for dist_name, data in datasets.items():
            print(f"  распределение {dist_name}:")
            for algo_name, fn in algorithms:
                if algo_name == "counting_sort" and dist_name == "random_wide":
                    print(f"    {algo_name:>18}: пропуск (диапазон ключей)")
                    continue
                t = bench(fn, data)
                results[dist_name][algo_name].append((n, t))
                print(f"    {algo_name:>18}: t = {t:.6f} c")

    return results


def plot_results(results: dict, out_dir: Path) -> None:
    """Построение log-log графиков по каждому распределению."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен — график пропущен.")
        return

    dists = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for idx, dist_name in enumerate(dists):
        ax = axes_flat[idx]
        for algo_name, points in results[dist_name].items():
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            slope = log_log_slope(points)
            ax.plot(xs, ys, marker="o", label=f"{algo_name} (k≈{slope:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Распределение: {dist_name}")
        ax.set_xlabel("Размер массива n")
        ax.set_ylabel("Время t, с")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=8)

    # Скрываем шестую пустую панель
    axes_flat[5].axis("off")

    fig.tight_layout()
    out_path = out_dir / "lab04_times.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nГрафик времени сохранён:\n  {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                    help="каталог для сохранения графиков")
    args = ap.parse_args()
    seed = 30 + args.variant
    print(f"Запуск ЛР-04: вариант {args.variant}, seed={seed}")
    random.seed(seed)

    self_check()
    print_properties_table()
    results = run_benchmarks(seed)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()