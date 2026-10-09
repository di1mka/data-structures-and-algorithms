#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 3. Простые сортировки и QuickSort с рандомизацией — реализация (вариант 7)."""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Простые сортировки со счётчиками (сортируют КОПИЮ входа, вход не меняют)
#    Каждая функция возвращает кортеж: (отсортированный список,
#    число сравнений ключей, число обменов/перемещений элементов).
# ---------------------------------------------------------------------------


def bubble_sort(a: list, key=None) -> tuple[list, int, int]:
    """Сортировка пузырьком с флагом досрочного выхода.

    Худший случай: O(n^2) сравнений и обменов.
    Лучший случай (упорядоченный вход): O(n) сравнений, 0 обменов.
    """
    res = list(a)
    n = len(res)
    comps = 0
    swaps = 0
    if key is None:
        for i in range(n):
            swapped = False
            for j in range(0, n - 1 - i):
                comps += 1
                if res[j] > res[j + 1]:
                    res[j], res[j + 1] = res[j + 1], res[j]
                    swaps += 1
                    swapped = True
            if not swapped:
                break
    else:
        for i in range(n):
            swapped = False
            for j in range(0, n - 1 - i):
                comps += 1
                if key(res[j]) > key(res[j + 1]):
                    res[j], res[j + 1] = res[j + 1], res[j]
                    swaps += 1
                    swapped = True
            if not swapped:
                break
    return res, comps, swaps


def insertion_sort(a: list, key=None) -> tuple[list, int, int]:
    """Сортировка вставками.

    Худший случай (обратный порядок): O(n^2) сравнений и сдвигов.
    Лучший случай (упорядоченный вход): O(n) сравнений, 0 сдвигов.
    """
    res = list(a)
    n = len(res)
    comps = 0
    moves = 0
    if key is None:
        for i in range(1, n):
            cur = res[i]
            j = i - 1
            while j >= 0:
                comps += 1
                if res[j] > cur:
                    res[j + 1] = res[j]
                    moves += 1
                    j -= 1
                else:
                    break
            res[j + 1] = cur
    else:
        for i in range(1, n):
            cur = res[i]
            cur_k = key(cur)
            j = i - 1
            while j >= 0:
                comps += 1
                if key(res[j]) > cur_k:
                    res[j + 1] = res[j]
                    moves += 1
                    j -= 1
                else:
                    break
            res[j + 1] = cur
    return res, comps, moves


def selection_sort(a: list, key=None) -> tuple[list, int, int]:
    """Сортировка выбором.

    Сложность по сравнениям строго Theta(n^2) на любом входе: n*(n-1)/2.
    Число обменов: не более n - 1.
    """
    res = list(a)
    n = len(res)
    comps = 0
    swaps = 0
    if key is None:
        for i in range(n - 1):
            min_idx = i
            for j in range(i + 1, n):
                comps += 1
                if res[j] < res[min_idx]:
                    min_idx = j
            if min_idx != i:
                res[i], res[min_idx] = res[min_idx], res[i]
                swaps += 1
    else:
        for i in range(n - 1):
            min_idx = i
            min_k = key(res[min_idx])
            for j in range(i + 1, n):
                comps += 1
                val_k = key(res[j])
                if val_k < min_k:
                    min_idx = j
                    min_k = val_k
            if min_idx != i:
                res[i], res[min_idx] = res[min_idx], res[i]
                swaps += 1
    return res, comps, swaps


# ---------------------------------------------------------------------------
# 2. QuickSort с рандомизированным опорным элементом
# ---------------------------------------------------------------------------


def quick_sort(a: list, rng: random.Random | None = None, key=None) -> list:
    """QuickSort с выбором опорного элемента через rng.randrange.

    Ожидаемая сложность: O(n log n) в среднем за счёт балансировки разбиений.
    Использует 3-way разбиение (флаг Дейкстры) и элиминацию хвостовой рекурсии.
    """
    if rng is None:
        rng = random.Random()
    res = list(a)
    n = len(res)
    if n <= 1:
        return res

    if key is None:
        def _qsort(lo: int, hi: int) -> None:
            while lo < hi:
                pivot_idx = rng.randrange(lo, hi + 1)
                pivot_val = res[pivot_idx]

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

                # Рекурсивно вызываем для меньшей части, итерируем по большей (O(log n) глубина стека)
                if (lt - 1 - lo) < (hi - (gt + 1)):
                    _qsort(lo, lt - 1)
                    lo = gt + 1
                else:
                    _qsort(gt + 1, hi)
                    hi = lt - 1
    else:
        def _qsort(lo: int, hi: int) -> None:
            while lo < hi:
                pivot_idx = rng.randrange(lo, hi + 1)
                pivot_val = key(res[pivot_idx])

                lt = lo
                gt = hi
                i = lo
                while i <= gt:
                    val = key(res[i])
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
# 3. Демонстрация стабильности на парах (ключ, метка)
# ---------------------------------------------------------------------------


class Pair:
    """Элемент пары (ключ, метка), сравнение производится ТОЛЬКО по ключу."""

    __slots__ = ("key", "tag")

    def __init__(self, key: int, tag: str) -> None:
        self.key = key
        self.tag = tag

    def __lt__(self, other: Pair) -> bool:
        return self.key < other.key

    def __le__(self, other: Pair) -> bool:
        return self.key <= other.key

    def __gt__(self, other: Pair) -> bool:
        return self.key > other.key

    def __ge__(self, other: Pair) -> bool:
        return self.key >= other.key

    def __eq__(self, other: Pair) -> bool:
        return self.key == other.key

    def __repr__(self) -> str:
        return f"({self.key}, '{self.tag}')"


def stability_demo() -> None:
    """Показать, какие из четырёх сортировок стабильны."""
    pairs = [
        Pair(2, "a"),
        Pair(1, "b"),
        Pair(2, "c"),
        Pair(1, "d"),
        Pair(2, "e"),
    ]
    print("\n--- Демонстрация стабильности на парах (ключ, метка) ---")
    print(f"Исходный массив: {pairs}")

    res_b, _, _ = bubble_sort(pairs)
    stable_b = [p.tag for p in res_b if p.key == 2] == ["a", "c", "e"]
    print(f"Bubble Sort:    {res_b} -> {'СТАБИЛЬНА' if stable_b else 'НЕСТАБИЛЬНА'}")

    res_i, _, _ = insertion_sort(pairs)
    stable_i = [p.tag for p in res_i if p.key == 2] == ["a", "c", "e"]
    print(f"Insertion Sort: {res_i} -> {'СТАБИЛЬНА' if stable_i else 'НЕСТАБИЛЬНА'}")

    res_s, _, _ = selection_sort(pairs)
    stable_s = [p.tag for p in res_s if p.key == 2] == ["a", "c", "e"]
    print(f"Selection Sort: {res_s} -> {'СТАБИЛЬНА' if stable_s else 'НЕСТАБИЛЬНА'}")

    res_q = quick_sort(pairs, random.Random(0))
    stable_q = [p.tag for p in res_q if p.key == 2] == ["a", "c", "e"]
    print(f"QuickSort:      {res_q} -> {'СТАБИЛЬНА' if stable_q else 'НЕСТАБИЛЬНА'}\n")


# ---------------------------------------------------------------------------
# 4. Классы входов (данные по варианту, детерминированно по seed)
# ---------------------------------------------------------------------------


def make_inputs(n: int, seed: int) -> dict[str, list[int]]:
    """Три класса входов размера n: упорядоченный, случайный, обратный."""
    rng = random.Random(seed + n)
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]
    return {
        "упорядоченный": sorted(data),
        "случайный": data,
        "обратный": sorted(data, reverse=True),
    }


# ---------------------------------------------------------------------------
# 5. Верификация (шаги 1–3 методики docs/ai-verification.md)
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "bubble_sort": lambda a: bubble_sort(a)[0],
    "insertion_sort": lambda a: insertion_sort(a)[0],
    "selection_sort": lambda a: selection_sort(a)[0],
    "quick_sort": lambda a: quick_sort(a, random.Random(0)),
}


def is_sorted(a: list) -> bool:
    """Инвариант 1: неубывающий порядок элементов."""
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def self_check() -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],                    # пустой массив
        [7],                   # один элемент
        [5, 5, 5, 5],          # все элементы равны
        [1, 2, 3, 4, 5],       # уже отсортирован
        [5, 4, 3, 2, 1],       # обратный порядок
    ]
    for name, fn in ALGORITHMS.items():
        for a in boundary:
            res = fn(list(a))
            assert is_sorted(res), f"{name}: нарушен порядок на {a}"
            assert sorted(res) == sorted(a), f"{name}: не перестановка входа {a}"

    # Сверка с эталоном из стандартной библиотеки на сотнях случайных входов.
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randint(-100, 100) for _ in range(rng.randint(0, 80))]
        expected = sorted(a)
        for name, fn in ALGORITHMS.items():
            assert fn(list(a)) == expected, f"{name}: расходится с sorted() на {a}"

    # Счётчики: у сортировки выбором ровно n*(n-1)/2 сравнений на любом входе.
    _, comparisons, _ = selection_sort([3, 1, 2, 5, 4])
    assert comparisons == 10, "selection_sort: неверный счётчик сравнений"
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 6. Бенчмарк и графики
# ---------------------------------------------------------------------------

SIZES_QUADRATIC = [500, 1_000, 2_000, 4_000, 8_000]
SIZES_QUICK = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5


def bench(fn, data: list) -> float:
    """Медиана времени выполнения fn(копия data) по REPEATS запускам, с прогревом."""
    fn(list(data))
    times = []
    for _ in range(REPEATS):
        arg = list(data)
        t0 = time.perf_counter()
        fn(arg)
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


def print_counters_table(seed: int) -> None:
    """Вывод таблицы счётчиков операций для простых сортировок."""
    print("\n--- Таблица операций (сравнения и обмены/сдвиги) на n = 1000 ---")
    print(f"{'Алгоритм':<16} {'Класс входа':<15} {'Сравнения':>12} {'Обмены / сдвиги':>16}")
    print("-" * 63)
    sorts = [
        ("bubble_sort", bubble_sort),
        ("insertion_sort", insertion_sort),
        ("selection_sort", selection_sort),
    ]
    for name, sort_fn in sorts:
        inputs = make_inputs(1000, seed)
        for cls, data in inputs.items():
            _, comps, moves = sort_fn(data)
            print(f"{name:<16} {cls:<15} {comps:>12} {moves:>16}")


def run_benchmarks(seed: int) -> dict:
    """Замеры всех алгоритмов на трёх классах входов."""
    plans = [
        ("bubble_sort", ALGORITHMS["bubble_sort"], SIZES_QUADRATIC),
        ("insertion_sort", ALGORITHMS["insertion_sort"], SIZES_QUADRATIC),
        ("selection_sort", ALGORITHMS["selection_sort"], SIZES_QUADRATIC),
        ("quick_sort", lambda a: quick_sort(a, random.Random(seed)), SIZES_QUICK),
    ]

    bench_results = {"упорядоченный": {}, "случайный": {}, "обратный": {}}

    for name, fn, sizes in plans:
        print(f"\n{name}:")
        for cls in ("упорядоченный", "случайный", "обратный"):
            points = []
            for n in sizes:
                data = make_inputs(n, seed)[cls]
                t = bench(fn, data)
                points.append((n, t))
                print(f"  n={n:>7}  вход={cls:<13} t={t:.6f} c")
            bench_results[cls][name] = points

    return bench_results


def plot_results(bench_results: dict, out_dir: Path) -> None:
    """Построение трёх log-log графиков (по классам входов) и сохранение изображения."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен — график пропущен.")
        return

    classes = ["упорядоченный", "случайный", "обратный"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, cls in zip(axes, classes):
        for name, points in bench_results[cls].items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            slope = log_log_slope(points)
            ax.plot(xs, ys, marker="o", label=f"{name} (наклон ≈ {slope:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Класс входа: {cls}")
        ax.set_xlabel("Размер массива n")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Время выполнения t, с")
    fig.tight_layout()
    out_path = out_dir / "lab03_times.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nГрафик времени сохранён:\n  {out_path}")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                    help="каталог для графиков (по умолчанию attachments)")
    args = ap.parse_args()
    seed = 30 + args.variant
    print(f"Запуск ЛР-03: вариант {args.variant}, seed={seed}")
    random.seed(seed)

    self_check()
    stability_demo()
    print_counters_table(seed)
    results = run_benchmarks(seed)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()