#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 7. Сравнительный анализ КМП, Рабина–Карпа и Бойера–Мура (вариант 7)."""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Алгоритмы поиска подстрок (реализуются вручную, без re и str.find
#    в основной части; str.find используется только в эталоне для сверки)
# ---------------------------------------------------------------------------


def naive_search(text: str, pattern: str) -> tuple[list[int], int]:
    """Наивный поиск со сдвигом на одну позицию.

    Временная сложность: O((n - m + 1) * m) в худшем случае, O(n) в среднем
    на большом алфавите. Память: O(1) сверх списка ответа.
    """
    n = len(text)
    m = len(pattern)
    if m == 0 or m > n:
        return [], 0

    occurrences: list[int] = []
    comparisons = 0

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            comparisons += 1
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            occurrences.append(i)

    return occurrences, comparisons


def prefix_function(pattern: str) -> list[int]:
    """Префикс-функция шаблона: pi[i] — длина наибольшего собственного
    префикса подстроки pattern[:i+1], совпадающего с её суффиксом.

    Временная сложность построения: Theta(m), память: Theta(m).
    """
    m = len(pattern)
    pi = [0] * m
    for i in range(1, m):
        j = pi[i - 1]
        while j > 0 and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        pi[i] = j
    return pi


def kmp_search(text: str, pattern: str) -> tuple[list[int], int]:
    """Поиск Кнута–Морриса–Пратта.

    Временная сложность: Theta(n + m) во всех случаях (не более 2n сравнений
    символов текста), память: Theta(m) на префикс-функцию.
    """
    n = len(text)
    m = len(pattern)
    if m == 0 or m > n:
        return [], 0

    pi = prefix_function(pattern)
    occurrences: list[int] = []
    comparisons = 0
    j = 0  # число совпавших символов в текущем префиксе шаблона

    for i in range(n):
        while j > 0:
            comparisons += 1
            if text[i] == pattern[j]:
                break
            j = pi[j - 1]
        else:
            comparisons += 1
            if text[i] == pattern[j]:
                j += 1
            if j == m:
                occurrences.append(i - m + 1)
                j = pi[j - 1]
            continue

        # Вышли из while по break -> совпадение text[i] == pattern[j] уже учтено
        j += 1
        if j == m:
            occurrences.append(i - m + 1)
            j = pi[j - 1]

    return occurrences, comparisons


def rabin_karp_search(
    text: str, pattern: str, base: int = 256, mod: int = 1_000_000_007
) -> tuple[list[int], int]:
    """Поиск Рабина–Карпа с полиномиальным хешем по модулю mod.

    Средняя сложность: O(n + m) при редких коллизиях; худший случай (при сплошных
    совпадениях или коллизиях): O((n - m + 1) * m). Память: O(1).
    """
    n = len(text)
    m = len(pattern)
    if m == 0 or m > n:
        return [], 0

    occurrences: list[int] = []
    comparisons = 0

    # Вычисляем base^(m-1) % mod и стартовые хеши шаблона и первого окна
    high_pow = pow(base, m - 1, mod)
    p_hash = 0
    t_hash = 0
    for j in range(m):
        p_hash = (p_hash * base + ord(pattern[j])) % mod
        t_hash = (t_hash * base + ord(text[j])) % mod

    for i in range(n - m + 1):
        if p_hash == t_hash:
            # Обязательная посимвольная проверка при совпадении хешей
            match = True
            for j in range(m):
                comparisons += 1
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                occurrences.append(i)

        # Скользящий пересчёт хеша для следующего окна за O(1)
        if i < n - m:
            t_hash = (
                (t_hash - ord(text[i]) * high_pow) * base + ord(text[i + m])
            ) % mod

    return occurrences, comparisons


def bad_char_table(pattern: str) -> dict[str, int]:
    """Таблица плохого символа: для каждого символа шаблона — индекс его
    последнего вхождения в pattern. Сложность: Theta(m).
    """
    return {ch: idx for idx, ch in enumerate(pattern)}


def good_suffix_table(pattern: str) -> list[int]:
    """Таблица сдвигов по эвристике хорошего суффикса (уровень «10» рубрики).

    Возвращает массив shift длины m + 1:
    - shift[j + 1] — безопасный сдвиг при несовпадении на позиции j шаблона;
    - shift[0] — безопасный сдвиг после полного совпадения всего шаблона.
    Построение за O(m).
    """
    m = len(pattern)
    shift = [0] * (m + 1)
    border = [0] * (m + 1)

    # Фаза 1: сильное правило хорошего суффикса (внутренние вхождения суффикса)
    i = m
    j = m + 1
    border[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = border[j]
        i -= 1
        j -= 1
        border[i] = j

    # Фаза 2: префикс шаблона, совпадающий с суффиксом хорошего суффикса
    j = border[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = border[j]

    return shift


def boyer_moore_search(text: str, pattern: str) -> tuple[list[int], int]:
    """Поиск Бойера–Мура: сравнение справа налево, максимум сдвигов
    по эвристикам плохого символа и хорошего суффикса.

    Лучший/средний случай на естественном тексте: O(n / m) сублинейно;
    худший случай при поиске всех перекрывающихся вхождений: O(n * m).
    """
    n = len(text)
    m = len(pattern)
    if m == 0 or m > n:
        return [], 0

    bad_char = bad_char_table(pattern)
    good_suffix = good_suffix_table(pattern)
    occurrences: list[int] = []
    comparisons = 0

    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0:
            comparisons += 1
            if pattern[j] != text[i + j]:
                break
            j -= 1

        if j < 0:
            occurrences.append(i)
            # После полного совпадения сдвигаемся по правилу хорошего суффикса (не менее 1)
            i += max(1, good_suffix[0])
        else:
            bc_shift = j - bad_char.get(text[i + j], -1)
            gs_shift = good_suffix[j + 1]
            i += max(1, bc_shift, gs_shift)

    return occurrences, comparisons


# ---------------------------------------------------------------------------
# 2. Репрезентативные тесты и инварианты (шаги 1–3 методики верификации)
# ---------------------------------------------------------------------------


def reference_occurrences(text: str, pattern: str) -> list[int]:
    """Эталон из стандартной библиотеки: все вхождения через str.find
    с шагом 1 (перекрывающиеся вхождения учитываются)."""
    if not pattern:
        return []
    result = []
    pos = text.find(pattern)
    while pos != -1:
        result.append(pos)
        pos = text.find(pattern, pos + 1)
    return result


def self_check() -> None:
    """Граничные и типовые случаи + сверка всех алгоритмов с эталоном."""
    # эталон сам по себе должен вести себя ожидаемо
    assert reference_occurrences("aaaa", "aa") == [0, 1, 2]
    assert reference_occurrences("abc", "abcd") == []

    # префикс-функция: классические примеры
    assert prefix_function("abcabcd") == [0, 0, 0, 1, 2, 3, 0]
    assert prefix_function("aabaaab") == [0, 1, 0, 1, 2, 2, 3]

    algorithms = [naive_search, kmp_search, rabin_karp_search, boyer_moore_search]
    for search in algorithms:
        name = search.__name__
        occ, _ = search("aaaa", "aa")
        assert occ == [0, 1, 2], f"{name}: перекрывающиеся вхождения теряются"
        occ, _ = search("abc", "abc")
        assert occ == [0], f"{name}: шаблон, равный тексту"
        occ, _ = search("abc", "abcd")
        assert occ == [], f"{name}: шаблон длиннее текста"
        occ, _ = search("", "a")
        assert occ == [], f"{name}: пустой текст"
        occ, _ = search("abc", "")
        assert occ == [], f"{name}: пустой шаблон (инвариант заготовки)"

    # Проверка защиты Рабина–Карпа от ложных срабатываний при частых коллизиях (mod=3)
    occ_rk_coll, cmp_rk_coll = rabin_karp_search("ababababa", "aba", base=2, mod=3)
    assert occ_rk_coll == [0, 2, 4, 6] and cmp_rk_coll > 0

    # Сверка с эталоном на случайных строках над малым алфавитом:
    # короткий алфавит провоцирует и перекрытия, и коллизии хешей
    rng = random.Random(0)
    for _ in range(300):
        text = "".join(rng.choice("ab") for _ in range(rng.randint(1, 60)))
        pattern = "".join(rng.choice("ab") for _ in range(rng.randint(1, 5)))
        expected = reference_occurrences(text, pattern)
        for search in algorithms:
            occ, comparisons = search(text, pattern)
            assert occ == expected, (
                f"{search.__name__}: text={text!r}, pattern={pattern!r}, "
                f"получено {occ}, ожидалось {expected}"
            )
            assert comparisons >= 0
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 3. Данные по варианту и бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

TEXT_SIZE = 100_000
PATTERN_LENGTHS = [4, 16, 64]
REPEATS = 5

LEXICON = [
    "data",
    "search",
    "pattern",
    "index",
    "hash",
    "log",
    "event",
    "system",
    "error",
    "server",
    "signal",
    "buffer",
    "stream",
    "packet",
    "record",
]


def make_small_alphabet_text(rng: random.Random, length: int) -> str:
    """Текст над малым алфавитом {a, b} — худший режим для наивного поиска."""
    return "".join(rng.choice("ab") for _ in range(length))


def make_natural_text(rng: random.Random, length: int) -> str:
    """Естественно-подобный текст: случайные слова лексикона через пробел."""
    words = []
    total = 0
    while total < length:
        word = rng.choice(LEXICON)
        words.append(word)
        total += len(word) + 1
    return " ".join(words)[:length]


def plant_pattern(rng: random.Random, text: str, pattern: str, count: int) -> str:
    """Вставить count гарантированных вхождений шаблона в копию текста."""
    chars = list(text)
    for _ in range(count):
        pos = rng.randrange(0, len(chars) - len(pattern) + 1)
        chars[pos : pos + len(pattern)] = pattern
    return "".join(chars)


def bench(search, text: str, pattern: str) -> float:
    """Медиана времени выполнения search(text, pattern) по REPEATS запускам."""
    search(text, pattern)  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        search(text, pattern)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_benchmarks(seed: int) -> dict:
    rng = random.Random(seed)
    texts = {
        "малый алфавит (ab)": make_small_alphabet_text(rng, TEXT_SIZE),
        "естественно-подобный": make_natural_text(rng, TEXT_SIZE),
    }
    algorithms = [
        ("наивный", naive_search),
        ("КМП", kmp_search),
        ("Рабин–Карп", rabin_karp_search),
        ("Бойер–Мур", boyer_moore_search),
    ]
    results: dict = {
        t_name: {a_name: [] for a_name, _ in algorithms} for t_name in texts
    }

    for text_name, text in texts.items():
        for m in PATTERN_LENGTHS:
            start = rng.randrange(0, len(text) - m + 1)
            pattern = text[start : start + m]
            sample = plant_pattern(rng, text, pattern, 5)
            print(f"\nТекст: {text_name}, |T| = {len(sample)}, |P| = {m}")
            for name, search in algorithms:
                occurrences, comparisons = search(sample, pattern)
                t = bench(search, sample, pattern)
                results[text_name][name].append((m, len(occurrences), comparisons, t))
                print(
                    f"  {name:<12} вхождений = {len(occurrences):>5}  "
                    f"сравнений = {comparisons:>10}  t = {t:.6f} c"
                )
    return results


def plot_results(results: dict, out_dir: Path) -> None:
    """Построение графиков числа сравнений и времени в зависимости от длины шаблона."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен — график пропущен.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    markers = {"наивный": "s", "КМП": "o", "Рабин–Карп": "d", "Бойер–Мур": "^"}

    for col, text_name in enumerate(results.keys()):
        ax_cmp = axes[0, col]
        ax_time = axes[1, col]

        for algo_name, pts in results[text_name].items():
            ms = [p[0] for p in pts]
            cmps = [p[2] for p in pts]
            ts = [p[3] for p in pts]
            ax_cmp.plot(ms, cmps, marker=markers[algo_name], label=algo_name)
            ax_time.plot(ms, ts, marker=markers[algo_name], label=algo_name)

        ax_cmp.set_xscale("log", base=2)
        ax_cmp.set_yscale("log")
        ax_cmp.set_xticks(PATTERN_LENGTHS)
        ax_cmp.set_xticklabels([str(m) for m in PATTERN_LENGTHS])
        ax_cmp.set_title(f"Сравнения символов от |P| — {text_name}")
        ax_cmp.set_xlabel("Длина шаблона |P|")
        ax_cmp.set_ylabel("Число сравнений символов (лог. шкала)")
        ax_cmp.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_cmp.legend()

        ax_time.set_xscale("log", base=2)
        ax_time.set_yscale("log")
        ax_time.set_xticks(PATTERN_LENGTHS)
        ax_time.set_xticklabels([str(m) for m in PATTERN_LENGTHS])
        ax_time.set_title(f"Время поиска от |P| — {text_name}")
        ax_time.set_xlabel("Длина шаблона |P|")
        ax_time.set_ylabel("Время t, с (лог. шкала)")
        ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax_time.legend()

    fig.tight_layout()
    out_path = out_dir / "lab07_substring.png"
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
    print(f"Запуск ЛР-07: вариант {args.variant}, seed={seed}")
    random.seed(seed)
    self_check()
    results = run_benchmarks(seed)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()
