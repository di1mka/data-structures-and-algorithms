#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 5. Бинарное дерево поиска: обходы, вставка и удаление (вариант 7)."""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path

# Увеличиваем лимит рекурсии для безопасной работы на глубоких деревьях (в т. ч. в ДЗ 2)
sys.setrecursionlimit(max(sys.getrecursionlimit(), 200_000))

# ---------------------------------------------------------------------------
# 1. Бинарное дерево поиска (реализуется вручную, без сторонних библиотек)
# ---------------------------------------------------------------------------


class Node:
    """Узел BST: ключ и ссылки на левое и правое поддеревья."""

    __slots__ = ("key", "left", "right")

    def __init__(self, key: int) -> None:
        self.key = key
        self.left: Node | None = None
        self.right: Node | None = None


class BST:
    """Бинарное дерево поиска без дубликатов ключей.

    Инвариант: для каждого узла все ключи левого поддерева строго меньше key,
    все ключи правого поддерева — строго больше key.
    Ожидаемая сложность insert / search / delete: O(h), где h — высота дерева
    (O(log n) в среднем на случайных ключах и O(n) в худшем случае при вырождении).
    """

    def __init__(self) -> None:
        self.root: Node | None = None

    def insert(self, key: int) -> None:
        """Вставка ключа за O(h); повторная вставка существующего ключа игнорируется."""
        if self.root is None:
            self.root = Node(key)
            return

        cur = self.root
        while True:
            if key < cur.key:
                if cur.left is None:
                    cur.left = Node(key)
                    return
                cur = cur.left
            elif key > cur.key:
                if cur.right is None:
                    cur.right = Node(key)
                    return
                cur = cur.right
            else:
                return  # Дубликат игнорируется

    def search(self, key: int) -> bool:
        """Возвращает True, если ключ есть в дереве. Сложность: O(h)."""
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False

    def delete(self, key: int) -> None:
        """Удаление ключа за O(h) с обработкой трёх случаев:

        1) лист — отцепляется от родителя (возвращается None);
        2) один ребёнок — узел заменяется единственным ребёнком;
        3) два ребёнка — ключ заменяется преемником (минимум правого
           поддерева), после чего преемник удаляется из правого поддерева.
        """

        def _delete(node: Node | None, k: int) -> Node | None:
            if node is None:
                return None
            if k < node.key:
                node.left = _delete(node.left, k)
            elif k > node.key:
                node.right = _delete(node.right, k)
            else:
                # Случай 1 (нет детей) и Случай 2 (один ребёнок)
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left

                # Случай 3 (два ребёнка): ищем преемника — минимум в правом поддереве
                succ = node.right
                while succ.left is not None:
                    succ = succ.left
                node.key = succ.key
                node.right = _delete(node.right, succ.key)
            return node

        self.root = _delete(self.root, key)

    # -- Обходы --------------------------------------------------------------

    def in_order(self) -> list[int]:
        """Симметричный обход (лево — узел — право): рекурсивно за O(n)."""
        res: list[int] = []

        def _dfs(node: Node | None) -> None:
            if node is None:
                return
            _dfs(node.left)
            res.append(node.key)
            _dfs(node.right)

        _dfs(self.root)
        return res

    def pre_order(self) -> list[int]:
        """Прямой обход (узел — лево — право): рекурсивно за O(n)."""
        res: list[int] = []

        def _dfs(node: Node | None) -> None:
            if node is None:
                return
            res.append(node.key)
            _dfs(node.left)
            _dfs(node.right)

        _dfs(self.root)
        return res

    def post_order(self) -> list[int]:
        """Обратный обход (лево — право — узел): рекурсивно за O(n)."""
        res: list[int] = []

        def _dfs(node: Node | None) -> None:
            if node is None:
                return
            _dfs(node.left)
            _dfs(node.right)
            res.append(node.key)

        _dfs(self.root)
        return res

    def level_order(self) -> list[int]:
        """Обход в ширину по уровням: итеративно через очередь (deque) за O(n)."""
        if self.root is None:
            return []
        res: list[int] = []
        queue: deque[Node] = deque([self.root])
        while queue:
            node = queue.popleft()
            res.append(node.key)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        return res

    # -- Характеристики дерева ----------------------------------------------

    def height(self) -> int:
        """Высота дерева: число рёбер на самом длинном пути от корня.

        Пустое дерево имеет высоту -1, дерево из одного узла — 0.
        """

        def _height(node: Node | None) -> int:
            if node is None:
                return -1
            return 1 + max(_height(node.left), _height(node.right))

        return _height(self.root)

    def size(self) -> int:
        """Число узлов в дереве за O(n)."""

        def _size(node: Node | None) -> int:
            if node is None:
                return 0
            return 1 + _size(node.left) + _size(node.right)

        return _size(self.root)


# ---------------------------------------------------------------------------
# 2. Верификация: инвариант BST и сверка с эталоном (docs/ai-verification.md)
# ---------------------------------------------------------------------------


def self_check() -> None:
    """Граничные случаи + сверка операций с эталоном set/sorted."""
    # Пустое дерево
    t = BST()
    assert t.search(1) is False
    assert t.in_order() == []
    assert t.pre_order() == []
    assert t.post_order() == []
    assert t.level_order() == []
    assert t.height() == -1 and t.size() == 0
    t.delete(1)  # удаление из пустого дерева не должно падать

    # Один узел
    t.insert(5)
    assert t.search(5) and not t.search(4)
    assert t.height() == 0 and t.size() == 1

    # Три случая удаления на маленьком дереве: 5, 2, 8, 1, 3
    for k in (2, 8, 1, 3):
        t.insert(k)
    t.delete(1)  # лист
    assert t.in_order() == [2, 3, 5, 8]
    t.delete(2)  # узел с одним ребёнком
    assert t.in_order() == [3, 5, 8]
    t.delete(5)  # корень с двумя детьми (через преемника)
    assert t.in_order() == [3, 8]

    # Обходы на известном дереве 4-2-6-1-3-5-7 (вставка по уровням)
    t2 = BST()
    for k in (4, 2, 6, 1, 3, 5, 7):
        t2.insert(k)
    assert t2.in_order() == [1, 2, 3, 4, 5, 6, 7]
    assert t2.pre_order() == [4, 2, 1, 3, 6, 5, 7]
    assert t2.post_order() == [1, 3, 2, 5, 7, 6, 4]
    assert t2.level_order() == [4, 2, 6, 1, 3, 5, 7]
    assert t2.height() == 2 and t2.size() == 7

    # Случайные последовательности команд: сверка с эталоном из stdlib
    rng = random.Random(0)
    for _ in range(100):
        t3 = BST()
        etalon: set[int] = set()
        for _ in range(rng.randint(1, 80)):
            k = rng.randint(-30, 30)
            if rng.random() < 0.6:
                t3.insert(k)
                etalon.add(k)
            else:
                t3.delete(k)
                etalon.discard(k)
            # Инвариант BST: in-order даёт отсортированную последовательность
            assert t3.in_order() == sorted(etalon)
        for k in range(-30, 31):
            assert t3.search(k) == (k in etalon)
        assert t3.size() == len(etalon)
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 3. Бенчмарк: поиск в BST против поиска в списке (docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
QUERIES = 1_000  # число поисков на один замер


def bench(fn) -> float:
    """Медиана времени выполнения fn() по REPEATS запускам, с прогревом."""
    fn()  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
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


def run_benchmarks(seed: int) -> list[tuple[int, int, float, float]]:
    """Сравнение времени QUERIES поисков в BST и в списке (оператор in)."""
    rng = random.Random(seed)
    print(
        f"\n{'n':>8} {'h(BST)':>7} {'log2(n)':>8} {'1.44log2':>9} {'t_bst, c':>10} {'t_list, c':>10} {'Ускорение':>10}"
    )
    print("-" * 70)
    rows: list[tuple[int, int, float, float]] = []

    for n in SIZES:
        keys = rng.sample(range(10 * n), n)  # случайные ключи без повторов
        queries = [rng.randint(0, 10 * n) for _ in range(QUERIES)]
        tree = BST()
        for k in keys:
            tree.insert(k)

        def search_bst() -> None:
            for q in queries:
                tree.search(q)

        def search_list() -> None:
            for q in queries:
                q in keys  # линейный поиск для сравнения

        h = tree.height()
        t_bst = bench(search_bst)
        t_list = bench(search_list)
        speedup = t_list / t_bst if t_bst > 0 else 0.0
        log2_n = math.log2(n)
        avl_bound = 1.44 * log2_n
        rows.append((n, h, t_bst, t_list))
        print(
            f"{n:>8} {h:>7} {log2_n:>8.2f} {avl_bound:>9.2f} {t_bst:>10.6f} {t_list:>10.6f} {speedup:>9.1f}x"
        )

    return rows


def plot_results(rows: list[tuple[int, int, float, float]], out_dir: Path) -> None:
    """Построение графика времени поиска (log-log) и графика высоты дерева."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен — график пропущен.")
        return

    ns = [r[0] for r in rows]
    hs = [r[1] for r in rows]
    t_bsts = [r[2] for r in rows]
    t_lists = [r[3] for r in rows]

    slope_bst = log_log_slope(list(zip(ns, t_bsts)))
    slope_list = log_log_slope(list(zip(ns, t_lists)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Log-log график времени поиска
    ax1.plot(ns, t_bsts, marker="o", label=f"Поиск в BST (k ≈ {slope_bst:.2f})")
    ax1.plot(ns, t_lists, marker="s", label=f"Поиск в list `in` (k ≈ {slope_list:.2f})")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title(f"Время {QUERIES} поисков: BST против list (log-log)")
    ax1.set_xlabel("Размер структуры n")
    ax1.set_ylabel("Время выполнения t, с")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.legend()

    # 2. График высоты дерева от log2(n)
    log2_vals = [math.log2(n) for n in ns]
    avl_vals = [1.44 * math.log2(n) for n in ns]
    ax2.plot(ns, hs, marker="o", label="Экспериментальная высота h(BST)")
    ax2.plot(
        ns,
        avl_vals,
        linestyle="--",
        marker="^",
        label="Верхняя граница AVL: 1.44·log2(n)",
    )
    ax2.plot(
        ns, log2_vals, linestyle=":", marker="d", label="Идеальная высота: log2(n)"
    )
    ax2.set_xscale("log")
    ax2.set_title("Рост высоты случайного BST")
    ax2.set_xlabel("Число узлов n (лог. шкала)")
    ax2.set_ylabel("Высота дерева h")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.legend()

    fig.tight_layout()
    out_path = out_dir / "lab05_search.png"
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
    print(f"Запуск ЛР-05: вариант {args.variant}, seed={seed}")
    random.seed(seed)
    self_check()
    rows = run_benchmarks(seed)
    plot_results(rows, args.out)


if __name__ == "__main__":
    main()
