"""Параметры демо.

ВАЖНО: все пороги здесь — редактируемые параметры демонстрационной версии.
Это не научные критерии и не государственные нормативы.
"""
from __future__ import annotations

from dataclasses import dataclass, astuple


@dataclass(frozen=True)
class Params:
    # --- Анализ тем ---
    min_post_topic: int = 15          # мин. учеников с post, чтобы оценивать тему
    low_score_cut: float = 50.0       # «низкий результат»: процент ниже этого значения
    hard_mean_below: float = 60.0     # трудная тема: средний post ниже, %
    hard_share_at_least: float = 40.0  # ...или доля учеников ниже порога не меньше, %

    # --- Сравнение учебников ---
    min_pairs_compare: int = 30       # мин. пар pre/post в каждой группе (по теме)
    min_classes_compare: int = 2      # мин. классов в каждой группе (среди пар)
    pre_gap_warn_pp: float = 10.0     # предупреждение, если pre групп различается больше, п.п.

    # --- Часы ---
    target: float = 70.0              # цель усвоения, %: получатели — ниже неё
    donor_from: float = 75.0          # доноры — от этого значения, %
    min_post_hours: int = 15          # мин. учеников с post для автоматического переноса часов

    def key(self) -> tuple:
        return astuple(self)


# Подписи для интерфейса: (поле, подпись, тип, мин, макс, шаг, подсказка)
PARAM_UI = [
    ("min_post_topic", "Мин. учеников с post для оценки темы", int, 1, 10_000, 1,
     "Если учеников меньше — тема получает статус «Мало данных», флаг трудности не ставится."),
    ("low_score_cut", "Порог «низкого результата», %", float, 0.0, 100.0, 1.0,
     "Доля учеников, у которых процент ниже этого порога."),
    ("hard_mean_below", "Трудная тема: средний post ниже, %", float, 0.0, 100.0, 1.0, ""),
    ("hard_share_at_least", "Трудная тема: доля ниже порога от, %", float, 0.0, 100.0, 1.0, ""),
    ("min_pairs_compare", "Мин. пар pre/post в группе (сравнение)", int, 1, 10_000, 1,
     "Нужно в каждой из двух сравниваемых групп учебников, по каждой теме."),
    ("min_classes_compare", "Мин. классов в группе (сравнение)", int, 1, 1_000, 1,
     "Считаются классы, из которых есть хотя бы одна пара pre/post."),
    ("pre_gap_warn_pp", "Предупреждение о разнице pre, п.п.", float, 0.0, 100.0, 1.0,
     "Только предупреждение. Не доказывает и не опровергает сопоставимость групп."),
    ("target", "Целевой результат post (получатели ниже), %", float, 0.0, 100.0, 1.0, ""),
    ("donor_from", "Доноры часов: результат post от, %", float, 0.0, 100.0, 1.0, ""),
    ("min_post_hours", "Мин. учеников с post для переноса часов", int, 1, 10_000, 1,
     "Отдельное правило: тема без достаточного числа post автоматически не меняется."),
]


def validate_params(p: Params) -> list[str]:
    """Возвращает список проблем с параметрами (пустой — всё в порядке)."""
    problems: list[str] = []
    for name in ("min_post_topic", "min_pairs_compare", "min_classes_compare", "min_post_hours"):
        v = getattr(p, name)
        if not isinstance(v, int) or v < 1:
            problems.append(f"Параметр {name} должен быть целым числом не меньше 1.")
    for name in ("low_score_cut", "hard_mean_below", "hard_share_at_least", "target", "donor_from"):
        v = getattr(p, name)
        if not (0 <= v <= 100):
            problems.append(f"Параметр {name} должен быть в диапазоне 0–100.")
    if p.pre_gap_warn_pp < 0:
        problems.append("Параметр pre_gap_warn_pp не может быть отрицательным.")
    if p.donor_from < p.target:
        problems.append(
            "Порог доноров не может быть ниже цели усвоения: иначе одна тема была бы "
            "одновременно и донором, и получателем."
        )
    return problems
