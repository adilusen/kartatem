"""Перераспределение учебных часов: прозрачная эвристика.

Часы относятся к ОБЩЕМУ плану группы (одна строка = одна тема), а не к ученикам.
Все пороги — настраиваемые допущения демо, не нормативы.

Алгоритм (по шагам):
1. Для каждой темы границы: низ = max(min_hours, hours − max_delta), верх = hours + max_delta.
2. В переносе участвуют только темы с достаточным числом post (отдельное правило).
   Зафиксированные методистом темы («Не изменять тему») сохраняют ИСХОДНЫЕ часы:
   они не бывают ни донорами, ни получателями.
3. Доноры: усвоение ≥ donor_from (по убыванию усвоения). Получатели: усвоение < target
   (по возрастанию усвоения). При равенстве — порядок topic_id.
4. Повторяем: берём 1 час у первого донора, у которого есть запас до нижней границы,
   и отдаём первому получателю, у которого есть место до верхней границы.
5. Стоп, когда донора или получателя не осталось. Это нормальный результат, не ошибка.

Приоритеты считаются по ИСХОДНЫМ значениям усвоения. Алгоритм не предсказывает, как
изменятся баллы после переноса.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd

from .config import Params

PLAN_COLUMNS = ["topic_id", "name", "hours_before", "hours_low", "hours_high",
                "mastery_pct", "eligible", "hours_proposed", "delta", "note"]


@dataclass
class Move:
    step: int
    from_topic: str
    to_topic: str
    reason: str


@dataclass
class HoursPlan:
    table: pd.DataFrame
    moves: list[Move] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    budget: int = 0

    def proposed(self) -> dict[str, int]:
        return {r.topic_id: int(r.hours_proposed) for r in self.table.itertuples(index=False)}

    def hours_moved(self) -> int:
        """Сколько часов переносится: сумма ПОЛОЖИТЕЛЬНЫХ изменений (каждый час учитывается один раз)."""
        return int(sum(d for d in self.table["delta"] if d > 0))

    def flows(self) -> list[tuple[str, str, int]]:
        """Направления переноса: (из темы, в тему, часов), в порядке первого появления в журнале."""
        counts: dict[tuple[str, str], int] = {}
        for m in self.moves:
            counts[(m.from_topic, m.to_topic)] = counts.get((m.from_topic, m.to_topic), 0) + 1
        return [(a, b, n) for (a, b), n in counts.items()]

    def moves_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"Шаг": m.step, "Из темы": m.from_topic, "В тему": m.to_topic,
                              "Причина": m.reason} for m in self.moves],
                            columns=["Шаг", "Из темы", "В тему", "Причина"])


def bounds(hours: int, min_hours: int, max_delta: int) -> tuple[int, int]:
    """(нижняя, верхняя) граница часов темы."""
    return max(min_hours, hours - max_delta), hours + max_delta


def _is_number(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def propose_hours(topics: pd.DataFrame, mastery: Mapping[str, float],
                  eligible: Mapping[str, bool], params: Params,
                  skip_notes: Mapping[str, str] | None = None,
                  locked: Iterable[str] | None = None) -> HoursPlan:
    """topics: topic_id, name, hours, min_hours, max_delta (целые).

    skip_notes — необязательные пояснения, почему тема не участвует (например,
    «выберите форму теста»).
    """
    ids = list(topics["topic_id"])
    names = dict(zip(topics["topic_id"], topics["name"]))
    hours = dict(zip(topics["topic_id"], topics["hours"].astype(int)))
    low, high = {}, {}
    for r in topics.itertuples(index=False):
        low[r.topic_id], high[r.topic_id] = bounds(int(r.hours), int(r.min_hours), int(r.max_delta))
    budget = int(sum(hours.values()))

    def m(t: str) -> float:
        v = mastery.get(t)
        return float(v) if _is_number(v) else float("nan")

    locked_set = set(locked or ()) & set(ids)
    active = [t for t in ids if t not in locked_set and eligible.get(t, False) and not math.isnan(m(t))]
    donors = sorted([t for t in active if m(t) >= params.donor_from], key=lambda t: (-m(t), t))
    receivers = sorted([t for t in active if m(t) < params.target], key=lambda t: (m(t), t))

    cur = dict(hours)
    moves: list[Move] = []
    limit = sum(high[t] - low[t] for t in ids) + 1
    for step in range(1, limit + 1):
        d = next((t for t in donors if cur[t] > low[t]), None)
        r = next((t for t in receivers if cur[t] < high[t] and t != d), None)
        if d is None or r is None:
            break
        cur[d] -= 1
        cur[r] += 1
        moves.append(Move(
            step=step, from_topic=d, to_topic=r,
            reason=(f"{names[d]} (итоговый тест {m(d):.1f}%, запас {m(d) - params.target:.1f} п.п. до цели) → "
                    f"{names[r]} (итоговый тест {m(r):.1f}%, дефицит {params.target - m(r):.1f} п.п.)")))

    messages: list[str] = []
    eligible_any = [t for t in ids if eligible.get(t, False) and not math.isnan(m(t))]
    if locked_set:
        messages.append("Зафиксированы исходные часы тем: "
                        + ", ".join(names[t] for t in ids if t in locked_set) + ".")
    if not eligible_any:
        messages.append("Допустимых переносов нет: ни в одной теме нет достаточного числа post "
                        f"(нужно не меньше {params.min_post_hours} учеников).")
    elif not moves:
        if not active:
            messages.append("Допустимых переносов нет: все темы с достаточными данными зафиксированы.")
        elif not receivers:
            messages.append(f"Допустимых переносов нет: нет тем с усвоением ниже {params.target:g}%.")
        elif not donors:
            messages.append(f"Допустимых переносов нет: нет тем с усвоением от {params.donor_from:g}%.")
        else:
            messages.append("Допустимых переносов нет: границы (минимум часов и допустимое изменение) "
                            "не позволяют ничего перенести.")
    if moves:
        still_low = [names[t] for t in receivers if cur[t] == hours[t]]
        if still_low:
            messages.append("Темы ниже цели, которые не получили часов (нет доноров или достигнут предел): "
                            + ", ".join(still_low) + ".")

    rows = []
    for t in ids:
        delta = cur[t] - hours[t]
        if t in locked_set:
            note = f"зафиксировано: исходные {hours[t]} ч"
        elif t not in active:
            note = ((skip_notes or {}).get(t)
                    or ("не участвует: мало данных post" if not eligible.get(t, False)
                        else "не участвует: нет усвоения"))
        elif delta < 0:
            note = f"донор: {delta:+d} ч"
        elif delta > 0:
            note = f"получатель: {delta:+d} ч"
        elif t in donors and cur[t] == low[t]:
            note = "донор, но нижняя граница достигнута"
        elif t in receivers and cur[t] == high[t]:
            note = "получатель, но верхняя граница достигнута"
        elif t in receivers:
            note = "ниже цели, но доноров не осталось"
        elif t in donors:
            note = "потенциальный донор, перенос не понадобился"
        else:
            note = f"между порогами {params.target:g}–{params.donor_from:g}%: без изменений"
        rows.append({
            "topic_id": t, "name": names[t], "hours_before": hours[t],
            "hours_low": low[t], "hours_high": high[t], "mastery_pct": m(t),
            "eligible": bool(eligible.get(t, False)), "hours_proposed": cur[t],
            "delta": delta, "note": note,
        })
    table = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    assert int(table["hours_proposed"].sum()) == budget  # инвариант: бюджет сохраняется
    return HoursPlan(table=table, moves=moves, messages=messages, budget=budget)


def validate_plan(topics: pd.DataFrame, plan: Mapping[str, Any],
                  locked: Iterable[str] | None = None) -> list[str]:
    """Проверка ручного плана. Пустой список = план корректен.

    Проверяется: все темы заданы, значения конечные и целые, границы темы, сумма = бюджет.
    """
    errors: list[str] = []
    budget = int(topics["hours"].astype(int).sum())
    known = set(topics["topic_id"])
    locked_set = set(locked or ())
    unknown_locked = sorted(locked_set - known)
    if unknown_locked:
        errors.append("Фиксация указана для неизвестных тем: " + ", ".join(map(str, unknown_locked)) + ".")
    extra = [t for t in plan if t not in known]
    if extra:
        errors.append("В плане есть неизвестные темы: " + ", ".join(map(str, extra)) + ".")
    values: dict[str, int] = {}
    for r in topics.itertuples(index=False):
        label = f"«{r.name}»"
        v = plan.get(r.topic_id)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            errors.append(f"Тема {label}: часы не заданы.")
            continue
        if not _is_number(v):
            errors.append(f"Тема {label}: значение «{v}» не является числом.")
            continue
        fv = float(v)
        if fv != math.floor(fv):
            errors.append(f"Тема {label}: часы должны быть целыми, получено {fv:g}.")
            continue
        iv = int(fv)
        lo, hi = bounds(int(r.hours), int(r.min_hours), int(r.max_delta))
        if r.topic_id in locked_set and iv != int(r.hours):
            errors.append(f"Тема {label} зафиксирована на исходных {int(r.hours)} ч, в плане {iv} ч. "
                          "Верните исходное значение или снимите фиксацию.")
        elif iv < lo or iv > hi:
            errors.append(f"Тема {label}: {iv} ч вне допустимого диапазона {lo}–{hi} ч.")
        values[r.topic_id] = iv
    if len(values) == len(topics):
        total = sum(values.values())
        if total != budget:
            errors.append(f"Сумма часов {total} не равна бюджету {budget} (разница {total - budget:+d}).")
    return errors
