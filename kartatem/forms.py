"""Единый выбор формы теста (test_id) по каждой теме.

Правило, общее для ВСЕХ расчётов (метрики тем, тепловая карта, сравнение, часы, экспорт):
* форма в теме одна → выбирается автоматически;
* форм несколько, выбор сделан → считаем только по выбранной форме;
* форм несколько, выбор не сделан → тема «ждёт выбора»: по ней нет ни средних, ни флагов,
  часы автоматически не меняются. Другие темы при этом считаются как обычно.

Одинаковый test_id — проверка формата, а не доказательство эквивалентности измерения.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

MODE_AUTO = "auto"
MODE_CHOSEN = "chosen"
MODE_NEEDS = "needs_choice"
MODE_NONE = "none"          # по теме нет ни одной валидной записи


@dataclass(frozen=True)
class FormInfo:
    topic_id: str
    forms: tuple[str, ...]
    selected: str | None
    mode: str


def resolve_forms(clean: pd.DataFrame, topic_ids: Iterable[str],
                  test_choice: Mapping[str, str] | None = None) -> dict[str, FormInfo]:
    """Для каждой темы: какие формы есть в данных и какая выбрана."""
    choice = test_choice or {}
    present: dict[str, tuple[str, ...]] = {}
    if len(clean):
        for tid, sub in clean.groupby("topic_id"):
            present[tid] = tuple(sorted(sub["test_id"].unique()))
    out: dict[str, FormInfo] = {}
    for tid in topic_ids:
        forms = present.get(tid, ())
        if not forms:
            out[tid] = FormInfo(tid, (), None, MODE_NONE)
        elif len(forms) == 1:
            out[tid] = FormInfo(tid, forms, forms[0], MODE_AUTO)
        elif choice.get(tid) in forms:
            out[tid] = FormInfo(tid, forms, choice[tid], MODE_CHOSEN)
        else:
            out[tid] = FormInfo(tid, forms, None, MODE_NEEDS)
    return out


def used_rows(clean: pd.DataFrame, infos: Mapping[str, FormInfo]) -> pd.DataFrame:
    """Записи, которые реально участвуют в расчётах: только выбранная форма каждой темы."""
    if clean.empty:
        return clean
    wanted = clean["topic_id"].map({t: (i.selected or "") for t, i in infos.items()}).fillna("")
    return clean[clean["test_id"] == wanted]


def form_key(infos: Mapping[str, FormInfo]) -> tuple:
    """Устойчивая сигнатура текущего выбора (для сброса черновика плана)."""
    return tuple((t, i.selected or "") for t, i in sorted(infos.items()))
