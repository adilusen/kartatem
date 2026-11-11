"""Показатели по темам. Все проценты — на общей шкале 0–100 (100 × score / max_score).

Все показатели считаются ТОЛЬКО по выбранной форме теста каждой темы (см. forms.py).
Тема, где форм несколько, а выбор не сделан, показателей не получает.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from .config import Params
from .forms import MODE_NEEDS, form_key, resolve_forms, used_rows

STATUS_HARD = "Трудная"
STATUS_OK = "Без флага"
STATUS_LOW = "Мало данных"
STATUS_CHOOSE = "Выберите форму теста"

TOPIC_COLUMNS = ["topic_id", "name", "selected_test", "form_mode", "n_test_forms",
                 "n_students", "n_classes", "n_post_rows", "n_post_file", "n_pre_rows",
                 "mean_post", "share_below", "hours_eligible", "status", "reason"]


def _student_level(post_rows: pd.DataFrame) -> pd.DataFrame:
    """Одна строка = один ученик (class_id + student_id) в теме.

    Метрики считаются именно по ученикам: класс с 30 учениками весит втрое больше
    класса с 10 — мы НЕ усредняем средние классов.
    """
    return post_rows.groupby("uid", as_index=False).agg(pct=("pct", "mean"), class_id=("class_id", "first"))


def classify_topic(n: int, mean: float, share: float, params: Params) -> tuple[str, str]:
    """Статус темы и пояснение. При малом n флаг трудности не ставится."""
    if n < params.min_post_topic:
        return STATUS_LOW, f"учеников с post: {n} < {params.min_post_topic}"
    reasons = []
    if mean < params.hard_mean_below:
        reasons.append(f"средний post {mean:.1f}% < {params.hard_mean_below:g}%")
    if share >= params.hard_share_at_least:
        reasons.append(f"доля ниже {params.low_score_cut:g}%: {share:.1f}% ≥ {params.hard_share_at_least:g}%")
    if reasons:
        return STATUS_HARD, "; ".join(reasons)
    return STATUS_OK, "пороги трудности не достигнуты"


def topic_metrics(clean: pd.DataFrame, topics: pd.DataFrame, params: Params,
                  test_choice: Mapping[str, str] | None = None) -> pd.DataFrame:
    infos = resolve_forms(clean, topics["topic_id"], test_choice)
    used = used_rows(clean, infos)
    post = used[used["stage"] == "post"]
    pre = used[used["stage"] == "pre"]
    post_file = clean[clean["stage"] == "post"]
    rows = []
    for t in topics.itertuples(index=False):
        info = infos[t.topic_id]
        sub = post[post["topic_id"] == t.topic_id]
        per_student = _student_level(sub) if len(sub) else pd.DataFrame(columns=["pct", "class_id"])
        n = int(len(per_student))
        mean = float(per_student["pct"].mean()) if n else float("nan")
        share = float(100.0 * (per_student["pct"] < params.low_score_cut).mean()) if n else float("nan")
        if info.mode == MODE_NEEDS:
            status = STATUS_CHOOSE
            reason = ("в теме несколько форм теста (" + ", ".join(info.forms) +
                      "): выберите одну, объединённое среднее не считается")
            eligible = False
        else:
            status, reason = classify_topic(n, mean, share, params)
            # Отдельное правило допуска к автоматическому переносу часов:
            eligible = bool(n >= params.min_post_hours)
        rows.append({
            "topic_id": t.topic_id,
            "name": t.name,
            "selected_test": info.selected or "",
            "form_mode": info.mode,
            "n_test_forms": len(info.forms),
            "n_students": n,
            "n_classes": int(sub["class_id"].nunique()),
            "n_post_rows": int(len(sub)),
            "n_post_file": int((post_file["topic_id"] == t.topic_id).sum()),
            "n_pre_rows": int((pre["topic_id"] == t.topic_id).sum()),
            "mean_post": mean,
            "share_below": share,
            "hours_eligible": eligible,
            "status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows, columns=TOPIC_COLUMNS)


def topic_by_textbook(clean: pd.DataFrame, topics: pd.DataFrame, params: Params,
                      test_choice: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Тема × учебник: средний post, доля ниже порога, n. Нужен для тепловой карты."""
    infos = resolve_forms(clean, topics["topic_id"], test_choice)
    post = used_rows(clean, infos)
    post = post[post["stage"] == "post"]
    textbooks = sorted(clean["textbook"].unique())
    rows = []
    for t in topics.itertuples(index=False):
        needs = infos[t.topic_id].mode == MODE_NEEDS
        for tb in textbooks:
            sub = post[(post["topic_id"] == t.topic_id) & (post["textbook"] == tb)]
            per_student = _student_level(sub) if len(sub) else pd.DataFrame(columns=["pct"])
            n = int(len(per_student))
            rows.append({
                "topic_id": t.topic_id,
                "name": t.name,
                "textbook": tb,
                "n_students": n,
                "n_classes": int(sub["class_id"].nunique()),
                "mean_post": float(per_student["pct"].mean()) if n else float("nan"),
                "share_below": float(100.0 * (per_student["pct"] < params.low_score_cut).mean()) if n else float("nan"),
                "enough": bool(n >= params.min_post_topic),
                "needs_choice": needs,
            })
    return pd.DataFrame(rows, columns=["topic_id", "name", "textbook", "n_students", "n_classes",
                                       "mean_post", "share_below", "enough", "needs_choice"])


def data_summary(clean: pd.DataFrame, used: pd.DataFrame | None = None) -> dict:
    """Раздельные счётчики: ученики, классы, строки; а также «в файле» и «использовано в расчёте»."""
    used = clean if used is None else used
    return {
        "students": int(clean["uid"].nunique()),
        "classes": int(clean["class_id"].nunique()),
        "rows": int(len(clean)),
        "rows_pre": int((clean["stage"] == "pre").sum()),
        "rows_post": int((clean["stage"] == "post").sum()),
        "textbooks": int(clean["textbook"].nunique()),
        "rows_used": int(len(used)),
        "rows_post_used": int((used["stage"] == "post").sum()),
        "students_used": int(used["uid"].nunique()) if len(used) else 0,
    }
