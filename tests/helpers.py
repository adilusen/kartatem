"""Маленькие фабрики данных для тестов (сырые таблицы в виде строк, как после чтения CSV)."""
from __future__ import annotations

import pandas as pd

from kartatem.validation import RESULTS_COLUMNS, TOPICS_COLUMNS


def results_df(rows: list[tuple]) -> pd.DataFrame:
    """rows: (student_id, class_id, textbook, topic_id, stage, test_id, score, max_score)."""
    return pd.DataFrame([[str(x) for x in r] for r in rows], columns=RESULTS_COLUMNS)


def topics_df(rows: list[tuple] | None = None) -> pd.DataFrame:
    rows = rows or [("T1", "Тема 1", 5, 3, 2)]
    return pd.DataFrame([[str(x) for x in r] for r in rows], columns=TOPICS_COLUMNS)


def make_group(textbook: str, class_ids: list[str], n_per_class: int, topic: str = "T1",
               test: str = "F1", pre_pct: float = 40, post_pct: float = 60, max_score: int = 100,
               prefix: str = "S") -> list[tuple]:
    """Группа учеников с заданными pre/post (в процентах) — удобно для проверки формул."""
    rows = []
    for c in class_ids:
        for i in range(1, n_per_class + 1):
            sid = f"{prefix}{i:02d}"
            rows.append((sid, c, textbook, topic, "pre", test, pre_pct * max_score / 100, max_score))
            rows.append((sid, c, textbook, topic, "post", test, post_pct * max_score / 100, max_score))
    return rows
