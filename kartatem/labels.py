"""Подписи тем для графиков и таблиц.

Темы связываются только по topic_id. Название (name) может повторяться, поэтому при
совпадении названий к подписи добавляется id, и подписи на графиках остаются уникальными.
"""
from __future__ import annotations

import pandas as pd


def topic_labels(topics: pd.DataFrame) -> dict[str, str]:
    """{topic_id: подпись}. name уникален → name; повторяется → «name (topic_id)»."""
    counts = topics["name"].value_counts()
    out: dict[str, str] = {}
    for tid, name in zip(topics["topic_id"], topics["name"]):
        out[tid] = name if counts[name] == 1 else f"{name} ({tid})"
    return out
