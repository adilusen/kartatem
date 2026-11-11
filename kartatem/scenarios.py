"""Сценарии плана часов: имя, часы, фиксации и контекст данных (без Streamlit).

Контекст = набор данных (подпись файлов) + параметры + выбранные формы теста.
Сценарий, созданный в другом контексте, помечается и не применяется без пересчёта:
он опирался на другие измерения. Фиксации в контекст НЕ входят — именно ими
сценарии на одних данных и отличаются друг от друга.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from .config import Params
from .hours import validate_plan

MAX_SCENARIOS = 3
MAX_NAME = 60


def context_parts(source_sig: str, params: Params, form_key: tuple) -> dict:
    return {"source": str(source_sig), "params": asdict(params),
            "forms": {t: f for t, f in form_key}}


def context_id(parts: dict) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def clean_name(name: str) -> str:
    return " ".join(str(name).split())[:MAX_NAME]


@dataclass
class Scenario:
    name: str
    hours: dict
    locked: list
    context: str
    origin: str = "manual"          # manual | algorithm | restored

    def to_dict(self) -> dict:
        return {"name": self.name, "hours": {k: int(v) for k, v in self.hours.items()},
                "locked": sorted(self.locked), "context": self.context, "origin": self.origin}


def scenario_problems(sc: Scenario, topics: pd.DataFrame, ctx: str) -> list[str]:
    """Почему сценарий нельзя утвердить в текущем контексте (пусто — можно)."""
    errs = []
    if sc.context != ctx:
        errs.append("Сценарий создан на других данных, параметрах или формах теста: пересчитайте его.")
    errs += validate_plan(topics, sc.hours, sc.locked)
    return errs


def add_scenario(scenarios: list[Scenario], sc: Scenario) -> tuple[list[Scenario], str | None]:
    """Добавляет сценарий (или заменяет одноимённый). Возвращает (новый список, ошибка)."""
    name = clean_name(sc.name)
    if not name:
        return scenarios, "Введите название сценария."
    sc = Scenario(name, dict(sc.hours), sorted(sc.locked), sc.context, sc.origin)
    out = [x for x in scenarios if x.name != name]
    if len(out) >= MAX_SCENARIOS:
        return scenarios, f"Можно сохранить не больше {MAX_SCENARIOS} сценариев. Удалите один из сохранённых."
    pos = next((i for i, x in enumerate(scenarios) if x.name == name), len(out))
    out.insert(pos, sc)
    return out, None


def compare_scenarios(topics: pd.DataFrame, a: Scenario, b: Scenario, labels: dict) -> pd.DataFrame:
    """Тема, исходно, А, Б, разница (Б − А) и фиксации. Без прогноза баллов."""
    rows = []
    for r in topics.itertuples(index=False):
        ha, hb = int(a.hours[r.topic_id]), int(b.hours[r.topic_id])
        rows.append({
            "Тема": labels[r.topic_id], "Исходно, ч": int(r.hours),
            f"«{a.name}», ч": ha, f"«{b.name}», ч": hb, "Разница Б − А, ч": hb - ha,
            "Фиксация": "/".join(x for x, s in (("А", a), ("Б", b)) if r.topic_id in s.locked) or "—",
        })
    return pd.DataFrame(rows)
