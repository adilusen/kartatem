"""Единая точка входа для расчётов: сырые таблицы → проверки → метрики → сравнение → часы.

Здесь нет Streamlit, поэтому всё тестируется обычным pytest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .compare import ComparisonResult, compare_textbooks, ST_OK, STATUS_LABELS
from .config import Params, validate_params
from .forms import FormInfo, MODE_NEEDS, form_key, resolve_forms, used_rows
from .hours import HoursPlan, propose_hours
from .labels import topic_labels
from .metrics import STATUS_CHOOSE, data_summary, topic_by_textbook, topic_metrics
from .validation import ResultsValidation, TopicsValidation, validate_results, validate_topics


@dataclass
class Analysis:
    topics_val: TopicsValidation
    results_val: ResultsValidation
    param_errors: list[str]
    block_reasons: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    topic_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    topic_tb: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison: ComparisonResult | None = None
    plan: HoursPlan | None = None
    sufficiency: pd.DataFrame = field(default_factory=pd.DataFrame)
    forms: dict = field(default_factory=dict)           # {topic_id: FormInfo}
    used: pd.DataFrame = field(default_factory=pd.DataFrame)   # записи, реально вошедшие в расчёт
    unresolved: list = field(default_factory=list)      # темы, где нужно выбрать форму теста
    form_key: tuple = ()
    textbooks: list = field(default_factory=list)   # все учебники набора (для постоянных цветов)
    params_used: Params | None = None
    locked: list = field(default_factory=list)      # темы с фиксированными исходными часами
    mastery: dict = field(default_factory=dict)
    eligible: dict = field(default_factory=dict)
    skip: dict = field(default_factory=dict)

    def plan_for(self, locked) -> HoursPlan:
        """Предложение алгоритма на ЭТИХ ЖЕ данных при другом наборе фиксаций."""
        return propose_hours(self.topics, self.mastery, self.eligible, self.params_used, self.skip, locked)

    @property
    def blocked(self) -> bool:
        return bool(self.block_reasons)

    @property
    def topics(self) -> pd.DataFrame:
        return self.topics_val.clean


def _sufficiency_table(topic_table: pd.DataFrame, comparison: ComparisonResult,
                       params: Params, labels: dict) -> pd.DataFrame:
    """Три РАЗНЫХ проверки по каждой теме: оценка темы, перенос часов, сравнение учебников."""
    rows = []
    for t in topic_table.itertuples(index=False):
        if t.status == STATUS_CHOOSE:
            rows.append({
                "Тема": labels[t.topic_id],
                "Учеников с post": 0,
                f"Оценка темы (≥ {params.min_post_topic})": STATUS_CHOOSE.lower(),
                f"Перенос часов (≥ {params.min_post_hours})": "не допускается: выберите форму теста",
                f"Сравнение учебников (≥ {params.min_pairs_compare} пар, ≥ {params.min_classes_compare} кл.)":
                    STATUS_CHOOSE.lower(),
            })
            continue
        cmp_rows = comparison.table[comparison.table["topic_id"] == t.topic_id]
        if cmp_rows.empty:
            cmp_text = "нет пары учебников"
        elif (cmp_rows["status"] == ST_OK).all():
            cmp_text = "достаточно"
        else:
            first = cmp_rows[cmp_rows["status"] != ST_OK].iloc[0]
            cmp_text = STATUS_LABELS[first["status"]]
            if first["status"] == "insufficient":
                cmp_text += f" (пар: {first['n_pairs_a']}/{first['n_pairs_b']}; классов: {first['classes_a']}/{first['classes_b']})"
            if len(cmp_rows) > 1:
                cmp_text += f" — пара «{first['textbook_a']}» и «{first['textbook_b']}»"
        rows.append({
            "Тема": labels[t.topic_id],
            "Учеников с post": t.n_students,
            f"Оценка темы (≥ {params.min_post_topic})": "достаточно" if t.n_students >= params.min_post_topic else "мало данных",
            f"Перенос часов (≥ {params.min_post_hours})": "допускается" if t.hours_eligible else "не допускается",
            f"Сравнение учебников (≥ {params.min_pairs_compare} пар, ≥ {params.min_classes_compare} кл.)": cmp_text,
        })
    return pd.DataFrame(rows)


def analyze(results_raw: pd.DataFrame | None, topics_raw: pd.DataFrame | None,
            params: Params, test_choice: dict[str, str] | None = None,
            locked=None) -> Analysis:
    topics_val = validate_topics(topics_raw)
    known = set(topics_val.clean["topic_id"]) if topics_val.ok else None
    results_val = validate_results(results_raw, known)
    a = Analysis(topics_val=topics_val, results_val=results_val,
                 param_errors=validate_params(params))
    a.params_used = params

    if not topics_val.ok:
        a.block_reasons.append("Ошибки в topics.csv — анализ заблокирован.")
    if not results_val.ok:
        a.block_reasons.append("Ошибки в results.csv — анализ заблокирован.")
    if a.param_errors:
        a.block_reasons.append("Некорректные параметры — исправьте их в боковой панели.")
    if a.blocked:
        return a

    clean, topics = results_val.clean, topics_val.clean
    # Единый выбор формы теста: ОДНИ И ТЕ ЖЕ записи идут в метрики, карту, сравнение, часы и экспорт.
    a.textbooks = sorted(clean["textbook"].unique())
    a.forms = resolve_forms(clean, topics["topic_id"], test_choice)
    a.used = used_rows(clean, a.forms)
    a.form_key = form_key(a.forms)
    a.unresolved = [t for t, i in a.forms.items() if i.mode == MODE_NEEDS]
    a.summary = data_summary(clean, a.used)
    a.topic_table = topic_metrics(clean, topics, params, test_choice)
    a.topic_tb = topic_by_textbook(clean, topics, params, test_choice)
    a.comparison = compare_textbooks(clean, topics, params, test_choice,
                                     results_val.changed_textbook_uids)
    a.sufficiency = _sufficiency_table(a.topic_table, a.comparison, params, topic_labels(topics))
    mastery = dict(zip(a.topic_table["topic_id"], a.topic_table["mean_post"]))
    eligible = dict(zip(a.topic_table["topic_id"], a.topic_table["hours_eligible"]))
    skip = {t: "не участвует: выберите форму теста" for t in a.unresolved}
    a.locked = sorted(set(locked or ()) & set(topics["topic_id"]))
    a.mastery, a.eligible, a.skip = mastery, eligible, skip
    a.plan = propose_hours(topics, mastery, eligible, params, skip, a.locked)
    return a
