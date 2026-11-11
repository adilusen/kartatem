"""Текущее решение и три его экспорта (без Streamlit).

Карточка «Решение», паспорт HTML, план JSON и отчёт CSV строятся из ОДНОГО варианта:
утверждённого плана, если он есть, иначе — текущего черновика (как неутверждённого).
"""
from __future__ import annotations

from dataclasses import dataclass

from .export import build_export, source_note, to_csv_bytes
from .hours import validate_plan
from .passport import build_passport
from .plan_io import plan_to_json

DRAFT_NAME = "Черновик"


@dataclass
class Decision:
    name: str
    hours: dict
    locked: list
    approved: bool
    errors: list
    html: str
    plan_json: bytes | None      # None — показанный черновик невосстановим, файл не выдаётся
    csv: bytes


def current_decision(analysis, *, applied, applied_name, applied_locked, draft, locked):
    """(название, часы, фиксации, утверждён?)."""
    if applied is not None and not analysis.unresolved:
        return applied_name or DRAFT_NAME, dict(applied), list(applied_locked or []), True
    return DRAFT_NAME, dict(draft or {}), list(locked or []), False


def build_decision(analysis, params, *, source: dict, context: str, context_parts: dict, test_choice: dict,
                   scenarios: list, applied, applied_name, applied_locked, draft, locked) -> Decision:
    name, hours, lk, approved = current_decision(analysis, applied=applied, applied_name=applied_name,
                                                 applied_locked=applied_locked, draft=draft, locked=locked)
    errors = validate_plan(analysis.topics, hours, lk)
    src_text = source_note(source["kind"], source["synthetic"], source["label"])
    html = build_passport(analysis, params, source_text=src_text, synthetic=source["synthetic"],
                          scenario_name=name, hours=hours, locked=lk, approved=approved, context=context)
    plan_json = None
    if not errors:
        plan_json = plan_to_json(context=context, context_parts=context_parts, params=params,
                                 test_choice=test_choice, locked=lk, draft=hours, scenarios=scenarios,
                                 approved=({"name": name, "hours": hours, "locked": lk} if approved else None),
                                 topics=analysis.topics)
    csv = to_csv_bytes(build_export(analysis, params, src_text, hours if approved else None,
                                    scenario=name, locked=lk, draft=None if approved else hours))
    return Decision(name, hours, lk, approved, errors, html, plan_json, csv)
