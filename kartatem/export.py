"""Өлшем: экспорт отчёта в CSV (utf-8-sig, чтобы русский текст открывался в Excel)."""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from .config import Params
from .pipeline import Analysis

DISCLAIMER = ("Сценарий распределения, не прогноз: алгоритм не предсказывает рост баллов. "
              "Пороги — настраиваемые допущения демо, не нормативы.")


def source_note(kind: str, synthetic: bool, label: str) -> str:
    if synthetic:
        return f"СИНТЕТИЧЕСКИЕ данные ({label}). Не реальные результаты школ; не доказательство образовательного эффекта."
    return f"Загруженные пользователем файлы ({label}). Источник и качество измерения не проверялись."


def _safe_cell(v):
    """Защита от формул в Excel: текст, начинающийся с = + - @, получает префикс '."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@"):
        return "'" + v
    return v


def build_export(analysis: Analysis, params: Params, source_text: str,
                 applied: Mapping[str, int] | None, *, scenario: str | None = None,
                 locked=None, draft: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Таблица для скачивания. Невалидный черновик сюда не попадает:
    в столбце «утверждено» только применённый (прошедший проверку) план."""
    if analysis.blocked or analysis.plan is None:
        raise ValueError("Экспорт недоступен: анализ заблокирован ошибками в данных.")
    if analysis.unresolved:
        # Пока форма теста не выбрана хотя бы по одной теме, план не может считаться утверждённым.
        applied = None
    locked = list(analysis.locked if locked is None else locked)
    plan_obj = analysis.plan if locked == list(analysis.locked) else analysis.plan_for(locked)
    plan = plan_obj.table
    tt = analysis.topic_table.set_index("topic_id")
    cut = f"{params.low_score_cut:g}"
    if applied is not None:
        status = "Утверждён методистом в этой сессии"
    elif draft and {k: int(v) for k, v in draft.items()} != plan_obj.proposed():
        status = "Не утверждён: черновик изменён вручную"
    elif draft:
        status = "Не утверждён: черновик совпадает с предложением алгоритма"
    else:
        status = "Не утверждён"
    if analysis.unresolved:
        status += "; не выбрана форма теста"
    rows = []
    for r in plan.itertuples(index=False):
        t = tt.loc[r.topic_id]
        rows.append({
            "Продукт": "Өлшем",
            "Источник данных": source_text,
            "Тема (id)": r.topic_id,
            "Тема": r.name,
            "Часы исходные": int(r.hours_before),
            "Часы допустимо от": int(r.hours_low),
            "Часы допустимо до": int(r.hours_high),
            "Часы предложено алгоритмом": int(r.hours_proposed),
            "Часы утверждено": (int(applied[r.topic_id]) if applied is not None else ""),
            "Часы в неутверждённом черновике": (int(draft[r.topic_id]) if (applied is None and draft
                                                 and r.topic_id in draft) else ""),
            "Сценарий": scenario or "",
            "Не изменять (зафиксировано)": "да" if r.topic_id in locked else "",
            "Статус плана": status,
            "Форма теста": (t["selected_test"] or "не выбрана"),
            "Записей post в файле (валидных)": int(t["n_post_file"]),
            "Записей post использовано": int(t["n_post_rows"]),
            "Бюджет часов": int(plan_obj.budget),
            "Среднее post, %": round(float(t["mean_post"]), 1) if pd.notna(t["mean_post"]) else "",
            f"Доля ниже {cut}%, %": round(float(t["share_below"]), 1) if pd.notna(t["share_below"]) else "",
            "Учеников с post": int(t["n_students"]),
            "Классов": int(t["n_classes"]),
            "Строк post": int(t["n_post_rows"]),
            "Статус темы": t["status"],
            "Причина статуса": t["reason"],
            "Достаточно post для переноса часов": "да" if bool(t["hours_eligible"]) else "нет",
            "Примечание алгоритма": r.note,
            "Предупреждение": DISCLAIMER,
        })
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].map(_safe_cell)
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
