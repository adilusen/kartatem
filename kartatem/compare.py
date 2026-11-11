"""Описательное сравнение учебников по темам.

Что это и чем НЕ является:
* «Прирост» = post − pre у одного и того же ученика, в п.п. на общей процентной шкале.
  Это ОПИСАТЕЛЬНОЕ изменение результата, а не статистическая корректировка исходного уровня.
* Группы не сформированы случайно: на различия влияют учитель, школа, состав класса.
  Поэтому здесь нет доверительных интервалов, регрессии, «победителя» и общего рейтинга.
* Единый идентификатор теста (test_id) — проверка формата, а не доказательство качества измерения.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .config import Params
from .forms import MODE_NEEDS, MODE_NONE, resolve_forms

ST_OK = "ok"
ST_INSUFFICIENT = "insufficient"
ST_NEEDS_CHOICE = "needs_test_choice"
ST_NO_COMMON = "no_common_test"

STATUS_LABELS = {
    ST_OK: "Прирост сравнивается (описательно)",
    ST_INSUFFICIENT: "Мало данных для сравнения: только post",
    ST_NEEDS_CHOICE: "Выберите форму теста",
    ST_NO_COMMON: "Выбранная форма не представлена у обоих учебников: сравнение не проводится",
}

COMPARISON_COLUMNS = [
    "topic_id", "topic_name", "textbook_a", "textbook_b", "status", "test_id", "candidates",
    "n_post_a", "n_post_b", "post_a", "post_b",
    "n_pairs_a", "n_pairs_b", "pair_share_a", "pair_share_b", "classes_a", "classes_b",
    "pre_a", "pre_b", "post_pair_a", "post_pair_b", "gain_a", "gain_b",
    "gain_diff", "pre_gap", "direction", "warnings",
]


@dataclass
class ComparisonResult:
    table: pd.DataFrame
    messages: list[str] = field(default_factory=list)

    def selection_needed(self) -> dict[str, list[str]]:
        """{topic_id: [варианты test_id]} для тем, где нужно выбрать форму теста."""
        out: dict[str, list[str]] = {}
        t = self.table[self.table["status"] == ST_NEEDS_CHOICE]
        for r in t.itertuples(index=False):
            out[r.topic_id] = list(r.candidates)
        return out


def _nan() -> float:
    return float("nan")


def _empty_row(tid: str, name: str, a: str, b: str, status: str) -> dict:
    row = {c: _nan() for c in COMPARISON_COLUMNS}
    row.update({"topic_id": tid, "topic_name": name, "textbook_a": a, "textbook_b": b,
                "status": status, "test_id": "", "candidates": [], "direction": "", "warnings": []})
    return row


def fmt_gap_vs_threshold(gap: float, threshold: float) -> tuple[str, str]:
    """Строки (разница, порог) так, чтобы срабатывание порога было видно.

    Если после округления до 0,1 они совпали (разница 10.03 при пороге 10), добавляем знаки,
    пока строки не станут различаться. Сама арифметика сравнения не меняется.
    """
    v = abs(gap)
    if f"{v:.1f}" != f"{threshold:.1f}":
        return f"{v:.1f}", f"{threshold:g}"
    for d in (2, 3, 4):
        sv, st_ = f"{v:.{d}f}", f"{threshold:.{d}f}"
        if sv != st_:
            return sv, st_
    return f"{v:.4f}", f"{threshold:.4f}"


def direction_text(a: str, b: str, diff: float) -> str:
    """Формулировка без слов «лучше/эффективнее»: только описательное различие."""
    r = round(diff, 1)
    if r == 0:
        return "Средние приросты совпадают при округлении до 0,1 п.п."
    bigger = a if diff > 0 else b
    return (f"Средний прирост больше у «{bigger}» на {abs(r):.1f} п.п. "
            "(описательное различие, не доказательство эффекта учебника).")


def _compare_pair(tdf: pd.DataFrame, tid: str, name: str, a: str, b: str,
                  params: Params, test: str, changed_uids: set) -> dict:
    """tdf — записи темы ТОЛЬКО по выбранной форме test (выбор сделан в forms.py)."""
    post_rows = tdf[tdf["stage"] == "post"]
    missing = [x for x in (a, b) if not (post_rows["textbook"] == x).any()]
    if missing:
        row = _empty_row(tid, name, a, b, ST_NO_COMMON)
        row["test_id"] = test
        row["n_post_a"] = int((post_rows["textbook"] == a).sum())
        row["n_post_b"] = int((post_rows["textbook"] == b).sum())
        row["warnings"] = [
            f"Выбранная форма теста «{test}» не представлена у учебника "
            + ", ".join(f"«{m}»" for m in missing)
            + " (нет итоговых результатов post). Сравнение не проводится; другая форма в сравнение не подмешивается."]
        row["direction"] = "Сравнение не проводится."
        return row

    sub = tdf
    pre = (sub[sub["stage"] == "pre"][["uid", "class_id", "textbook", "pct"]]
           .rename(columns={"textbook": "tb_pre", "pct": "pre"}))
    post = (sub[sub["stage"] == "post"][["uid", "class_id", "textbook", "pct"]]
            .rename(columns={"textbook": "tb_post", "pct": "post"}))
    pairs = pre.merge(post, on=["uid", "class_id"], how="inner")
    pairs = pairs[(pairs["tb_pre"] == pairs["tb_post"]) & ~pairs["uid"].isin(changed_uids)].copy()
    pairs["textbook"] = pairs["tb_post"]
    pairs["gain"] = pairs["post"] - pairs["pre"]

    def stats(tb: str) -> dict:
        p_all = post[post["tb_post"] == tb]
        p_pairs = pairs[pairs["textbook"] == tb]
        n_post = int(len(p_all))
        n_pairs = int(len(p_pairs))
        return {
            "n_post": n_post,
            "post": float(p_all["post"].mean()) if n_post else _nan(),
            "n_pairs": n_pairs,
            "pair_share": (n_pairs / n_post) if n_post else _nan(),
            "classes": int(p_pairs["class_id"].nunique()),
            "pre": float(p_pairs["pre"].mean()) if n_pairs else _nan(),
            "post_pair": float(p_pairs["post"].mean()) if n_pairs else _nan(),
            "gain": float(p_pairs["gain"].mean()) if n_pairs else _nan(),
            "changed": int(p_all["uid"].isin(changed_uids).sum()),
        }

    sa, sb = stats(a), stats(b)
    row = _empty_row(tid, name, a, b, ST_INSUFFICIENT)
    row["test_id"] = test
    row.update({
        "n_post_a": sa["n_post"], "n_post_b": sb["n_post"], "post_a": sa["post"], "post_b": sb["post"],
        "n_pairs_a": sa["n_pairs"], "n_pairs_b": sb["n_pairs"],
        "pair_share_a": sa["pair_share"], "pair_share_b": sb["pair_share"],
        "classes_a": sa["classes"], "classes_b": sb["classes"],
    })
    warns: list[str] = []
    if sa["changed"] or sb["changed"]:
        warns.append(f"Ученики со сменой учебника не вошли в пары: {sa['changed'] + sb['changed']}.")

    problems = []
    for tb, s in ((a, sa), (b, sb)):
        if s["n_pairs"] < params.min_pairs_compare:
            problems.append(f"«{tb}»: пар pre/post {s['n_pairs']} < {params.min_pairs_compare}")
        if s["classes"] < params.min_classes_compare:
            problems.append(f"«{tb}»: классов среди пар {s['classes']} < {params.min_classes_compare}")

    if problems:
        warns.append("Данных недостаточно для сравнения приростов: " + "; ".join(problems)
                     + ". Показан только post (с n), без прироста и без вывода о направлении.")
        row["direction"] = "Сравнение приростов заблокировано (мало данных)."
        row["warnings"] = warns
        return row

    row["status"] = ST_OK
    row.update({
        "pre_a": sa["pre"], "pre_b": sb["pre"],
        "post_pair_a": sa["post_pair"], "post_pair_b": sb["post_pair"],
        "gain_a": sa["gain"], "gain_b": sb["gain"],
    })
    row["gain_diff"] = sa["gain"] - sb["gain"]
    row["pre_gap"] = sa["pre"] - sb["pre"]
    row["direction"] = direction_text(a, b, row["gain_diff"])
    if abs(row["pre_gap"]) > params.pre_gap_warn_pp:
        gap_s, thr_s = fmt_gap_vs_threshold(row["pre_gap"], params.pre_gap_warn_pp)
        warns.append(
            f"Входные уровни пар различаются на {gap_s} п.п. — больше порога предупреждения "
            f"{thr_s} п.п. (настраиваемый). "
            "Различие приростов может отражать разный стартовый уровень; корректировка "
            "исходного уровня в этой версии не выполняется.")
    row["warnings"] = warns
    return row


def compare_textbooks(clean: pd.DataFrame, topics: pd.DataFrame, params: Params,
                      test_choice: dict[str, str] | None = None,
                      changed_uids: set | None = None) -> ComparisonResult:
    changed_uids = changed_uids or set()
    infos = resolve_forms(clean, topics["topic_id"], test_choice)
    textbooks = sorted(clean["textbook"].unique())
    if len(textbooks) < 2:
        return ComparisonResult(
            pd.DataFrame(columns=COMPARISON_COLUMNS),
            ["Для сравнения нужно минимум два учебника, в данных их: " + str(len(textbooks)) + "."])
    rows = []
    for t in topics.itertuples(index=False):
        info = infos[t.topic_id]
        for a, b in combinations(textbooks, 2):
            if info.mode == MODE_NEEDS:
                row = _empty_row(t.topic_id, t.name, a, b, ST_NEEDS_CHOICE)
                row["candidates"] = list(info.forms)
                row["warnings"] = ["В теме несколько форм теста. Выберите одну в боковой панели: формы не смешиваются."]
                row["direction"] = "Сравнение не проводится, пока не выбрана форма теста."
            elif info.mode == MODE_NONE:
                row = _empty_row(t.topic_id, t.name, a, b, ST_NO_COMMON)
                row["warnings"] = ["По этой теме нет ни одной валидной записи."]
                row["direction"] = "Сравнение не проводится."
            else:
                tdf = clean[(clean["topic_id"] == t.topic_id) & (clean["test_id"] == info.selected)]
                row = _compare_pair(tdf, t.topic_id, t.name, a, b, params, info.selected, changed_uids)
            rows.append(row)
    return ComparisonResult(pd.DataFrame(rows, columns=COMPARISON_COLUMNS), [])
