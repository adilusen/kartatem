"""Свойства демо проверяются по РЕАЛЬНО сгенерированным числам, а не по задуманным."""
from pathlib import Path

import pandas as pd
import pytest

from kartatem.compare import ST_OK, ST_INSUFFICIENT
from kartatem.config import Params
from kartatem.demo import (generate_results, topics_frame, bad_results_frame, results_template_csv,
                           topics_template_csv, TEXTBOOKS)
from kartatem.export import build_export, to_csv_bytes, source_note, _safe_cell
from kartatem.metrics import STATUS_LOW, STATUS_HARD
from kartatem.pipeline import analyze
from kartatem.validation import read_csv_bytes, validate_results, validate_topics
from helpers import results_df, topics_df

P = Params()
DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def full():
    return analyze(generate_results(), topics_frame(), P)


@pytest.fixture(scope="module")
def small():
    return analyze(generate_results(small=True), topics_frame(), P)


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate_results(), generate_results())
    pd.testing.assert_frame_equal(generate_results(small=True), generate_results(small=True))
    assert not generate_results(seed=1).equals(generate_results(seed=2))


def test_full_demo_size_and_structure(full):
    assert not full.blocked, full.block_reasons
    clean = full.results_val.clean
    per_tb = clean.groupby("textbook").agg(students=("uid", "nunique"), classes=("class_id", "nunique"))
    assert set(per_tb.index) == set(TEXTBOOKS)
    assert (per_tb["students"] >= 75).all() and (per_tb["classes"] >= 3).all()
    assert clean["topic_id"].nunique() == 6 and len(full.topics) == 6
    assert clean.groupby("topic_id")["test_id"].nunique().eq(1).all()        # одна форма теста на тему


def test_full_demo_scenario_holds_on_actual_numbers(full):
    c = full.comparison.table
    assert (c["status"] == ST_OK).all()
    assert (c["pre_b"] > c["pre_a"]).all(), "у Б должен быть выше входной результат"
    assert (c["post_b"] > c["post_a"]).all(), "у Б должен быть выше итоговый результат"
    assert (c["gain_a"] > c["gain_b"]).all(), "у А должен быть больше прирост"
    # демо содержит и предупреждения о разнице pre, и темы без них
    n_warn = c["warnings"].map(lambda w: any("Входные уровни" in x for x in w))
    assert n_warn.any() and not n_warn.all()


def test_full_demo_topics_and_hours(full):
    tt = full.topic_table.set_index("topic_id")
    assert (tt["status"] == STATUS_HARD).any()
    assert not (tt["status"] == STATUS_LOW).any()
    assert full.plan.moves, "в демо должен быть хотя бы один перенос часов"
    assert full.plan.table["hours_proposed"].sum() == full.plan.budget == 30


def test_demo_has_explicit_missing_values_reported(full):
    rv = full.results_val
    assert rv.n_excluded_rows > 0 and rv.n_valid_rows + rv.n_excluded_rows == rv.n_input_rows


def test_small_demo_enough_for_topics_not_for_comparison(small):
    assert not small.blocked
    c = small.comparison.table
    assert (c["status"] == ST_INSUFFICIENT).all()
    assert c[["gain_a", "gain_b", "gain_diff"]].isna().all().all()
    assert c["post_a"].notna().all() and (c["n_post_a"] > 0).all()           # post и n остаются
    tt = small.topic_table.set_index("topic_id")
    assert (tt.loc[["T1", "T2", "T3", "T4", "T5"], "n_students"] >= 15).all()
    assert tt.loc["T6", "status"] == STATUS_LOW and tt.loc["T6", "n_students"] < 15
    assert not tt.loc["T6", "hours_eligible"]
    assert small.plan.proposed()["T6"] == 5                                   # тема без достаточного post не меняется


def test_bad_example_is_blocked_with_clear_errors():
    a = analyze(bad_results_frame(), topics_frame(), P)
    assert a.blocked and a.results_val.clean.empty
    text = " ".join(a.results_val.errors)
    for needle in ("student_id", "stage", "T9", "Конфликтующие дубли"):
        assert needle in text
    assert a.results_val.exclusion_counts          # исключаемые записи тоже перечислены


def test_empty_and_invalid_inputs_never_crash():
    for raw in (None, pd.DataFrame(), results_df([])):
        a = analyze(raw, topics_frame(), P)
        assert a.blocked
    a = analyze(generate_results(), topics_df([("T1", "A", 2, 3, 2)]), P)       # плохой topics.csv
    assert a.blocked and any("topics" in r for r in a.block_reasons)
    a = analyze(generate_results(), topics_frame(), Params(target=80, donor_from=70))
    assert a.blocked and a.param_errors


def test_committed_csv_files_are_valid_and_match_properties():
    topics = validate_topics(read_csv_bytes((DATA / "demo_topics.csv").read_bytes())[0])
    assert topics.ok
    for name, expect_comparison in (("demo_results.csv", True), ("demo_results_small.csv", False)):
        df, err = read_csv_bytes((DATA / name).read_bytes())
        assert err is None
        a = analyze(df, topics.clean, P)
        assert not a.blocked
        assert (a.comparison.table["status"] == ST_OK).all() == expect_comparison
    df, _ = read_csv_bytes((DATA / "bad_results_example.csv").read_bytes())
    assert analyze(df, topics.clean, P).blocked
    for tpl, validator in (("template_results.csv", lambda d: validate_results(d, None)),
                           ("template_topics.csv", validate_topics)):
        d, err = read_csv_bytes((DATA / tpl).read_bytes())
        assert err is None and validator(d).ok


def test_templates_match_generators():
    assert (DATA / "template_results.csv").read_text(encoding="utf-8") == results_template_csv()
    assert (DATA / "template_topics.csv").read_text(encoding="utf-8") == topics_template_csv()


# ---------- экспорт ----------
def test_export_contains_required_fields_and_encoding(full):
    src = source_note("demo", True, "seed=42")
    df = build_export(full, P, src, None)
    cols = " ".join(df.columns)
    for needle in ("исходные", "предложено", "утверждено", "Среднее post", "Учеников с post", "Источник данных"):
        assert needle in cols
    assert df["Часы исходные"].sum() == df["Часы предложено алгоритмом"].sum() == 30
    assert (df["Часы утверждено"] == "").all()                                 # не утверждено — пусто
    assert df["Статус плана"].str.startswith("Не утверждён").all()
    assert df["Предупреждение"].str.contains("не прогноз").all()
    assert df["Источник данных"].str.contains("СИНТЕТИЧЕСКИЕ").all()
    raw = to_csv_bytes(df)
    assert raw.startswith(b"\xef\xbb\xbf")                                     # BOM → utf-8-sig
    back = pd.read_csv(pd.io.common.BytesIO(raw), encoding="utf-8-sig")
    assert "Обыкновенные дроби" in set(back["Тема"])


def test_export_applied_plan_and_blocked_analysis(full):
    applied = full.plan.proposed()
    df = build_export(full, P, "x", applied)
    assert list(df["Часы утверждено"]) == [applied[t] for t in df["Тема (id)"]]
    assert df["Статус плана"].str.startswith("Утверждён").all()
    blocked = analyze(bad_results_frame(), topics_frame(), P)
    with pytest.raises(ValueError):
        build_export(blocked, P, "x", None)


def test_export_neutralizes_spreadsheet_formulas():
    assert _safe_cell("=SUM(A1)") == "'=SUM(A1)" and _safe_cell("@x") == "'@x" and _safe_cell("Дроби") == "Дроби"
    assert source_note("upload", False, "a.csv").startswith("Загруженные пользователем файлы")
    assert "не проверялись" in source_note("upload", False, "a.csv")
