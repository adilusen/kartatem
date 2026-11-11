"""Регрессионные проверки шести ошибок, найденных независимым ревью (ядро, без интерфейса)."""
import math

import pandas as pd
import pytest

from kartatem.compare import ST_NEEDS_CHOICE, ST_NO_COMMON, ST_INSUFFICIENT
from kartatem.config import Params
from kartatem.demo import generate_results, topics_frame
from kartatem.export import build_export
from kartatem.forms import MODE_AUTO, MODE_CHOSEN, MODE_NEEDS, resolve_forms, used_rows
from kartatem.labels import topic_labels
from kartatem.metrics import (STATUS_CHOOSE, STATUS_HARD, STATUS_LOW, data_summary, topic_by_textbook,
                              topic_metrics)
from kartatem.pipeline import analyze
from kartatem.validation import (make_uid, read_csv_bytes, validate_results, validate_topics)
from helpers import results_df, topics_df, make_group

P = Params()
H = "student_id,class_id,textbook,topic_id,stage,test_id,score,max_score"


# ============================================================================
# 1. Единый выбор формы теста
# ============================================================================
def repro_rows():
    """T1: 20 учеников с post — 10 по easy (95 %), 10 по hard (65 %). T2: у всех post 40 %."""
    rows = []
    for i in range(1, 21):
        sid = f"S{i:02d}"
        form, pct = ("easy", 95) if i <= 10 else ("hard", 65)
        rows.append((sid, "K01", "А", "T1", "post", form, pct, 100))
        rows.append((sid, "K01", "А", "T2", "post", "F", 40, 100))
    return rows


TOPICS2 = topics_df([("T1", "Тема 1", 5, 3, 2), ("T2", "Тема 2", 5, 3, 2)])


def run2(choice=None, rows=None):
    return analyze(results_df(rows or repro_rows()), TOPICS2, P, choice)


def test_unresolved_multiform_topic_gets_no_mean_no_flag_and_hours_stay():
    a = run2()
    assert not a.blocked and a.unresolved == ["T1"]
    t1 = a.topic_table.set_index("topic_id").loc["T1"]
    assert t1.status == STATUS_CHOOSE and math.isnan(t1.mean_post) and math.isnan(t1.share_below)
    assert t1.n_students == 0 and not t1.hours_eligible
    assert "объединённое среднее не считается" in t1.reason
    assert a.plan.proposed() == {"T1": 5, "T2": 5}                      # раньше было 3/7
    assert "выберите форму теста" in a.plan.table.set_index("topic_id").loc["T1", "note"]


def test_unresolved_topic_does_not_block_other_topics():
    t2 = run2().topic_table.set_index("topic_id").loc["T2"]
    assert t2.status == STATUS_HARD and t2.n_students == 20 and math.isclose(t2.mean_post, 40)
    assert t2.selected_test == "F" and t2.form_mode == MODE_AUTO


@pytest.mark.parametrize("form", ["easy", "hard"])
def test_chosen_form_with_too_few_students_gets_no_automatic_transfer(form):
    a = run2({"T1": form})
    t1 = a.topic_table.set_index("topic_id").loc["T1"]
    assert t1.form_mode == MODE_CHOSEN and t1.n_students == 10 and t1.status == STATUS_LOW
    assert not t1.hours_eligible and not a.plan.moves
    assert a.plan.proposed() == {"T1": 5, "T2": 5}
    assert math.isclose(t1.mean_post, 95 if form == "easy" else 65)    # только выбранная форма, без смеси


def switching_rows():
    """T1: 16 по easy (95 %) и 16 по hard (30 %); T2: 20 учеников по 80 %."""
    rows = []
    for i in range(1, 33):
        form, pct = ("easy", 95) if i <= 16 else ("hard", 30)
        rows.append((f"S{i:02d}", "K01", "А", "T1", "post", form, pct, 100))
    rows += [(f"S{i:02d}", "K01", "А", "T2", "post", "F", 80, 100) for i in range(1, 21)]
    return rows


def test_changing_form_changes_metrics_and_hours():
    easy = analyze(results_df(switching_rows()), TOPICS2, P, {"T1": "easy"})
    hard = analyze(results_df(switching_rows()), TOPICS2, P, {"T1": "hard"})
    assert easy.plan.proposed() == {"T1": 5, "T2": 5}               # обе темы ≥ 75 %: получателей нет
    assert hard.plan.proposed() == {"T1": 7, "T2": 3}               # T1 (30 %) получает часы у T2 (80 %)
    assert math.isclose(easy.topic_table.set_index("topic_id").loc["T1", "mean_post"], 95)
    assert math.isclose(hard.topic_table.set_index("topic_id").loc["T1", "mean_post"], 30)
    assert hard.topic_table.set_index("topic_id").loc["T1", "status"] == STATUS_HARD
    assert hard.form_key != easy.form_key                           # сигнатура состояния меняется
    assert [m.to_topic for m in hard.plan.moves] == ["T1", "T1"]


def test_single_form_is_selected_automatically_and_invalid_choice_is_ignored():
    infos = resolve_forms(validate_results(results_df(repro_rows()), {"T1", "T2"}).clean, ["T1", "T2"],
                          {"T1": "нет-такой", "T2": "F"})
    assert infos["T2"].mode == MODE_AUTO and infos["T2"].selected == "F"
    assert infos["T1"].mode == MODE_NEEDS and infos["T1"].forms == ("easy", "hard")


def test_same_rows_feed_heatmap_comparison_and_metrics():
    rows = (make_group("А", ["K01", "K02"], 20, test="F1", pre_pct=40, post_pct=60, prefix="S")
            + make_group("Б", ["K03", "K04"], 20, test="F1", pre_pct=50, post_pct=62, prefix="S")
            + make_group("А", ["K01", "K02"], 20, test="F2", pre_pct=70, post_pct=90, prefix="X")
            + make_group("Б", ["K03", "K04"], 20, test="F2", pre_pct=70, post_pct=90, prefix="X"))
    topics = topics_df()
    a_none = analyze(results_df(rows), topics, P, {})
    assert a_none.comparison.table.iloc[0].status == ST_NEEDS_CHOICE
    assert a_none.topic_tb["needs_choice"].all() and a_none.topic_tb["mean_post"].isna().all()
    a = analyze(results_df(rows), topics, P, {"T1": "F1"})
    tt, tb, c = a.topic_table.iloc[0], a.topic_tb, a.comparison.table.iloc[0]
    assert tt.n_students == 80 and math.isclose(tt.mean_post, 61)
    assert tb["n_students"].sum() == tt.n_students                  # карта и таблица тем — одни и те же записи
    assert c.n_post_a + c.n_post_b == tt.n_students and c.test_id == "F1"
    assert len(a.used) == 160 and a.summary["rows_used"] == 160     # F2 в расчёт не попала


def test_file_volume_differs_from_used_volume():
    a = run2({"T1": "easy"})
    t1 = a.topic_table.set_index("topic_id").loc["T1"]
    assert t1.n_post_file == 20 and t1.n_post_rows == 10
    assert a.summary["rows"] == 40 and a.summary["rows_used"] == 30 and a.summary["students_used"] == 20
    unresolved = run2()
    assert unresolved.summary["rows_used"] == 20                    # только T2


def test_post_only_topic_stays_available_without_gain():
    rows = (make_group("А", ["K01", "K02"], 20, post_pct=60)
            + make_group("Б", ["K03", "K04"], 20, post_pct=70))
    rows = [r for r in rows if r[4] == "post"]
    a = analyze(results_df(rows), topics_df(), P)
    assert not a.blocked
    assert a.topic_table.iloc[0].n_students == 80 and math.isclose(a.topic_table.iloc[0].mean_post, 65)
    c = a.comparison.table.iloc[0]
    assert c.status == ST_INSUFFICIENT and c.n_pairs_a == 0 and math.isnan(c.gain_a) and c.post_a == 60


def test_chosen_form_missing_in_one_textbook_is_reported():
    rows = (make_group("А", ["K01", "K02"], 20, test="F1") + make_group("Б", ["K03", "K04"], 20, test="F1")
            + make_group("А", ["K01", "K02"], 20, test="F2", prefix="X"))      # F2 есть только у А
    c = analyze(results_df(rows), topics_df(), P, {"T1": "F2"}).comparison.table.iloc[0]
    assert c.status == ST_NO_COMMON and any("«Б»" in w and "не представлена" in w for w in c.warnings)


def test_export_follows_form_choice_and_never_approves_unresolved():
    applied = {"T1": 5, "T2": 5}
    unresolved = run2()
    df = build_export(unresolved, P, "src", applied)               # передан «утверждённый» план
    assert (df["Часы утверждено"] == "").all()                      # но при нерешённой форме он не утверждён
    assert df["Статус плана"].str.contains("не выбрана форма теста").all()
    assert df.set_index("Тема (id)").loc["T1", "Форма теста"] == "не выбрана"
    chosen = run2({"T1": "hard"})
    df2 = build_export(chosen, P, "src", applied).set_index("Тема (id)")
    assert df2.loc["T1", "Форма теста"] == "hard" and df2.loc["T1", "Среднее post, %"] == 65.0
    assert df2.loc["T1", "Записей post в файле (валидных)"] == 20 and df2.loc["T1", "Записей post использовано"] == 10
    assert (df2["Часы утверждено"] == [5, 5]).all() and df2["Статус плана"].str.startswith("Утверждён").all()
    other = build_export(run2({"T1": "easy"}), P, "src", None).set_index("Тема (id)")
    assert other.loc["T1", "Среднее post, %"] == 95.0              # экспорт обновился вместе с формой


def test_standard_demo_is_unchanged_by_form_logic():
    a = analyze(generate_results(), topics_frame(), P)
    assert not a.unresolved and a.plan.budget == 30
    assert [a.plan.proposed()[t] for t in ("T1", "T2", "T3", "T4", "T5", "T6")] == [3, 5, 7, 6, 3, 6]
    assert a.summary["rows_used"] == a.summary["rows"]


# ============================================================================
# 4. Дубли заголовков
# ============================================================================
@pytest.mark.parametrize("header, expect", [
    (H.replace("max_score", "Score"), "score"),                                      # score и Score
    (H + ",score", "score"),                                                         # точный дубль
    ('"student_id","class_id","textbook","topic_id","stage","test_id","score"," SCORE ","max_score"', "score"),
    (H.replace(",", ";") + ";Score", "score"),                                       # другой разделитель
    ("﻿" + H + ",STUDENT_ID", "student_id"),                                    # BOM и регистр
])
def test_duplicate_headers_block_with_clear_message(header, expect):
    df, err = read_csv_bytes((header + "\n").encode("utf-8"))
    assert df is None and err and f"«{expect}»" in err and "повторяется" in err


def test_quoted_header_with_separator_inside_is_not_a_duplicate():
    df, err = read_csv_bytes((H + ',"a,b"\nS,K,А,T1,pre,F,1,2,3\n').encode("utf-8"))
    assert err is None and "a,b" in df.columns


def test_duplicate_headers_in_direct_dataframes_do_not_crash():
    res = validate_results(pd.DataFrame([["1", "2"]], columns=["score", "Score"]), None)
    assert not res.ok and "«score»" in res.errors[0]
    exact = pd.DataFrame([["1", "2"]], columns=["score", "score"])
    assert not validate_results(exact, None).ok
    top = pd.DataFrame([["T1", "A", "5", "3", "2", "9"]],
                       columns=["topic_id", "name", "hours", "min_hours", "max_delta", " HOURS "])
    tv = validate_topics(top)
    assert not tv.ok and "«hours»" in tv.errors[0]


def test_duplicate_header_error_reaches_analysis_as_block():
    raw = pd.DataFrame([["1", "2"]], columns=["score", "Score"])
    a = analyze(raw, topics_frame(), P)
    assert a.blocked and any("повторяется" in e for e in a.results_val.errors)


# ============================================================================
# 5. Ключ ученика
# ============================================================================
def colliding_rows(textbook2="А"):
    # (класс «X::Y», ученик «Z») и (класс «X», ученик «Y::Z») — простая склейка давала один ключ
    return [("Z", "X::Y", "А", "T1", "pre", "F", 40, 100), ("Z", "X::Y", "А", "T1", "post", "F", 60, 100),
            ("Y::Z", "X", textbook2, "T1", "pre", "F", 10, 100), ("Y::Z", "X", textbook2, "T1", "post", "F", 90, 100)]


def test_uid_has_no_collisions():
    assert make_uid("X::Y", "Z") != make_uid("X", "Y::Z")
    res = validate_results(results_df(colliding_rows()), {"T1"})
    assert res.ok and res.clean["uid"].nunique() == 2
    tm = topic_metrics(res.clean, topics_df(), Params(min_post_topic=1))
    assert tm.iloc[0].n_students == 2 and math.isclose(tm.iloc[0].mean_post, 75)
    assert data_summary(res.clean)["students"] == 2


def test_colliding_students_do_not_pair_with_each_other():
    # у первого ученика только pre, у второго только post — пар быть не должно
    rows = [("Z", "X::Y", "А", "T1", "pre", "F", 40, 100), ("Y::Z", "X", "А", "T1", "post", "F", 90, 100),
            ("A", "K9", "Б", "T1", "pre", "F", 40, 100), ("A", "K9", "Б", "T1", "post", "F", 60, 100)]
    a = analyze(results_df(rows), topics_df(), Params(min_pairs_compare=1, min_classes_compare=1))
    c = a.comparison.table.iloc[0]
    assert c.n_pairs_a == 0 and c.n_pairs_b == 1


def test_changed_textbook_exclusion_stays_correct_for_colliding_ids():
    rows = colliding_rows()
    rows[3] = ("Y::Z", "X", "Б", "T1", "post", "F", 90, 100)      # второй ученик сменил учебник
    res = validate_results(results_df(rows), {"T1"})
    assert res.changed_textbook_uids == {make_uid("X", "Y::Z")}   # первый ученик не задет
    assert make_uid("X::Y", "Z") not in res.changed_textbook_uids


# ============================================================================
# 6. Подписи тем (topic_id — ключ, name может повторяться)
# ============================================================================
def test_topic_labels_add_id_only_for_duplicate_names():
    tp = topics_df([("T1", "Дроби", 5, 3, 2), ("T2", "Дроби", 5, 3, 2), ("T3", "Проценты", 5, 3, 2)])
    assert validate_topics(tp).ok                                  # одинаковые name допустимы
    labels = topic_labels(tp)
    assert labels == {"T1": "Дроби (T1)", "T2": "Дроби (T2)", "T3": "Проценты"}
    assert len(set(labels.values())) == 3


def test_duplicate_names_keep_separate_metrics_in_core_tables():
    tp = topics_df([("T1", "Дроби", 5, 3, 2), ("T2", "Дроби", 5, 3, 2)])
    rows = [(f"S{i:02d}", "K01", "А", t, "post", "F", pct, 100) for i in range(1, 21)
            for t, pct in (("T1", 90), ("T2", 30))]
    a = analyze(results_df(rows), tp, P)
    mean = a.topic_table.set_index("topic_id")["mean_post"]
    assert mean["T1"] == 90 and mean["T2"] == 30
    tb = topic_by_textbook(validate_results(results_df(rows), {"T1", "T2"}).clean, tp, P)
    assert sorted(tb["topic_id"]) == ["T1", "T2"] and sorted(tb["mean_post"]) == [30, 90]
