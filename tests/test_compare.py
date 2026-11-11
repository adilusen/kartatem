import math

import pandas as pd

from kartatem.compare import (compare_textbooks, ST_OK, ST_INSUFFICIENT, ST_NEEDS_CHOICE,
                              ST_NO_COMMON, direction_text)
from kartatem.config import Params
from kartatem.metrics import topic_metrics
from kartatem.validation import make_uid
from kartatem.validation import validate_results
from helpers import results_df, topics_df, make_group

P = Params()


def prep(rows, topics=None):
    tp = topics if topics is not None else topics_df()
    ids = set(tp["topic_id"])
    res = validate_results(results_df(rows), ids)
    assert res.ok, res.errors
    return res, tp


def run(rows, params=P, choice=None, topics=None):
    res, tp = prep(rows, topics)
    out = compare_textbooks(res.clean, tp, params, choice, res.changed_textbook_uids)
    return out, res, tp


def big_rows(pre_a=40, post_a=60, pre_b=50, post_b=62, n=20):
    return (make_group("А", ["K01", "K02"], n, pre_pct=pre_a, post_pct=post_a)
            + make_group("Б", ["K03", "K04"], n, pre_pct=pre_b, post_pct=post_b))


def test_gain_on_same_paired_sample_and_direction():
    out, *_ = run(big_rows())
    r = out.table.iloc[0]
    assert r.status == ST_OK and r.n_pairs_a == 40 and r.n_pairs_b == 40
    # прирост = post − pre на одной и той же парной выборке
    assert math.isclose(r.gain_a, r.post_pair_a - r.pre_a) and math.isclose(r.gain_b, r.post_pair_b - r.pre_b)
    assert math.isclose(r.gain_a, 20) and math.isclose(r.gain_b, 12) and math.isclose(r.gain_diff, 8)
    assert math.isclose(r.pre_gap, -10)
    assert "больше у «А» на 8.0 п.п." in r.direction
    assert "не доказательство" in r.direction


def test_direction_text_has_no_winner_words():
    for diff in (5.0, -5.0, 0.01):
        t = direction_text("А", "Б", diff).lower()
        assert "лучше" not in t and "эффективнее" not in t and "победител" not in t


def test_only_valid_pairs_are_used_and_share_is_reported():
    rows = big_rows()
    # у 5 учеников группы А убираем pre → пар станет 35, post-учеников по-прежнему 40
    drop = {("S01", "K01"), ("S02", "K01"), ("S03", "K01"), ("S04", "K01"), ("S05", "K01")}
    rows = [r for r in rows if not ((r[0], r[1]) in drop and r[4] == "pre")]
    out, *_ = run(rows)
    r = out.table.iloc[0]
    assert r.n_post_a == 40 and r.n_pairs_a == 35 and math.isclose(r.pair_share_a, 35 / 40)
    assert r.status == ST_OK


def test_missing_pre_means_no_pair_and_blank_score_is_not_zero():
    rows = big_rows()
    raw = results_df(rows)
    raw.loc[(raw.student_id == "S01") & (raw.class_id == "K01") & (raw.stage == "pre"), "score"] = ""
    res = validate_results(raw, {"T1"})
    out = compare_textbooks(res.clean, topics_df(), P, None, res.changed_textbook_uids).table.iloc[0]
    assert out.n_pairs_a == 39 and math.isclose(out.gain_a, 20)   # если бы пропуск стал нулём, прирост был бы больше


def test_student_who_changed_textbook_is_excluded_from_pairs():
    rows = big_rows()
    # ученик S01 класса K01: pre по учебнику А, post по учебнику Б
    rows = [(r[0], r[1], "Б" if (r[0], r[1], r[4]) == ("S01", "K01", "post") else r[2], *r[3:]) for r in rows]
    out, res, _ = run(rows)
    r = out.table.iloc[0]
    assert make_uid("K01", "S01") in res.changed_textbook_uids
    assert r.n_pairs_a == 39 and r.n_pairs_b == 40
    assert any("смен" in w for w in r.warnings)


def test_student_with_different_textbooks_across_topics_is_excluded_everywhere():
    # S01/K01: в теме T1 учится по А (pre и post по А), в теме T2 — по Б. Внутри каждой темы пара «чистая»,
    # но ученик сменил учебник, поэтому в парное сравнение не входит нигде.
    tp = topics_df([("T1", "Тема 1", 5, 3, 2), ("T2", "Тема 2", 5, 3, 2)])
    rows = (make_group("А", ["K01", "K02"], 20, topic="T1") + make_group("Б", ["K03", "K04"], 20, topic="T1")
            + make_group("А", ["K01", "K02"], 20, topic="T2") + make_group("Б", ["K03", "K04"], 20, topic="T2"))
    rows = [(r[0], r[1], "Б" if (r[0], r[1], r[3]) == ("S01", "K01", "T2") else r[2], *r[3:]) for r in rows]
    out, res, _ = run(rows, topics=tp)
    assert res.changed_textbook_uids == {make_uid("K01", "S01")}
    t = out.table.set_index("topic_id")
    assert t.loc["T1", "n_pairs_a"] == 39 and t.loc["T1", "n_pairs_b"] == 40
    assert t.loc["T2", "n_pairs_a"] == 39 and t.loc["T2", "n_pairs_b"] == 40


def test_same_student_id_in_different_classes_pairs_correctly():
    out, *_ = run(big_rows())     # во всех классах те же S01..S20
    r = out.table.iloc[0]
    assert r.n_pairs_a == 40 and r.n_pairs_b == 40


def test_insufficient_pairs_keeps_post_but_blocks_gain_and_direction():
    rows = (make_group("А", ["K01", "K02"], 10, pre_pct=40, post_pct=60)
            + make_group("Б", ["K03", "K04"], 10, pre_pct=50, post_pct=62))
    out, *_ = run(rows)
    r = out.table.iloc[0]
    assert r.status == ST_INSUFFICIENT
    assert r.n_post_a == 20 and math.isclose(r.post_a, 60) and math.isclose(r.post_b, 62)
    assert r.n_pairs_a == 20
    assert math.isnan(r.gain_a) and math.isnan(r.gain_b) and math.isnan(r.gain_diff)
    assert math.isnan(r.pre_a) and "заблокировано" in r.direction
    assert any("недостаточно" in w for w in r.warnings)


def test_insufficient_classes_even_with_many_pairs():
    rows = (make_group("А", ["K01"], 40, pre_pct=40, post_pct=60)          # 40 пар, но 1 класс
            + make_group("Б", ["K03", "K04"], 20, pre_pct=50, post_pct=62))
    out, *_ = run(rows)
    r = out.table.iloc[0]
    assert r.n_pairs_a == 40 and r.classes_a == 1
    assert r.status == ST_INSUFFICIENT and math.isnan(r.gain_diff)


def test_pre_gap_is_only_a_warning():
    out, *_ = run(big_rows(pre_a=40, pre_b=60, post_a=60, post_b=70))
    r = out.table.iloc[0]
    assert r.status == ST_OK and not math.isnan(r.gain_diff)           # сравнение не блокируется
    assert any("Входные уровни" in w for w in r.warnings)
    out2, *_ = run(big_rows(pre_a=40, pre_b=45, post_a=60, post_b=70))
    assert not any("Входные уровни" in w for w in out2.table.iloc[0].warnings)
    # порог настраивается
    out3, *_ = run(big_rows(pre_a=40, pre_b=60, post_a=60, post_b=70), params=Params(pre_gap_warn_pp=30))
    assert not any("Входные уровни" in w for w in out3.table.iloc[0].warnings)


# ---------- формы теста ----------
def test_no_common_test_form_blocks_gain():
    # У А только форма FA, у Б только FB. Форм в теме две → сначала нужен выбор;
    # какую бы форму ни выбрали, она не представлена у второго учебника → сравнения нет.
    rows = (make_group("А", ["K01", "K02"], 20, test="FA") + make_group("Б", ["K03", "K04"], 20, test="FB"))
    out, *_ = run(rows)
    assert out.table.iloc[0].status == ST_NEEDS_CHOICE
    for form, missing in (("FA", "Б"), ("FB", "А")):
        r = run(rows, choice={"T1": form})[0].table.iloc[0]
        assert r.status == ST_NO_COMMON and r.test_id == form
        assert math.isnan(r.gain_a) and math.isnan(r.post_a) and math.isnan(r.gain_diff)
        assert any("не представлена" in w and f"«{missing}»" in w for w in r.warnings)


def test_multiple_forms_require_choice_and_are_not_mixed():
    # форма F1: у А прирост 20, у Б 12; форма F2: у обеих групп прирост 0
    rows = (make_group("А", ["K01", "K02"], 20, test="F1", pre_pct=40, post_pct=60, prefix="S")
            + make_group("Б", ["K03", "K04"], 20, test="F1", pre_pct=50, post_pct=62, prefix="S")
            + make_group("А", ["K01", "K02"], 20, test="F2", pre_pct=70, post_pct=70, prefix="X")
            + make_group("Б", ["K03", "K04"], 20, test="F2", pre_pct=70, post_pct=70, prefix="X"))
    out, *_ = run(rows)
    r = out.table.iloc[0]
    assert r.status == ST_NEEDS_CHOICE and r.candidates == ["F1", "F2"]
    assert out.selection_needed() == {"T1": ["F1", "F2"]}
    assert math.isnan(r.gain_a)
    chosen, *_ = run(rows, choice={"T1": "F1"})
    c = chosen.table.iloc[0]
    assert c.status == ST_OK and c.test_id == "F1"
    assert math.isclose(c.gain_a, 20) and math.isclose(c.gain_b, 12) and c.n_pairs_a == 40   # F2 не примешана
    bogus, *_ = run(rows, choice={"T1": "NOPE"})
    assert bogus.table.iloc[0].status == ST_NEEDS_CHOICE


def test_pre_and_post_must_use_same_test_form():
    # pre по форме F1, post по форме F2 у обеих групп. Формы не смешиваются ни при каком выборе.
    rows = []
    for tb, classes in (("А", ["K01", "K02"]), ("Б", ["K03", "K04"])):
        for c in classes:
            for i in range(1, 21):
                rows.append((f"S{i:02d}", c, tb, "T1", "pre", "F1", 40, 100))
                rows.append((f"S{i:02d}", c, tb, "T1", "post", "F2", 60, 100))
    assert run(rows)[0].table.iloc[0].status == ST_NEEDS_CHOICE
    r2 = run(rows, choice={"T1": "F2"})[0].table.iloc[0]       # только post: пар нет, прироста нет
    assert r2.status == ST_INSUFFICIENT and r2.n_pairs_a == 0 and math.isnan(r2.gain_a)
    assert r2.n_post_a == 40 and math.isclose(r2.post_a, 60)    # post-only анализ темы остаётся доступным
    r1 = run(rows, choice={"T1": "F1"})[0].table.iloc[0]        # F1 есть только в pre → post нет вообще
    assert r1.status == ST_NO_COMMON and math.isnan(r1.gain_a)


def test_single_textbook_gives_message_not_crash():
    rows = make_group("А", ["K01", "K02"], 20)
    out, *_ = run(rows)
    assert out.table.empty and out.messages


# ---------- независимость допусков ----------
def test_topic_can_be_assessed_while_comparison_is_blocked():
    rows = (make_group("А", ["K01", "K02"], 10, pre_pct=40, post_pct=60)
            + make_group("Б", ["K03", "K04"], 10, pre_pct=50, post_pct=62))
    out, res, tp = run(rows)
    tm = topic_metrics(res.clean, tp, P).iloc[0]
    assert tm.n_students == 40 and tm.status != "Мало данных"      # темы оценить хватает
    assert out.table.iloc[0].status == ST_INSUFFICIENT            # а сравнивать — нет


def test_hours_eligibility_and_comparison_thresholds_are_independent():
    rows = big_rows()
    res, tp = prep(rows)
    base_t = topic_metrics(res.clean, tp, P).iloc[0]
    base_c = compare_textbooks(res.clean, tp, P, None, set()).table.iloc[0]
    assert base_t.hours_eligible and base_c.status == ST_OK
    # строгий порог для часов не меняет допуск сравнения
    strict_hours = Params(min_post_hours=1000)
    t2 = topic_metrics(res.clean, tp, strict_hours).iloc[0]
    c2 = compare_textbooks(res.clean, tp, strict_hours, None, set()).table.iloc[0]
    assert not t2.hours_eligible and c2.status == ST_OK
    # строгий порог для сравнения не меняет допуск часов и оценку темы
    strict_cmp = Params(min_pairs_compare=1000)
    t3 = topic_metrics(res.clean, tp, strict_cmp).iloc[0]
    c3 = compare_textbooks(res.clean, tp, strict_cmp, None, set()).table.iloc[0]
    assert t3.hours_eligible and t3.status == base_t.status and c3.status == ST_INSUFFICIENT
    # порог оценки темы не меняет допуск часов
    t4 = topic_metrics(res.clean, tp, Params(min_post_topic=1000)).iloc[0]
    assert t4.status == "Мало данных" and t4.hours_eligible
