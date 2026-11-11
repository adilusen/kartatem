import math

from kartatem.config import Params
from kartatem.metrics import (topic_metrics, topic_by_textbook, STATUS_HARD, STATUS_OK, STATUS_LOW)
from kartatem.validation import validate_results
from helpers import results_df, topics_df, make_group

P = Params()


def clean_of(rows):
    res = validate_results(results_df(rows), {"T1"})
    assert res.ok, res.errors
    return res.clean


def post_rows(textbook, cls, pcts, topic="T1", start=1):
    return [(f"S{start + i:02d}", cls, textbook, topic, "post", "F1", p, 100) for i, p in enumerate(pcts)]


def test_topic_mean_is_weighted_by_students_not_by_class():
    # класс K01: 10 учеников по 100 %, класс K02: 2 ученика по 0 % → среднее по ученикам 83.3, а не 50
    rows = post_rows("А", "K01", [100] * 10) + post_rows("А", "K02", [0] * 2)
    t = topic_metrics(clean_of(rows), topics_df(), Params(min_post_topic=5)).iloc[0]
    assert math.isclose(t.mean_post, 1000 / 12, rel_tol=1e-9)
    assert t.n_students == 12 and t.n_classes == 2 and t.n_post_rows == 12


def test_counts_are_separate_and_pre_is_not_in_post_mean():
    rows = make_group("А", ["K01", "K02"], 10, pre_pct=20, post_pct=70)
    t = topic_metrics(clean_of(rows), topics_df(), P).iloc[0]
    assert t.n_students == 20 and t.n_classes == 2
    assert t.n_post_rows == 20 and t.n_pre_rows == 20
    assert t.mean_post == 70.0


def test_share_below_cut_is_strict():
    rows = post_rows("А", "K01", [49.9, 50, 50.1, 80] * 5)
    t = topic_metrics(clean_of(rows), topics_df(), P).iloc[0]
    assert t.share_below == 25.0      # только 49.9 ниже 50; ровно 50 — не «ниже»


def test_missing_post_is_excluded_not_counted_as_zero():
    raw = results_df(post_rows("А", "K01", [80] * 15))
    raw.loc[0, "score"] = ""
    res = validate_results(raw, {"T1"})
    t = topic_metrics(res.clean, topics_df(), P).iloc[0]
    assert t.n_students == 14 and t.mean_post == 80.0


def test_status_low_data_never_flags_hard_even_with_bad_scores():
    rows = post_rows("А", "K01", [10] * 14)
    t = topic_metrics(clean_of(rows), topics_df(), P).iloc[0]
    assert t.status == STATUS_LOW and "14 < 15" in t.reason


def test_status_hard_by_mean_and_boundaries():
    # среднее ровно 60 и доля < 40 % → не трудная (строгое «< 60»)
    t = topic_metrics(clean_of(post_rows("А", "K01", [60] * 15)), topics_df(), P).iloc[0]
    assert t.status == STATUS_OK
    t = topic_metrics(clean_of(post_rows("А", "K01", [59] * 15)), topics_df(), P).iloc[0]
    assert t.status == STATUS_HARD


def test_status_hard_by_share_even_when_mean_is_high():
    # 6 из 15 = 40 % ниже 50; остальные 100 → среднее выше 60
    rows = post_rows("А", "K01", [10] * 6 + [100] * 9)
    t = topic_metrics(clean_of(rows), topics_df(), P).iloc[0]
    assert t.mean_post >= 60 and t.share_below == 40.0 and t.status == STATUS_HARD
    rows = post_rows("А", "K01", [10] * 5 + [100] * 10)     # 33.3 % < 40 %
    assert topic_metrics(clean_of(rows), topics_df(), P).iloc[0].status == STATUS_OK


def test_topic_without_data_is_low_data_not_crash():
    rows = make_group("А", ["K01"], 20)
    tp = topics_df([("T1", "Тема 1", 5, 3, 2), ("T2", "Тема 2", 5, 3, 2)])
    res = validate_results(results_df(rows), {"T1", "T2"})
    out = topic_metrics(res.clean, tp, P)
    t2 = out[out.topic_id == "T2"].iloc[0]
    assert t2.n_students == 0 and math.isnan(t2.mean_post) and t2.status == STATUS_LOW


def test_topic_by_textbook_matrix():
    rows = make_group("А", ["K01"], 20, post_pct=50) + make_group("Б", ["K02"], 10, post_pct=90)
    m = topic_by_textbook(clean_of(rows), topics_df(), P)
    a = m[m.textbook == "А"].iloc[0]
    b = m[m.textbook == "Б"].iloc[0]
    assert a.mean_post == 50 and a.n_students == 20 and a.enough
    assert b.mean_post == 90 and b.n_students == 10 and not b.enough
