import numpy as np

from kartatem.validation import (make_uid, read_csv_bytes, validate_results, validate_topics,
                                 REASON_SCORE_EMPTY, REASON_MAX_NONPOS, REASON_SCORE_OVER,
                                 REASON_SCORE_NEG, REASON_SCORE_BAD)
from helpers import results_df, topics_df

KNOWN = {"T1", "T2"}


def row(sid="S01", cls="K01", tb="А", topic="T1", stage="post", test="F1", score="10", mx="20"):
    return (sid, cls, tb, topic, stage, test, score, mx)


# ---------- чтение файла ----------
def test_read_empty_and_garbage_do_not_crash():
    assert read_csv_bytes(b"")[1] == "Файл пуст."
    assert read_csv_bytes(b"   \n  ")[1] == "Файл пуст."
    df, err = read_csv_bytes(bytes([0x00, 0xFF, 0xFE, 0x98, 0x01]))
    # либо понятная ошибка чтения, либо таблица, которую отвергнет валидатор — главное, без исключения
    assert err is not None or df is not None


def test_read_semicolon_bom_and_cp1251():
    text = "student_id;class_id;textbook;topic_id;stage;test_id;score;max_score\nS01;K01;Учебник А;T1;pre;F1;5;10\n"
    df, err = read_csv_bytes(("﻿" + text).encode("utf-8"))
    assert err is None and list(df.columns)[0] == "student_id" and df.loc[0, "textbook"] == "Учебник А"
    df2, err2 = read_csv_bytes(text.encode("cp1251"))
    assert err2 is None and df2.loc[0, "textbook"] == "Учебник А"


def test_header_only_file_is_blocked_not_crashing():
    df, err = read_csv_bytes(b"student_id,class_id,textbook,topic_id,stage,test_id,score,max_score\n")
    assert err is None
    res = validate_results(df, KNOWN)
    assert not res.ok and res.clean.empty


# ---------- структура ----------
def test_missing_columns_block():
    df = results_df([row()]).drop(columns=["max_score"])
    res = validate_results(df, KNOWN)
    assert not res.ok and "max_score" in res.errors[0]


def test_empty_identifiers_invalid_stage_unknown_topic_block():
    df = results_df([row(sid=""), row(sid="S02", stage="midterm"), row(sid="S03", topic="T9")])
    res = validate_results(df, KNOWN)
    assert not res.ok and res.clean.empty
    joined = " ".join(res.errors)
    assert "student_id" in joined and "stage" in joined and "T9" in joined


def test_stage_and_whitespace_normalized():
    df = results_df([row(stage=" PRE "), row(sid="S02", stage="Post", tb="  А  "), row(sid="S03", stage="до")])
    res = validate_results(df, KNOWN)
    assert res.ok
    assert set(res.clean["stage"]) == {"pre", "post"}
    assert set(res.clean["textbook"]) == {"А"}


# ---------- баллы ----------
def test_missing_scores_are_excluded_not_zero():
    df = results_df([row(sid="S01", score="10"), row(sid="S02", score=""), row(sid="S03", score="14")])
    res = validate_results(df, KNOWN)
    assert res.ok and len(res.clean) == 2
    assert res.exclusion_counts == {REASON_SCORE_EMPTY: 1}
    assert res.clean["pct"].mean() == 60.0   # (50 + 70) / 2; если бы пропуск стал нулём, было бы 40


def test_invalid_score_reasons():
    df = results_df([
        row(sid="S01", score="5", mx="0"),
        row(sid="S02", score="25", mx="20"),
        row(sid="S03", score="-1", mx="20"),
        row(sid="S04", score="abc", mx="20"),
        row(sid="S05", score="inf", mx="20"),
        row(sid="S06", score="10", mx="20"),
    ])
    res = validate_results(df, KNOWN)
    assert res.ok and len(res.clean) == 1
    c = res.exclusion_counts
    assert c[REASON_MAX_NONPOS] == 1 and c[REASON_SCORE_OVER] == 1 and c[REASON_SCORE_NEG] == 1
    assert c[REASON_SCORE_BAD] == 2
    assert np.isfinite(res.clean["pct"]).all()


def test_max_score_zero_never_divides():
    res = validate_results(results_df([row(score="0", mx="0"), row(sid="S02")]), KNOWN)
    assert res.ok and len(res.clean) == 1 and np.isfinite(res.clean["pct"]).all()


def test_decimal_comma_accepted():
    res = validate_results(results_df([row(score="7,5", mx="10")]), KNOWN)
    assert res.ok and res.clean.loc[0, "pct"] == 75.0


def test_fully_invalid_file_does_not_crash():
    df = results_df([row(score="", sid="S01"), row(score="x", sid="S02"), row(score="5", mx="0", sid="S03")])
    res = validate_results(df, KNOWN)
    assert not res.ok and res.clean.empty
    assert "ни одной валидной" in res.errors[0]
    assert res.n_excluded_rows == 3


# ---------- дубли ----------
def test_conflicting_duplicates_block_and_are_not_averaged():
    df = results_df([row(score="10"), row(score="18")])
    res = validate_results(df, KNOWN)
    assert not res.ok and res.clean.empty
    assert "Конфликтующие дубли" in " ".join(res.errors)


def test_exact_duplicates_are_collapsed_with_warning():
    df = results_df([row(score="10"), row(score="10.0")])
    res = validate_results(df, KNOWN)
    assert res.ok and len(res.clean) == 1
    assert any("точных дублей" in w for w in res.warnings)


def test_same_student_id_in_different_classes_is_not_a_duplicate():
    df = results_df([row(cls="K01", score="10"), row(cls="K02", score="18")])
    res = validate_results(df, KNOWN)
    assert res.ok and len(res.clean) == 2
    assert res.clean["uid"].nunique() == 2


def test_changed_textbook_detected():
    df = results_df([row(tb="А", stage="pre"), row(tb="Б", stage="post"), row(sid="S02", tb="А")])
    res = validate_results(df, KNOWN)
    assert res.ok and res.changed_textbook_uids == {make_uid("K01", "S01")}


# ---------- topics.csv ----------
def test_topics_valid():
    res = validate_topics(topics_df([("T1", "Дроби", 5, 3, 2), ("T2", "Пропорции", 4, 3, 2)]))
    assert res.ok and list(res.clean["hours"]) == [5, 4]


def test_topics_errors():
    assert not validate_topics(topics_df([("T1", "A", 5, 3, 2), ("T1", "B", 5, 3, 2)])).ok        # дубль id
    assert not validate_topics(topics_df([("T1", "A", 5.5, 3, 2)])).ok                            # нецелые часы
    assert not validate_topics(topics_df([("T1", "A", -1, 0, 2)])).ok                             # отрицательные
    assert not validate_topics(topics_df([("T1", "A", 2, 3, 2)])).ok                              # min > hours
    assert not validate_topics(topics_df([("T1", "A", "x", 3, 2)])).ok                            # не число
    assert not validate_topics(topics_df([("T1", "A", "inf", 3, 2)])).ok                          # не конечное
    assert not validate_topics(topics_df([("", "A", 5, 3, 2)])).ok                                # пустой id
    assert not validate_topics(None).ok
