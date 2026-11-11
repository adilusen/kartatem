import random

import pandas as pd

from kartatem.config import Params
from kartatem.hours import propose_hours, validate_plan, bounds
from helpers import topics_df

P = Params()

# Пример из проектного разбора (иллюстрация): порядок тем и усвоение
EXAMPLE_TOPICS = topics_df([
    ("T1", "Дроби", 5, 3, 2), ("T2", "Десятичные", 5, 3, 2), ("T3", "Проценты", 5, 3, 2),
    ("T4", "Уравнения", 6, 3, 2), ("T5", "Пропорции", 4, 3, 2), ("T6", "Геометрия", 5, 3, 2),
])
EXAMPLE_MASTERY = {"T1": 82, "T2": 74, "T3": 48, "T4": 61, "T5": 79, "T6": 55}
ALL_ELIGIBLE = {t: True for t in EXAMPLE_MASTERY}


def proposed_list(plan, topics):
    d = plan.proposed()
    return [d[t] for t in topics["topic_id"]]


def test_bounds_formula():
    assert bounds(5, 3, 2) == (3, 7)
    assert bounds(4, 3, 2) == (3, 6)      # hours − max_delta = 2, но минимум 3
    assert bounds(5, 0, 0) == (5, 5)


def test_example_from_design_review():
    plan = propose_hours(EXAMPLE_TOPICS, EXAMPLE_MASTERY, ALL_ELIGIBLE, P)
    assert proposed_list(plan, EXAMPLE_TOPICS) == [3, 5, 7, 6, 3, 6]
    assert sum(proposed_list(plan, EXAMPLE_TOPICS)) == 30 == plan.budget
    assert [m.step for m in plan.moves] == [1, 2, 3]
    assert [(m.from_topic, m.to_topic) for m in plan.moves] == [("T1", "T3"), ("T1", "T3"), ("T5", "T6")]
    assert all("→" in m.reason for m in plan.moves)
    assert validate_plan(EXAMPLE_TOPICS, plan.proposed()) == []


def test_budget_and_bounds_hold_on_random_scenarios():
    rnd = random.Random(2026)
    for _ in range(300):
        n = rnd.randint(1, 9)
        rows = []
        for i in range(n):
            hours = rnd.randint(0, 9)
            rows.append((f"T{i + 1}", f"Тема {i + 1}", hours, rnd.randint(0, hours), rnd.randint(0, 4)))
        tp = topics_df(rows)
        mastery = {f"T{i + 1}": rnd.uniform(0, 100) for i in range(n)}
        elig = {f"T{i + 1}": rnd.random() < 0.8 for i in range(n)}
        plan = propose_hours(tp, mastery, elig, P)
        assert plan.table["hours_proposed"].sum() == tp["hours"].astype(int).sum()
        for r in plan.table.itertuples():
            assert r.hours_low <= r.hours_proposed <= r.hours_high
            assert float(r.hours_proposed).is_integer()
            if not r.eligible:
                assert r.hours_proposed == r.hours_before
        assert validate_plan(tp, plan.proposed()) == []


def test_no_donors_is_normal_result_not_error():
    mastery = {t: 55 for t in EXAMPLE_MASTERY}           # все ниже цели, доноров нет
    plan = propose_hours(EXAMPLE_TOPICS, mastery, ALL_ELIGIBLE, P)
    assert plan.moves == []
    assert proposed_list(plan, EXAMPLE_TOPICS) == [5, 5, 5, 6, 4, 5]
    assert any("Допустимых переносов нет" in m for m in plan.messages)


def test_no_receivers_is_normal_result():
    mastery = {t: 90 for t in EXAMPLE_MASTERY}
    plan = propose_hours(EXAMPLE_TOPICS, mastery, ALL_ELIGIBLE, P)
    assert plan.moves == [] and any("нет тем с усвоением ниже" in m for m in plan.messages)


def test_bounds_can_block_every_move():
    tp = topics_df([("T1", "A", 5, 5, 2), ("T2", "B", 5, 3, 0)])    # донору нельзя ниже 5, получателю нельзя выше 5
    plan = propose_hours(tp, {"T1": 90, "T2": 40}, {"T1": True, "T2": True}, P)
    assert plan.moves == [] and any("границы" in m for m in plan.messages)


def test_topics_without_enough_post_are_never_changed():
    elig = dict(ALL_ELIGIBLE, T1=False, T3=False)         # T1 (донор) и T3 (получатель) — мало данных
    plan = propose_hours(EXAMPLE_TOPICS, EXAMPLE_MASTERY, elig, P)
    got = plan.proposed()
    assert got["T1"] == 5 and got["T3"] == 5
    assert sum(got.values()) == 30
    assert got["T5"] == 3 and got["T6"] == 6              # перенос идёт только между допущенными темами
    note = plan.table.set_index("topic_id").loc["T1", "note"]
    assert "мало данных" in note


def test_nobody_eligible_gives_message():
    plan = propose_hours(EXAMPLE_TOPICS, EXAMPLE_MASTERY, {t: False for t in EXAMPLE_MASTERY}, P)
    assert plan.moves == [] and any("достаточного числа post" in m for m in plan.messages)


def test_ties_resolve_by_topic_id():
    tp = topics_df([("T2", "B", 5, 3, 2), ("T1", "A", 5, 3, 2), ("T4", "D", 5, 3, 2), ("T3", "C", 5, 3, 2)])
    mastery = {"T1": 80, "T2": 80, "T3": 50, "T4": 50}          # одинаковые усвоения
    plan = propose_hours(tp, mastery, {t: True for t in mastery}, P)
    assert plan.moves[0].from_topic == "T1" and plan.moves[0].to_topic == "T3"
    assert plan.proposed() == {"T1": 3, "T2": 3, "T3": 7, "T4": 7}      # 2+2 часа донорам, 2+2 получателям
    again = propose_hours(tp, mastery, {t: True for t in mastery}, P)
    assert [m.reason for m in again.moves] == [m.reason for m in plan.moves]   # детерминированность


def test_priorities_use_original_mastery_no_invented_growth():
    plan = propose_hours(EXAMPLE_TOPICS, EXAMPLE_MASTERY, ALL_ELIGIBLE, P)
    # усвоение в таблице остаётся исходным: никакого «ожидаемого роста» после переноса
    assert dict(zip(plan.table.topic_id, plan.table.mastery_pct)) == {k: float(v) for k, v in EXAMPLE_MASTERY.items()}


# ---------- ручная правка ----------
def plan_ok():
    return {"T1": 3, "T2": 5, "T3": 7, "T4": 6, "T5": 3, "T6": 6}


def test_validate_plan_accepts_valid():
    assert validate_plan(EXAMPLE_TOPICS, plan_ok()) == []
    assert validate_plan(EXAMPLE_TOPICS, {"T1": 5.0, "T2": 5, "T3": 5, "T4": 6, "T5": 4, "T6": 5}) == []   # 5.0 — целое


def test_validate_plan_blocks_wrong_sum():
    p = plan_ok(); p["T2"] = 6
    errs = validate_plan(EXAMPLE_TOPICS, p)
    assert len(errs) == 1 and "Сумма часов 31" in errs[0] and "+1" in errs[0]


def test_validate_plan_blocks_non_integer_nan_missing_and_text():
    p = plan_ok(); p["T2"] = 4.5
    assert any("целыми" in e for e in validate_plan(EXAMPLE_TOPICS, p))
    p = plan_ok(); p["T2"] = float("nan")
    assert any("не заданы" in e for e in validate_plan(EXAMPLE_TOPICS, p))
    p = plan_ok(); del p["T2"]
    assert any("не заданы" in e for e in validate_plan(EXAMPLE_TOPICS, p))
    p = plan_ok(); p["T2"] = "пять"
    assert any("не является числом" in e for e in validate_plan(EXAMPLE_TOPICS, p))
    p = plan_ok(); p["T2"] = float("inf")
    assert validate_plan(EXAMPLE_TOPICS, p)
    p = plan_ok(); p["T9"] = 1
    assert any("неизвестные" in e for e in validate_plan(EXAMPLE_TOPICS, p))


def test_validate_plan_blocks_bounds_even_if_sum_is_ok():
    p = plan_ok(); p["T1"] = 2; p["T3"] = 8             # сумма 30, но 2 < 3 и 8 > 7
    errs = validate_plan(EXAMPLE_TOPICS, p)
    assert len(errs) == 2 and all("вне допустимого диапазона" in e for e in errs)
    assert not any("Сумма" in e for e in errs)
