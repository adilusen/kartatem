"""v4: фиксации тем, сценарии, контекст, JSON-восстановление, паспорт и CSV текущего плана."""
import itertools
import json

import pytest

from kartatem.config import Params
from kartatem.demo import generate_results, topics_frame
from kartatem.export import build_export
from kartatem.hours import bounds, validate_plan
from kartatem.passport import build_passport
from kartatem.pipeline import analyze
from kartatem.plan_io import parse_plan, plan_to_json, restore_plan
from kartatem.scenarios import (MAX_SCENARIOS, Scenario, add_scenario, compare_scenarios, context_id,
                                context_parts, scenario_problems)

IDS = ["T1", "T2", "T3", "T4", "T5", "T6"]


@pytest.fixture(scope="module")
def demo():
    r = generate_results()
    return r, analyze(r, topics_frame(), Params()).topics       # проверенная таблица тем (целые часы)


def hours_list(plan):
    return [plan.proposed()[t] for t in IDS]


def test_control_example_without_and_with_lock(demo):
    r, t = demo
    assert hours_list(analyze(r, t, Params()).plan) == [3, 5, 7, 6, 3, 6]
    p = analyze(r, t, Params(), locked=["T1"]).plan
    assert hours_list(p) == [5, 5, 6, 6, 3, 5]
    assert sum(p.proposed().values()) == 30
    assert p.flows() == [("T5", "T3", 1)]


@pytest.mark.parametrize("k", range(0, 7))
def test_every_lock_set_keeps_budget_bounds_and_locks(demo, k):
    r, t = demo
    a = analyze(r, t, Params())
    for locked in itertools.combinations(IDS, k):
        p1, p2 = a.plan_for(locked), a.plan_for(locked)
        assert p1.proposed() == p2.proposed()                       # детерминированно
        assert sum(p1.proposed().values()) == 30
        for row in t.itertuples():
            h = p1.proposed()[row.topic_id]
            lo, hi = bounds(row.hours, row.min_hours, row.max_delta)
            assert lo <= h <= hi
            if row.topic_id in locked:
                assert h == row.hours
        assert validate_plan(t, p1.proposed(), locked) == []


def test_ranking_uses_original_results_not_moved_hours(demo):
    r, t = demo
    p = analyze(r, t, Params(), locked=["T5"]).plan
    assert {f for f, _, _ in p.flows()} == {"T1"}                    # единственный донор — T1
    assert hours_list(p) == [3, 5, 7, 6, 4, 5]                        # оба часа — в самую слабую T3


def test_all_eligible_locked_means_no_transfer(demo):
    r, t = demo
    p = analyze(r, t, Params(), locked=IDS).plan
    assert not p.moves and hours_list(p) == list(t["hours"])
    assert any("все темы с достаточными данными зафиксированы" in m for m in p.messages)


def test_validate_plan_rejects_change_of_locked_topic(demo):
    _, t = demo
    good = {"T1": 3, "T2": 5, "T3": 7, "T4": 6, "T5": 3, "T6": 6}
    assert validate_plan(t, good) == []
    errs = validate_plan(t, good, ["T1"])
    assert errs and "зафиксирована на исходных 5 ч" in errs[0]
    assert any("неизвестных тем" in e for e in validate_plan(t, good, ["T9"]))


def test_scenario_store_limits_names_and_replacement():
    sc = lambda n: Scenario(n, {"T1": 1}, [], "c")
    lst, err = add_scenario([], sc("  "))
    assert err and lst == []
    lst = []
    for n in ("А", "Б", "В"):
        lst, err = add_scenario(lst, sc(n))
        assert err is None
    same, err = add_scenario(lst, Scenario("Б", {"T1": 2}, [], "c"))      # замена одноимённого
    assert err is None and [s.name for s in same] == ["А", "Б", "В"] and same[1].hours == {"T1": 2}
    _, err = add_scenario(lst, sc("Г"))
    assert err and str(MAX_SCENARIOS) in err


def test_compare_scenarios_and_context(demo):
    r, t = demo
    a = analyze(r, t, Params())
    ctx = context_id(context_parts("sig", Params(), a.form_key))
    s1 = Scenario("Алгоритм", a.plan.proposed(), [], ctx)
    s2 = Scenario("Без дробей", a.plan_for(["T1"]).proposed(), ["T1"], ctx)
    df = compare_scenarios(t, s1, s2, {x: x for x in IDS}).set_index("Тема")
    assert list(df["Разница Б − А, ч"]) == [2, 0, -1, 0, 0, -1]
    assert df.loc["T1", "Фиксация"] == "Б" and df.loc["T2", "Фиксация"] == "—"
    assert list(df["Исходно, ч"]) == list(t["hours"])
    assert scenario_problems(s2, t, ctx) == []
    other = context_id(context_parts("sig", Params(donor_from=80.0), a.form_key))
    assert other != ctx and context_id(context_parts("sig2", Params(), a.form_key)) != ctx
    assert scenario_problems(s2, t, other)                            # чужой контекст — не применять


def _doc(a, t, *, sig="s1", draft=None, scen=None, locked=(), approved=None, params=Params()):
    parts = context_parts(sig, params, a.form_key)
    ctx = context_id(parts)
    scs = scen if scen is not None else [Scenario("Без дробей", a.plan_for(["T1"]).proposed(), ["T1"], ctx)]
    raw = plan_to_json(context=ctx, context_parts=parts, params=params, test_choice={}, locked=list(locked),
                       draft=draft or a.plan.proposed(), scenarios=scs, approved=approved, topics=t)
    return raw, ctx


def test_json_roundtrip_restores_settings_scenarios_but_not_approval(demo):
    r, t = demo
    a = analyze(r, t, Params())
    raw, ctx = _doc(a, t, locked=["T1"], draft=a.plan_for(["T1"]).proposed(),
                    approved={"name": "Без дробей", "hours": {}, "locked": []})
    doc, errs = parse_plan(raw, source_sig="s1", topics=t)
    assert errs == []
    rest, errs = restore_plan(doc, topics=t)
    assert errs == [] and rest.context == ctx and rest.locked == ["T1"]
    assert rest.params == Params() and [s.name for s in rest.scenarios] == ["Без дробей"]
    assert rest.draft == {"T1": 5, "T2": 5, "T3": 6, "T4": 6, "T5": 3, "T6": 5}
    assert not hasattr(rest, "approved")                               # утверждению из файла не доверяем
    text = raw.decode("utf-8")
    assert "student_id" not in text and "class_id" not in text and "S0" not in text   # нет поученичных данных


def test_json_errors_are_clear(demo):
    r, t = demo
    a = analyze(r, t, Params())
    raw, _ = _doc(a, t)
    assert parse_plan(raw, source_sig="другой", topics=t)[1][0].startswith("План сохранён для других исходных данных")
    assert "корректным JSON" in parse_plan(b"{oops", source_sig="s1", topics=t)[1][0]
    assert "не файл плана" in parse_plan(b'{"a":1}', source_sig="s1", topics=t)[1][0]
    d = json.loads(raw); d["version"] = 99
    assert "версия" in parse_plan(json.dumps(d).encode(), source_sig="s1", topics=t)[1][0]
    assert "больше" in parse_plan(b" " * 300_000, source_sig="s1", topics=t)[1][0]
    d = json.loads(raw); d["context"]["topics"] = d["context"]["topics"][:-1]
    assert any("Набор тем" in e for e in parse_plan(json.dumps(d).encode(), source_sig="s1", topics=t)[1])

    def restore_errs(mut):
        d = json.loads(raw); mut(d)
        doc, e = parse_plan(json.dumps(d).encode(), source_sig="s1", topics=t)
        assert e == []
        return " ".join(restore_plan(doc, topics=t)[1])
    assert "неизвестная тема «T9»" in restore_errs(lambda d: d["draft"].__setitem__("T9", 1))
    assert "целым числом" in restore_errs(lambda d: d["draft"].__setitem__("T2", 5.5))
    assert "не равна бюджету" in restore_errs(lambda d: d["draft"].__setitem__("T2", 6))
    assert "вне допустимого" in restore_errs(lambda d: d["scenarios"][0]["hours"].update(T3=9, T2=3))
    assert "зафиксирована" in restore_errs(lambda d: d["scenarios"][0]["hours"].update(T1=4, T3=7))
    assert "Параметры" in restore_errs(lambda d: d["settings"]["params"].__setitem__("target", "70"))
    assert "Порог доноров" in restore_errs(lambda d: d["settings"]["params"].__setitem__("donor_from", 10))
    assert "Фиксация для неизвестных" in restore_errs(lambda d: d.__setitem__("locked", ["T9"]))


def test_passport_and_csv_match_current_plan_and_escape_text(demo):
    r, t = demo
    a = analyze(r, t, Params(), locked=["T1"])
    hours = a.plan.proposed()
    html = build_passport(a, Params(), source_text="<script>alert(1)</script>", synthetic=True,
                          scenario_name="<b>Мой</b>", hours=hours, locked=["T1"], approved=False, context="c1")
    assert "<script>" not in html and "&lt;script&gt;" in html and "&lt;b&gt;Мой&lt;/b&gt;" in html
    assert "НЕ УТВЕРЖДЁН" in html and "Синтетические демо-данные" in html
    assert "зафиксировано: исходные 5 ч" in html and "Бюджет: <b>30 ч</b>" in html
    ok = build_passport(a, Params(), source_text="x", synthetic=False, scenario_name="Без дробей", hours=hours,
                        locked=["T1"], approved=True, context="c1")
    assert "Утверждён методистом: сценарий «Без дробей»" in ok and "НЕ УТВЕРЖДЁН" not in ok
    bad = dict(hours, T2=6)
    assert "НЕ УТВЕРЖДЁН" in build_passport(a, Params(), source_text="x", synthetic=False, scenario_name="x",
                                            hours=bad, locked=[], approved=False, context="c1")

    df = build_export(a, Params(), "src", None, scenario="Черновик", locked=["T1"], draft=hours).set_index("Тема (id)")
    assert list(df["Часы в неутверждённом черновике"]) == [5, 5, 6, 6, 3, 5]
    assert set(df["Часы утверждено"]) == {""} and df.loc["T1", "Не изменять (зафиксировано)"] == "да"
    assert list(df["Часы предложено алгоритмом"]) == [5, 5, 6, 6, 3, 5]
    df = build_export(a, Params(), "src", hours, scenario="Без дробей", locked=["T1"]).set_index("Тема (id)")
    assert list(df["Часы утверждено"]) == [5, 5, 6, 6, 3, 5] and set(df["Сценарий"]) == {"Без дробей"}
    assert set(df["Статус плана"]) == {"Утверждён методистом в этой сессии"}
