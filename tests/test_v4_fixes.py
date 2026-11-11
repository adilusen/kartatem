"""Регрессии v4-fixed: контекст сценариев при восстановлении, единое решение в трёх экспортах,
невалидный JSON даёт ошибку, а не исключение."""
import json

import pytest

from kartatem.config import Params
from kartatem.decision import build_decision
from kartatem.demo import generate_results, topics_frame
from kartatem.export import build_export
from kartatem.pipeline import analyze
from kartatem.plan_io import parse_plan, plan_to_json, restore_plan
from kartatem.scenarios import Scenario, context_id, context_parts, scenario_problems
from tests.test_app import click, go, new_app
from tests.test_app_v2 import multiform_results

SIG = "sig-demo"


@pytest.fixture(scope="module")
def demo():
    r = generate_results()
    return r, analyze(r, topics_frame(), Params()).topics


def export_doc(a, t, params, scenarios, *, draft=None, locked=(), test_choice=None):
    parts = context_parts(SIG, params, a.form_key)
    ctx = context_id(parts)
    raw = plan_to_json(context=ctx, context_parts=parts, params=params, test_choice=test_choice or {},
                       locked=list(locked), draft=draft or a.plan.proposed(), scenarios=scenarios,
                       approved=None, topics=t)
    return raw, ctx


def roundtrip(raw, t):
    doc, errs = parse_plan(raw, source_sig=SIG, topics=t)
    assert errs == []
    rest, errs = restore_plan(doc, topics=t)
    assert errs == [], errs
    return rest


# ---------- 1. Контекст каждого сценария переживает экспорт/импорт
def test_stale_scenario_stays_stale_after_param_change_roundtrip(demo):
    r, t = demo
    a75 = analyze(r, t, Params(donor_from=75.0))
    ctx75 = context_id(context_parts(SIG, Params(donor_from=75.0), a75.form_key))
    old = Scenario("А", a75.plan.proposed(), [], ctx75)
    p76 = Params(donor_from=76.0)
    a76 = analyze(r, t, p76)
    raw, ctx76 = export_doc(a76, t, p76, [old])
    assert ctx76 != ctx75 and scenario_problems(old, t, ctx76)            # до экспорта — старый
    rest = roundtrip(raw, t)
    assert rest.context == ctx76
    sc = rest.scenarios[0]
    assert sc.context == ctx75                                            # свой контекст, не общий
    probs = scenario_problems(sc, t, ctx76)
    assert probs and "пересчитайте" in probs[0]                           # утвердить без пересчёта нельзя


def test_stale_scenario_stays_stale_after_form_change_roundtrip():
    r = multiform_results()
    t = analyze(r, topics_frame(), Params(), {"T1": "M6-T1-F1"}).topics
    a1 = analyze(r, t, Params(), {"T1": "M6-T1-F1"})
    a2 = analyze(r, t, Params(), {"T1": "FORM2"})
    c1 = context_id(context_parts(SIG, Params(), a1.form_key))
    raw, c2 = export_doc(a2, t, Params(), [Scenario("Форма 1", a1.plan.proposed(), [], c1)],
                         test_choice={"T1": "FORM2"})
    assert c1 != c2
    rest = roundtrip(raw, t)
    assert rest.test_choice == {"T1": "FORM2"} and rest.scenarios[0].context == c1
    assert scenario_problems(rest.scenarios[0], t, c2)


@pytest.mark.parametrize("bad", [None, "", "x", 12, [], "ABCDEF123456", "0123456789abc"])
def test_missing_or_corrupt_scenario_context_is_rejected(demo, bad):
    r, t = demo
    a = analyze(r, t, Params())
    raw, ctx = export_doc(a, t, Params(), [Scenario("А", a.plan.proposed(), [], "0" * 12)])
    d = json.loads(raw)
    if bad is None:
        del d["scenarios"][0]["context"]
    else:
        d["scenarios"][0]["context"] = bad
    doc, errs = parse_plan(json.dumps(d).encode(), source_sig=SIG, topics=t)
    assert errs == []
    rest, errs = restore_plan(doc, topics=t)
    assert rest is None and any("контекст данных отсутствует или записан неверно" in e for e in errs)


# ---------- 2. Три экспорта — одно и то же показанное решение
def _decision_from_state(ss, a, params):
    g = lambda k, default=None: ss[k] if k in ss else default
    return build_decision(a, params, source=ss["source"], context=ss["_ctx"], context_parts=ss["_ctx_parts"],
                          test_choice=g("test_choice", {}), scenarios=g("scenarios", []), applied=g("applied"),
                          applied_name=g("applied_name"), applied_locked=g("applied_locked", []),
                          draft=ss["draft"], locked=g("locked", []))


def test_json_csv_html_follow_approved_plan_not_invalid_draft():
    at = go(click(new_app(), "Открыть демо"), "Сценарии")
    at = click(at, "Утвердить черновик")
    approved = dict(at.session_state["applied"])
    assert sum(approved.values()) == 30
    at.number_input(key="hrs_T2").set_value(6).run()                      # черновик 31, утверждён 30
    assert sum(at.session_state["draft"].values()) == 31 and at.session_state["applied"] == approved
    at = go(at, "Решение")
    assert not at.exception
    ss = at.session_state
    a = ss["_analysis"]
    d = _decision_from_state(ss, a, a.params_used)
    assert d.approved and d.hours == approved and d.plan_json is not None
    doc = json.loads(d.plan_json)
    assert doc["draft"] == approved and sum(doc["draft"].values()) == 30
    pd_, errs = parse_plan(d.plan_json, source_sig=ss["source"]["sig"], topics=a.topics)
    rest, errs = restore_plan(pd_, topics=a.topics)
    assert errs == [] and rest.draft == approved                          # восстановимо: 30 ч
    assert "approved" not in rest.__dict__                                # утверждение не переносится
    csv = d.csv.decode("utf-8-sig")
    assert "Утверждён методистом в этой сессии" in csv
    assert "Утверждён методистом: сценарий «Черновик»" in d.html
    assert all(not e.proto.disabled for e in at.get("download_button"))


def test_invalid_unapproved_draft_disables_json():
    at = go(click(new_app(), "Открыть демо"), "Сценарии")
    at.number_input(key="hrs_T2").set_value(6).run()
    at = go(at, "Решение")
    ss = at.session_state
    d = _decision_from_state(ss, ss["_analysis"], ss["_analysis"].params_used)
    assert not d.approved and d.errors and d.plan_json is None
    btn = next(e for e in at.get("download_button") if "olshem_plan" in str(e.proto) or "JSON" in e.proto.label)
    assert btn.proto.disabled
    assert any("План (JSON) недоступен" in c.value for c in at.caption)


def test_csv_status_is_truthful_for_manual_and_algorithm_drafts(demo):
    r, t = demo
    a = analyze(r, t, Params())
    alg = a.plan.proposed()
    manual = dict(alg, T4=7, T6=5)
    df = build_export(a, Params(), "s", None, scenario="Черновик", locked=[], draft=manual)
    assert set(df["Статус плана"]) == {"Не утверждён: черновик изменён вручную"}
    assert list(df["Часы в неутверждённом черновике"]) == [manual[x] for x in df["Тема (id)"]]
    assert list(df["Часы предложено алгоритмом"]) == [alg[x] for x in df["Тема (id)"]]
    assert set(df["Часы утверждено"]) == {""}
    df = build_export(a, Params(), "s", None, scenario="Черновик", locked=[], draft=alg)
    assert set(df["Статус плана"]) == {"Не утверждён: черновик совпадает с предложением алгоритма"}


# ---------- 3. Невалидный JSON — понятная ошибка, а не исключение
def _base(demo):
    r, t = demo
    a = analyze(r, t, Params())
    raw, _ = export_doc(a, t, Params(), [Scenario("А", a.plan.proposed(), [], "0" * 12)])
    return json.loads(raw), t


def _errors(d, t):
    doc, errs = parse_plan(json.dumps(d).encode() if not isinstance(d, bytes) else d, source_sig=SIG, topics=t)
    if errs:
        return errs
    rest, errs = restore_plan(doc, topics=t)
    assert rest is None
    return errs


@pytest.mark.parametrize("mutate, expect", [
    (lambda d: d["context"]["topics"][0].__setitem__("topic_id", []), "строковым topic_id"),
    (lambda d: d["context"].__setitem__("topics", "T1"), "строковым topic_id"),
    (lambda d: d["context"]["topics"].append(dict(d["context"]["topics"][0])), "повторяются"),
    (lambda d: d["context"].__setitem__("id", 5), "context.id"),
    (lambda d: d["context"].__setitem__("source", ["x"]), "context.source"),
    (lambda d: d.__setitem__("draft", {}), "указаны не все темы"),
    (lambda d: d.__setitem__("draft", []), "ожидается объект"),
    (lambda d: d["scenarios"].append(dict(d["scenarios"][0])), "название повторяется"),
    (lambda d: d["scenarios"].append(dict(d["scenarios"][0], name=" А ")), "название повторяется"),
    (lambda d: d["scenarios"][0].__setitem__("name", "   "), "нет названия"),
    (lambda d: d["scenarios"][0].__setitem__("hours", {}), "указаны не все темы"),
    (lambda d: d["scenarios"][0].__setitem__("locked", [1]), "строковых topic_id"),
    (lambda d: d.__setitem__("locked", ["T1", "T1"]), "повторяются"),
    (lambda d: d.__setitem__("locked", {"T1": 1}), "строковых topic_id"),
    (lambda d: d.__setitem__("settings", []), "settings"),
    (lambda d: d["settings"]["params"].__setitem__("target", True), "числом"),
    (lambda d: d["settings"]["params"].__setitem__("min_post_topic", 10 ** 400), "вне допустимого"),
    (lambda d: d["draft"].__setitem__("T1", 10 ** 400), "целым числом"),
    (lambda d: d["draft"].__setitem__("T1", True), "целым числом"),
])
def test_bad_structure_is_an_error_not_exception(demo, mutate, expect):
    d, t = _base(demo)
    mutate(d)
    errs = _errors(d, t)
    assert errs and any(expect in e for e in errs), errs


def test_non_finite_and_deep_json_are_errors(demo):
    d, t = _base(demo)
    raw = json.dumps(d).replace('"T1": 3', '"T1": 1e400', 1).encode()
    assert any("целым числом" in e for e in _errors(raw, t))
    raw = json.dumps(d).replace('"T1": 3', '"T1": NaN', 1).encode()
    assert any("корректным JSON" in e for e in _errors(raw, t))
    deep = b'{"format":"olshem-plan","version":1,"context":' + b"[" * 50000 + b"]" * 50000 + b"}"
    assert _errors(deep, t)
