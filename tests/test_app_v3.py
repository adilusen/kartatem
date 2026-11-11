"""v3: Обзор, переходы, постоянные цвета учебников, состояния ошибок (AppTest, без пикселей и CSS)."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from kartatem.compare import fmt_gap_vs_threshold
from kartatem.config import Params
from kartatem.demo import generate_results, topics_frame
from kartatem.pipeline import analyze
from kartatem.theme import textbook_colors
from tests.test_app_v2 import app_with, charts, go, click, no_exc, three_textbooks, multiform_results
from tests.test_app import texts, new_app

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


def open_demo():
    return click(new_app(), "Открыть демо")


def test_overview_without_data_has_no_invented_numbers():
    at = new_app()
    assert at.session_state["nav"] == "Обзор" and not at.metric and not at.dataframe
    assert [b.label for b in at.button][:2] == ["Открыть демо", "Загрузить свои данные"]
    at = click(at, "Загрузить свои данные")
    assert at.session_state["nav"] == "Данные" and "source" not in at.session_state


def test_overview_demo_numbers_come_from_the_core():
    at = open_demo()
    no_exc(at)
    core = analyze(generate_results(), topics_frame(), Params())
    m = metrics(at)
    assert m["Учеников в анализе"] == str(core.summary["students_used"]) == "164"
    assert m["Трудных тем"] == "3 из 6"
    assert m["Бюджет часов"] == "30 ч" and m["Часов к переносу"] == "3 ч"
    body = texts(at)
    assert "Синтетические демо-данные" in body
    assert "Проценты" in body and "Геометрия: площади" in body
    assert "**2 ч**" in body and "**1 ч**" in body                      # T1→T3, T5→T6
    assert "не прогноз" in body
    assert not any("рейтинг" in e.value.lower() and "балл" in e.value.lower() for e in at.metric)


def test_overview_buttons_keep_source_forms_and_applied_plan():
    at = go(app_with(multiform_results()), "Сценарии")
    at = at.sidebar.selectbox(key="tc_T1").select("FORM2").run()
    at = click(at, "Утвердить черновик")
    applied, source, choice = at.session_state["applied"], at.session_state["source"], dict(at.session_state["test_choice"])
    assert applied is not None
    at = go(at, "Обзор")
    at = click(at, "Открыть план часов")
    no_exc(at)
    assert at.session_state["nav"] == "Сценарии"
    assert at.session_state["applied"] == applied and at.session_state["test_choice"] == choice
    assert at.session_state["source"] is source
    at = go(at, "Обзор")
    at = click(at, "Сравнить учебники")
    assert at.session_state["nav"] == "Учебники" and at.session_state["applied"] == applied


def test_overview_unresolved_form_shows_state_not_fake_metrics():
    at = app_with(multiform_results())
    no_exc(at)
    body = texts(at)
    assert "Выберите форму теста" in body
    assert "нельзя утвердить" in body
    assert "Обыкновенные дроби" not in "".join(x.value for x in at.markdown if "-badge[Трудная]" in x.value)


def test_overview_no_transfers_is_honest_empty_state():
    at = open_demo()
    at.sidebar.number_input(key="p_donor_from").set_value(90.0).run()
    assert metrics(at)["Часов к переносу"] == "0 ч"
    assert any("Допустимых переносов нет" in e.value for e in at.info)


def test_small_sample_overview_does_not_call_low_n_topics_hard():
    at = click(go(new_app(), "Данные"), "Демо с малой выборкой")
    at = go(at, "Обзор")
    no_exc(at)
    assert "Мало данных" in texts(at) or "Тем, достигших порогов" in texts(at)
    core = analyze(generate_results(small=True), topics_frame(), Params()).topic_table
    from kartatem.metrics import STATUS_HARD, STATUS_LOW
    hard, low = (core.status == STATUS_HARD).sum(), (core.status == STATUS_LOW).sum()
    assert metrics(at)["Трудных тем"] == f"{hard} из {len(core) - low}"   # малые темы не считаются трудными
    assert not set(core[core.status == STATUS_LOW].topic_id) & set(core[core.status == STATUS_HARD].topic_id)


def test_textbook_colors_are_constant_across_pairs():
    at = go(app_with(three_textbooks()), "Учебники")
    seen = {}
    for label in ("«Учебник А» и «Учебник Б»", "«Учебник А» и «Учебник В»", "«Учебник Б» и «Учебник В»"):
        at = at.selectbox(key="pair_sel").select(label).run()
        for c in charts(at):
            if c["layout"].get("barmode") != "group":
                continue
            for tr in c["data"]:
                if tr["name"] in ("Учебник А", "Учебник Б", "Учебник В"):
                    seen.setdefault(tr["name"], set()).add(tr["marker"]["color"])
    assert all(len(v) == 1 for v in seen.values()) and len(seen) == 3
    pal = textbook_colors(["Учебник А", "Учебник Б", "Учебник В"])
    assert {k: next(iter(v)) for k, v in seen.items()} == {k: pal[k][0] for k in pal}


def test_chart_titles_are_short_and_single_topic_view_matches_core():
    at = go(open_demo(), "Учебники")
    for c in charts(at):
        assert len(c["layout"]["title"]["text"]) <= 40
    core = analyze(generate_results(), topics_frame(), Params()).comparison.table
    at.selectbox(key="topic_sel").select("T2").run()
    r = core[core.topic_id == "T2"].iloc[0]
    assert "Входные уровни пар различаются на 10.03 п.п." in texts(at)
    assert "порога предупреждения 10.00" in texts(at)
    assert abs(r.pre_gap) > 10


def test_hours_page_core_elements():
    at = go(open_demo(), "Сценарии")
    body = texts(at)
    assert "Сценарий распределения, не прогноз роста баллов" in body
    assert "**Предложение алгоритма, не утверждено**" in texts(at)
    at = click(at, "Утвердить черновик")
    assert "**Утверждён**" in texts(go(at, "Сценарии"))


def test_fmt_gap_vs_threshold_shows_trigger():
    assert fmt_gap_vs_threshold(-10.026, 10.0) == ("10.03", "10.00")
    assert fmt_gap_vs_threshold(12.34, 10.0) == ("12.3", "10")
    assert fmt_gap_vs_threshold(10.0001, 10.0) == ("10.0001", "10.0000")


def test_hours_moved_counts_positive_deltas_only():
    core = analyze(generate_results(), topics_frame(), Params())
    plan = core.plan
    assert plan.hours_moved() == 3 == -sum(d for d in plan.table["delta"] if d < 0)
    assert plan.flows() == [("T1", "T3", 2), ("T5", "T6", 1)]
    assert core.textbooks == ["Учебник А", "Учебник Б"]
