"""v4 UI: фиксация, сценарии, сравнение, утверждение, сброс при смене контекста (AppTest, без пикселей)."""
from tests.test_app import click, go, new_app, texts
from tests.test_app_v2 import app_with, multiform_results


def scen_app():
    at = go(click(new_app(), "Открыть демо"), "Сценарии")
    assert not at.exception
    return at


def save(at, name):
    at.text_input(key="sc_name").input(name).run()
    return click(at, "Сохранить сценарий")


def test_lock_t1_recalculates_draft_and_disables_input():
    at = scen_app()
    at = at.checkbox(key="lock_T1").check().run()
    assert at.session_state["locked"] == ["T1"]
    assert at.session_state["draft"] == {"T1": 5, "T2": 5, "T3": 6, "T4": 6, "T5": 3, "T6": 5}
    assert at.number_input(key="hrs_T1").disabled
    assert "Черновик корректен" in texts(at)


def test_two_scenarios_survive_navigation_compare_and_approve():
    at = save(scen_app(), "Алгоритм")
    at = at.checkbox(key="lock_T1").check().run()
    at = save(at, "Без дробей")
    assert [s.name for s in at.session_state["scenarios"]] == ["Алгоритм", "Без дробей"]
    for sec in ("Обзор", "Темы", "Учебники", "Решение", "Данные", "Сценарии"):
        at = go(at, sec)
        assert not at.exception
    assert [s.name for s in at.session_state["scenarios"]] == ["Алгоритм", "Без дробей"]
    cmp = next(df.value for df in at.dataframe if "Разница Б − А, ч" in df.value.columns)
    assert list(cmp["Разница Б − А, ч"]) == [2, 0, -1, 0, 0, -1]
    assert "прогноз баллов не строится" in texts(at)
    at = at.button(key="sc_ok_1").click().run()
    assert at.session_state["applied"] == {"T1": 5, "T2": 5, "T3": 6, "T4": 6, "T5": 3, "T6": 5}
    assert at.session_state["applied_name"] == "Без дробей" and at.session_state["applied_locked"] == ["T1"]
    assert "**Утверждён**" in texts(at)
    at = go(at, "Решение")
    assert "План «Без дробей»" in texts(at)
    shown = next(df.value for df in at.dataframe if "Утверждено, ч" in df.value.columns)
    assert list(shown["Утверждено, ч"]) == [5, 5, 6, 6, 3, 5]


def test_context_change_invalidates_approval_and_marks_scenarios():
    at = save(scen_app(), "Алгоритм")
    at = click(at, "Утвердить черновик")
    assert at.session_state["applied"] is not None
    at.sidebar.number_input(key="p_donor_from").set_value(76.0).run()
    at = go(at, "Сценарии")
    assert at.session_state["applied"] is None
    assert "Другой контекст данных" in texts(at)
    assert at.button(key="sc_ok_0").disabled
    at = at.button(key="sc_recalc_0").click().run()
    sc = at.session_state["scenarios"][0]
    assert sc.context == at.session_state["_ctx"] and sum(sc.hours.values()) == 30
    assert not at.button(key="sc_ok_0").disabled


def test_scenario_cannot_be_approved_while_form_unresolved():
    at = go(app_with(multiform_results()), "Сценарии")
    at = save(at, "Вариант")
    assert at.button(key="sc_ok_0").disabled
    at = click(at, "Утвердить черновик")
    assert at.session_state["applied"] is None
    assert any("Не выбрана форма теста" in e.value for e in at.error)


def test_invalid_draft_cannot_be_saved_and_fourth_scenario_rejected():
    at = scen_app()
    at.number_input(key="hrs_T2").set_value(6).run()
    at = save(at, "Плохой")
    assert at.session_state["scenarios"] == [] and any("некорректен" in e.value for e in at.error)
    at.number_input(key="hrs_T2").set_value(5).run()
    for n in ("A", "B", "C"):
        at = save(at, n)
    at = save(at, "D")
    assert len(at.session_state["scenarios"]) == 3 and any("не больше 3" in e.value for e in at.error)


def test_overview_focus_and_check_reasons_button():
    at = click(new_app(), "Открыть демо")
    body = texts(at)
    assert "Проценты" in body and "47.8%" in body and "n = **164**" in body and "**+2 ч** из «Обыкновенные дроби»" in body
    at = click(at, "Проверить основания")
    assert at.session_state["nav"] == "Темы" and at.session_state["topic_focus"] == "T3"
    assert not at.exception
