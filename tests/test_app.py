"""Сценарные проверки интерфейса через Streamlit AppTest (без браузера)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SECTIONS = ["Обзор", "Темы", "Учебники", "Сценарии", "Решение", "Данные"]
MARK = "Синтетические демо-данные"


def new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=90).run()


def click(at: AppTest, label: str) -> AppTest:
    btn = next(b for b in at.button if b.label == label)
    return btn.click().run()


def go(at: AppTest, section: str) -> AppTest:
    return at.sidebar.radio(key="nav").set_value(section).run()


def texts(at: AppTest) -> str:
    parts = []
    for coll in (at.header, at.subheader, at.markdown, at.warning, at.error, at.success, at.info, at.caption):
        parts += [e.value for e in coll]
    return "\n".join(parts)


def no_exceptions(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


@pytest.fixture()
def demo_app() -> AppTest:
    at = click(new_app(), "Открыть демо")
    no_exceptions(at)
    return at


def test_app_imports_and_opens_without_data():
    at = new_app()
    no_exceptions(at)
    assert at.sidebar.radio(key="nav").options == SECTIONS
    assert at.session_state["nav"] == "Обзор"                 # Обзор — экран по умолчанию
    assert "Өлшем" in texts(at) or any("Өлш" in m.value for m in at.markdown)
    assert any(b.label == "Открыть демо" for b in at.button)
    assert "Найти слабые темы, проверить основания" in texts(at)
    assert not at.metric                                      # без данных нет выдуманных показателей
    at = go(at, "Сценарии")                       # без данных остальные экраны мягко просят открыть демо
    no_exceptions(at)
    assert any("Сначала откройте демо" in e.value for e in at.info)


def test_demo_opens_and_every_section_renders(demo_app):
    assert demo_app.session_state["source"]["synthetic"] is True
    assert MARK in texts(demo_app) and demo_app.metric          # Обзор сразу заполнен
    for sec in SECTIONS:
        at = go(demo_app, sec)
        no_exceptions(at)
        assert at.header[0].value == sec
        assert MARK in texts(at), f"нет пометки синтетики на экране {sec}"


def test_topics_screen_shows_hard_topics(demo_app):
    at = go(demo_app, "Темы")
    body = texts(at)
    assert body.count("-badge[Трудная]") == 3
    assert "не официальная" in body


def test_textbooks_screen_has_limits_next_to_conclusion(demo_app):
    at = go(demo_app, "Учебники")
    body = texts(at)
    assert "не корректируется" in body and "не доказывает" in body
    assert "Общего рейтинга учебников нет" in body
    assert "лучше" not in body.lower().replace("не лучше", "")


def test_small_demo_blocks_comparison_but_not_topics():
    at = click(go(new_app(), "Данные"), "Демо с малой выборкой")
    no_exceptions(at)
    at = go(at, "Учебники")
    no_exceptions(at)
    body = texts(at)
    assert "недостаточно для сравнения" in body
    at = go(at, "Темы")
    assert "Мало данных" in texts(at)
    assert not at.exception


def test_bad_csv_shows_errors_and_blocks_other_screens():
    at = click(go(new_app(), "Данные"), "Пример плохого CSV")
    no_exceptions(at)
    at = go(at, "Обзор")
    assert any("Анализ заблокирован" in e.value for e in at.error)       # Обзор: состояние ошибки, без метрик
    assert not at.metric
    at = go(at, "Данные")
    assert any("student_id" in e.value for e in at.error)
    assert any("Конфликтующие дубли" in e.value for e in at.error)
    for sec in ("Темы", "Учебники", "Сценарии", "Решение"):
        at = go(at, sec)
        no_exceptions(at)
        assert any("Анализ заблокирован" in e.value for e in at.error)


def hours_inputs(at: AppTest) -> dict:
    return {ni.key.removeprefix("hrs_"): ni for ni in at.number_input if ni.key and ni.key.startswith("hrs_")}


def test_hours_default_draft_matches_proposal_and_budget(demo_app):
    at = go(demo_app, "Сценарии")
    no_exceptions(at)
    inputs = hours_inputs(at)
    assert [inputs[t].value for t in ("T1", "T2", "T3", "T4", "T5", "T6")] == [3, 5, 7, 6, 3, 6]
    assert sum(i.value for i in inputs.values()) == 30
    assert any("Черновик корректен" in e.value for e in at.success)


def test_invalid_manual_draft_is_blocked_and_last_valid_plan_is_kept(demo_app):
    at = go(demo_app, "Сценарии")
    at = click(at, "Утвердить черновик")                       # применяем корректный план
    assert at.session_state["applied"] == {"T1": 3, "T2": 5, "T3": 7, "T4": 6, "T5": 3, "T6": 6}
    good = dict(at.session_state["applied"])

    hours_inputs(at)["T2"].set_value(6).run()                  # сумма 31 — невалидно
    assert any("Сумма часов 31" in e.value for e in at.warning)
    at = click(at, "Утвердить черновик")
    assert at.session_state["applied"] == good                 # утверждённый план не изменился
    assert any("План не принят" in e.value for e in at.error)

    hours_inputs(at)["T2"].set_value(5).run()                  # вернули — снова корректно
    hours_inputs(at)["T1"].set_value(2).run()                  # ниже границы (3) и сумма 29
    errs = " ".join(e.value for e in at.warning)
    assert "вне допустимого диапазона" in errs and "Сумма часов 29" in errs
    at = click(at, "Утвердить черновик")
    assert at.session_state["applied"] == good


def test_valid_manual_edit_can_be_applied(demo_app):
    at = go(demo_app, "Сценарии")
    inputs = hours_inputs(at)
    inputs["T4"].set_value(7).run()                            # +1 для «Уравнений» (граница 4–8)
    hours_inputs(at)["T5"].set_value(2).run()                  # пропорции: нижняя граница 3 → нарушение
    assert any("вне допустимого диапазона" in e.value for e in at.warning)
    hours_inputs(at)["T5"].set_value(3).run()
    hours_inputs(at)["T6"].set_value(5).run()                  # −1 у «Геометрии» компенсирует +1
    assert any("Черновик корректен" in e.value for e in at.success)
    at = click(at, "Утвердить черновик")
    assert at.session_state["applied"]["T4"] == 7 and at.session_state["applied"]["T6"] == 5
    assert sum(at.session_state["applied"].values()) == 30


def test_draft_and_applied_plan_survive_navigation_and_reruns(demo_app):
    at = go(demo_app, "Сценарии")
    hours_inputs(at)["T4"].set_value(7).run()
    hours_inputs(at)["T6"].set_value(5).run()
    at = click(at, "Утвердить черновик")
    applied = dict(at.session_state["applied"])
    for sec in ("Темы", "Учебники", "Решение", "Сценарии"):          # уход с экрана и возврат
        at = go(at, sec)
        no_exceptions(at)
    inputs = hours_inputs(at)
    assert inputs["T4"].value == 7 and inputs["T6"].value == 5
    assert at.session_state["applied"] == applied
    at.run()                                                   # обычная перерисовка
    assert at.session_state["applied"] == applied and hours_inputs(at)["T4"].value == 7


def test_reset_draft_button_restores_proposal(demo_app):
    at = go(demo_app, "Сценарии")
    hours_inputs(at)["T4"].set_value(9).run()
    at = click(at, "Сбросить черновик к предложению алгоритма")
    assert [hours_inputs(at)[t].value for t in ("T1", "T2", "T3", "T4", "T5", "T6")] == [3, 5, 7, 6, 3, 6]
    assert any("Черновик корректен" in e.value for e in at.success)


def test_changing_source_resets_draft_and_applied_plan(demo_app):
    at = go(demo_app, "Сценарии")
    hours_inputs(at)["T4"].set_value(7).run()
    hours_inputs(at)["T6"].set_value(5).run()
    at = click(at, "Утвердить черновик")
    assert at.session_state["applied"] is not None
    at = go(at, "Данные")
    at = click(at, "Демо с малой выборкой")
    assert at.session_state["applied"] is None
    at = go(at, "Сценарии")
    no_exceptions(at)
    assert hours_inputs(at)["T4"].value == at.session_state["_proposed"]["T4"]
    assert at.session_state["draft"] == at.session_state["_proposed"]


def test_changing_parameters_resets_plan_with_notice(demo_app):
    at = go(demo_app, "Сценарии")
    at = click(at, "Утвердить черновик")
    assert at.session_state["applied"] is not None
    at.sidebar.number_input(key="p_donor_from").set_value(90.0).run()      # меняем порог доноров
    assert at.session_state["applied"] is None
    at = go(at, "Сценарии")
    assert any("Параметры изменены" in e.value for e in at.info)
    assert any("Допустимых переносов нет" in e.value for e in at.success)  # нет доноров — нормальный результат


def test_report_screen_and_export_flags(demo_app):
    at = go(demo_app, "Решение")
    no_exceptions(at)
    assert "не прогноз" in texts(at)
    assert any("не утверждено" in str(df.value.to_dict()) for df in at.dataframe)
