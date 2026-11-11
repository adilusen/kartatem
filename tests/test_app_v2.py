"""Интерфейсные регрессии (AppTest): постоянный селектор форм, пара учебников, подписи тем."""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from kartatem.config import Params
from kartatem.demo import generate_results, topics_frame
from kartatem.labels import topic_labels
from kartatem.pipeline import analyze

APP = str(Path(__file__).resolve().parent.parent / "app.py")
P = Params()


# ---------------------------------------------------------------- помощники
def make_source(results: pd.DataFrame, topics: pd.DataFrame, label: str = "test") -> dict:
    sig = hashlib.sha1((results.to_csv(index=False) + topics.to_csv(index=False)).encode()).hexdigest()[:12]
    return {"kind": "upload", "label": label, "synthetic": False, "results": results, "topics": topics, "sig": sig}


def app_with(results, topics=None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["source"] = make_source(results, topics if topics is not None else topics_frame())
    return at.run()


def go(at, section):
    return at.sidebar.radio(key="nav").set_value(section).run()


def click(at, label):
    return next(b for b in at.button if b.label == label).click().run()


def charts(at) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def topic_view(at) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if "Использовано в расчёте" in d.value.columns)


def no_exc(at):
    assert not at.exception, [e.value for e in at.exception]


def multiform_results() -> pd.DataFrame:
    """Демо, где в теме T1 две формы: FORM2 (ученики S01–S10 каждого класса) и M6-T1-F1 (остальные)."""
    df = generate_results()
    mask = (df["topic_id"] == "T1") & df["student_id"].isin([f"S{i:02d}" for i in range(1, 11)])
    df.loc[mask, "test_id"] = "FORM2"
    return df


def three_textbooks() -> pd.DataFrame:
    """Демо + третий учебник «В»: копия учебника А с другими классами и нулевым приростом (post = pre)."""
    df = generate_results()
    a = df[df["textbook"] == "Учебник А"].copy()
    a["textbook"] = "Учебник В"
    a["class_id"] = a["class_id"].str.replace("K", "V", regex=False)
    pre = a[a["stage"] == "pre"].set_index(["class_id", "student_id", "topic_id"])["score"]
    is_post = a["stage"] == "post"
    keys = list(zip(a.loc[is_post, "class_id"], a.loc[is_post, "student_id"], a.loc[is_post, "topic_id"]))
    a.loc[is_post, "score"] = [pre.get(k, "") for k in keys]
    return pd.concat([df, a], ignore_index=True)


# ================================================================ 2. постоянный селектор форм
def test_selector_is_persistent_after_choice_and_across_screens():
    at = app_with(multiform_results())
    sel = at.sidebar.selectbox(key="tc_T1")
    assert sel.options == ["— выберите —", "FORM2", "M6-T1-F1"] and sel.value == "— выберите —"
    assert [s.key for s in at.sidebar.selectbox if s.key and s.key.startswith("tc_")] == ["tc_T1"]  # только темы с несколькими формами

    at = at.sidebar.selectbox(key="tc_T1").select("M6-T1-F1").run()
    assert at.session_state["test_choice"] == {"T1": "M6-T1-F1"}
    # ошибка №2: раньше после выбора селектор исчезал
    assert at.sidebar.selectbox(key="tc_T1").value == "M6-T1-F1"
    for section in ("Темы", "Учебники", "Сценарии", "Решение", "Данные"):
        at = go(at, section)
        no_exc(at)
        assert at.sidebar.selectbox(key="tc_T1").value == "M6-T1-F1", section
    # смена F1 → F2 без перезагрузки файла
    at = at.sidebar.selectbox(key="tc_T1").select("FORM2").run()
    assert at.session_state["test_choice"] == {"T1": "FORM2"}
    for section in ("Темы", "Сценарии"):
        at = go(at, section)
        no_exc(at)
        assert at.sidebar.selectbox(key="tc_T1").value == "FORM2"


def test_switching_form_updates_metrics_on_topics_screen():
    at = go(app_with(multiform_results()), "Темы")
    v = topic_view(at).set_index("Тема")
    assert v.loc["Обыкновенные дроби", "Статус"] == "Выберите форму теста"
    assert v.loc["Обыкновенные дроби", "Средний post, %"] == "—"                 # нет объединённого среднего
    assert v.loc["Обыкновенные дроби", "Использовано в расчёте"] == 0
    assert v.loc["Проценты", "Учеников (post)"] > 100                           # остальные темы посчитаны
    assert any("Выберите форму теста" in w.value for w in at.warning)

    at = at.sidebar.selectbox(key="tc_T1").select("FORM2").run()
    f2 = topic_view(at).set_index("Тема").loc["Обыкновенные дроби"]
    at = at.sidebar.selectbox(key="tc_T1").select("M6-T1-F1").run()
    f1 = topic_view(at).set_index("Тема").loc["Обыкновенные дроби"]
    assert f2["Форма теста"] == "FORM2" and f1["Форма теста"] == "M6-T1-F1"
    assert f2["Использовано в расчёте"] != f1["Использовано в расчёте"]
    assert f2["Записей post в файле"] == f1["Записей post в файле"] > f1["Использовано в расчёте"]
    assert f2["Средний post, %"] != f1["Средний post, %"]


def test_form_change_resets_draft_and_applied_plan_with_notice():
    at = go(app_with(multiform_results()), "Сценарии")
    at = click(at, "Утвердить черновик")                      # форма не выбрана → применить нельзя
    assert at.session_state["applied"] is None
    assert any("Не выбрана форма теста" in e.value for e in at.error)

    at = at.sidebar.selectbox(key="tc_T1").select("M6-T1-F1").run()
    at = click(at, "Утвердить черновик")
    assert at.session_state["applied"] is not None
    at = at.sidebar.selectbox(key="tc_T1").select("FORM2").run()
    assert at.session_state["applied"] is None and at.session_state["apply_msg"] is None
    assert any("Форма теста изменена" in i.value and "другой набор измерений" in i.value for i in at.info)
    assert at.session_state["draft"] == at.session_state["_proposed"]


def test_unresolved_form_keeps_hours_and_does_not_block_other_topics():
    at = go(app_with(multiform_results()), "Сценарии")
    no_exc(at)
    inputs = {n.key.removeprefix("hrs_"): n.value for n in at.number_input if n.key and n.key.startswith("hrs_")}
    assert inputs["T1"] == 5                                   # часы нерешённой темы не менялись
    assert inputs["T3"] > 5                                    # «Проценты» (трудная) по-прежнему получает часы


def test_new_source_does_not_inherit_old_form_choice():
    at = app_with(multiform_results())
    at = at.sidebar.selectbox(key="tc_T1").select("FORM2").run()
    assert at.session_state["test_choice"] == {"T1": "FORM2"}
    at = go(at, "Данные")
    at = click(at, "Открыть демо")                             # обычное демо: форма одна
    assert at.session_state["test_choice"] == {}
    assert not [s for s in at.sidebar.selectbox if s.key and s.key.startswith("tc_")]


# ================================================================ 3. три и более учебника
EXPECT_SERIES = {
    "«Учебник А» и «Учебник Б»": ("Учебник А", "Учебник Б"),
    "«Учебник А» и «Учебник В»": ("Учебник А", "Учебник В"),
    "«Учебник Б» и «Учебник В»": ("Учебник Б", "Учебник В"),
}


def gain_chart(at) -> dict:
    cs = [c for c in charts(at) if all(t.get("type") == "bar" for t in c["data"]) and
          "прирост" in json.dumps(c["layout"].get("title", {}), ensure_ascii=False).lower()]
    assert len(cs) == 1
    return cs[0]


def test_gain_chart_follows_selected_pair_with_correct_series():
    data = three_textbooks()
    core = analyze(data, topics_frame(), P).comparison.table
    assert len(core) == 18                                     # 6 тем × 3 пары — как в отчёте о баге
    at = go(app_with(data), "Учебники")
    no_exc(at)
    sel = at.selectbox(key="pair_sel")
    assert sel.options == list(EXPECT_SERIES)

    seen_gain = {}
    for label, (a_name, b_name) in EXPECT_SERIES.items():
        at = at.selectbox(key="pair_sel").select(label).run()
        no_exc(at)
        fig = gain_chart(at)
        assert [t["name"] for t in fig["data"]] == [a_name, b_name]               # подписи серий
        labels = list(topic_labels(topics_frame()).values())
        for trace in fig["data"]:
            assert list(trace["x"]) == labels and len(set(trace["x"])) == 6        # каждая тема ровно один раз
        # значения серий — это приросты именно этих учебников из ядра, а не смесь
        pair_rows = core[(core.textbook_a == a_name) & (core.textbook_b == b_name)].sort_values("topic_id")
        for trace, col in zip(fig["data"], ("gain_a", "gain_b")):
            assert [round(v, 6) for v in trace["y"]] == [round(v, 6) for v in pair_rows[col]]
        assert fig["layout"]["title"]["text"] == "Средний прирост, п.п."      # короткий заголовок; пара — в легенде
        seen_gain[label] = [list(t["y"]) for t in fig["data"]]
        # таблицы и подписи относятся к выбранной паре
        tables = [d.value for d in at.dataframe]
        books_in_tables = set().union(*[set(t["Учебник"]) for t in tables if "Учебник" in t.columns])
        assert books_in_tables == {a_name, b_name}
        pair_col = next(t for t in tables if "Пара" in t.columns)["Пара"]
        assert set(pair_col) == {label} and len(pair_col) == 6
    # третий учебник действительно отличается (нулевой прирост): серия «В» ≪ серии А
    assert max(seen_gain["«Учебник А» и «Учебник В»"][1]) < 5 < min(seen_gain["«Учебник А» и «Учебник В»"][0])


def test_pair_selection_survives_navigation_and_new_source_resets_it():
    at = go(app_with(three_textbooks()), "Учебники")
    at = at.selectbox(key="pair_sel").select("«Учебник Б» и «Учебник В»").run()
    for section in ("Решение", "Сценарии", "Учебники"):
        at = go(at, section)
        no_exc(at)
    assert at.selectbox(key="pair_sel").value == "«Учебник Б» и «Учебник В»"
    at = go(at, "Данные")
    at = click(at, "Открыть демо")
    at = go(at, "Учебники")
    assert at.selectbox(key="pair_sel").value == "«Учебник А» и «Учебник Б»"
    assert at.selectbox(key="pair_sel").options == ["«Учебник А» и «Учебник Б»"]


def test_single_textbook_shows_state_without_crash():
    df = generate_results()
    df = df[df["textbook"] == "Учебник А"]
    at = go(app_with(df), "Учебники")
    no_exc(at)
    assert any("минимум два учебника" in i.value for i in at.info)
    assert not at.get("plotly_chart")
    for section in ("Темы", "Сценарии", "Решение"):
        no_exc(go(at, section))


# ================================================================ 6. одинаковые названия тем
def dup_name_topics() -> pd.DataFrame:
    t = topics_frame()
    t.loc[t["topic_id"] == "T2", "name"] = t.loc[t["topic_id"] == "T1", "name"].iloc[0]
    return t


def test_duplicate_topic_names_keep_all_topics_on_every_chart():
    topics = dup_name_topics()
    assert topics["name"].nunique() == 5 and len(topics) == 6
    expected = ["Обыкновенные дроби (T1)", "Обыкновенные дроби (T2)", "Проценты", "Уравнения",
                "Пропорции", "Геометрия: площади"]
    assert list(topic_labels(topics).values()) == expected
    core = analyze(generate_results(), topics, P)
    mean = core.topic_table.set_index("topic_id")["mean_post"]

    at = go(app_with(generate_results(), topics), "Темы")
    no_exc(at)
    heat, bar = charts(at)
    assert heat["data"][0]["type"] == "heatmap" and list(heat["data"][0]["y"]) == expected   # было 5 строк
    assert len(heat["data"][0]["z"]) == 6
    assert all(v is not None for row in heat["data"][0]["z"] for v in row)
    assert list(bar["data"][0]["x"]) == expected and len(set(bar["data"][0]["x"])) == 6
    t1_cells = heat["data"][0]["z"][0]
    t2_cells = heat["data"][0]["z"][1]
    assert t1_cells != t2_cells                                                 # у T1 и T2 свои числа
    tb = core.topic_tb
    assert [round(v, 6) for v in t1_cells] == [round(v, 6) for v in tb[tb.topic_id == "T1"].sort_values("textbook")["mean_post"]]
    assert [round(v, 6) for v in t2_cells] == [round(v, 6) for v in tb[tb.topic_id == "T2"].sort_values("textbook")["mean_post"]]
    assert not math.isclose(mean["T1"], mean["T2"])
    assert len(topic_view(at)) == 6

    at = go(at, "Учебники")
    assert list(gain_chart(at)["data"][0]["x"]) == expected
    at = go(at, "Сценарии")
    hours_fig = charts(at)[0]
    assert all(list(t["x"]) == expected for t in hours_fig["data"])
    no_exc(go(at, "Решение"))
