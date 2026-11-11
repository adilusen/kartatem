"""Өлшем — интерфейс Streamlit (ранее «КартаТем»). Расчёты живут в пакете kartatem/, здесь только показ.

Запуск:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kartatem import demo
from kartatem.compare import ST_OK, STATUS_LABELS
from kartatem.config import PARAM_UI, Params
from kartatem.decision import build_decision
from kartatem.hours import validate_plan
from kartatem.labels import topic_labels
from kartatem.metrics import STATUS_CHOOSE, STATUS_HARD, STATUS_LOW, STATUS_OK
from kartatem.pipeline import analyze
from kartatem.plan_io import parse_plan, restore_plan
from kartatem.scenarios import (MAX_SCENARIOS, Scenario, add_scenario, compare_scenarios, context_id,
                                context_parts, scenario_problems)
from kartatem.theme import ACCENT, BORDER, FONT, MUTED, SURFACE, TEXT, textbook_colors
from kartatem.validation import read_csv_bytes

BRAND = "Өлшем"
TAGLINE = "Результаты обучения и план учебных часов"
st.set_page_config(page_title=BRAND, layout="wide")

SECTIONS = ["Обзор", "Темы", "Учебники", "Сценарии", "Решение", "Данные"]
PLACEHOLDER = "— выберите —"
DRAFT_NAME = "Черновик"
ss = st.session_state

# Небольшой набор стилей: логотип, тонкие рамки метрик и контейнеров с key="card_*".
# Ничего не скрывает элементы Streamlit и не зависит от их внутренней вёрстки.
CSS = f"""
<style>
.olshem-logo {{ font-size:1.6rem; font-weight:750; letter-spacing:.01em; color:{TEXT}; line-height:1.1; }}
.olshem-logo span {{ color:{ACCENT}; }}
[data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:6px; padding:10px 14px; }}
[class*="st-key-card"] {{ background:{SURFACE}; border-radius:6px; }}
</style>
"""


# ============================================================================
# Состояние и callbacks
# ============================================================================
def _make_source(kind: str, label: str, synthetic: bool, results: pd.DataFrame, topics: pd.DataFrame) -> dict:
    digest = hashlib.sha1((results.to_csv(index=False) + "||" + topics.to_csv(index=False)).encode("utf-8"))
    return {"kind": kind, "label": label, "synthetic": synthetic,
            "results": results, "topics": topics, "sig": digest.hexdigest()[:12]}


def _clear_keys(prefixes: tuple, exact: tuple = ()) -> None:
    for k in [k for k in ss.keys() if k.startswith(prefixes) or k in exact]:
        del ss[k]


def _set_source(src: dict) -> None:
    """Новый источник сбрасывает формы, пару, фиксации; черновик и утверждение сбросит sync_state().
    Сохранённые сценарии остаются, но помечаются как созданные в другом контексте."""
    ss["source"] = src
    ss["test_choice"] = {}
    ss["pair"] = None
    ss["cmp_topic"] = None
    ss["locked"] = []
    ss["state_ctx"] = None
    ss["state_parts"] = None
    ss["upload_error"] = None
    _clear_keys(("tc_", "lock_"), ("pair_sel", "topic_sel", "topic_focus"))


def cb_open_demo() -> None:
    _set_source(_make_source("demo", f"генератор, seed={demo.SEED}", True,
                             demo.generate_results(), demo.topics_frame()))


def cb_open_demo_small() -> None:
    _set_source(_make_source("demo_small", f"генератор, малая выборка, seed={demo.SEED}", True,
                             demo.generate_results(small=True), demo.topics_frame()))


def cb_open_bad() -> None:
    _set_source(_make_source("bad_example", "намеренно испорченный пример для проверки ошибок", True,
                             demo.bad_results_frame(), demo.topics_frame()))


def cb_use_uploads() -> None:
    f_res, f_top = ss.get("up_results"), ss.get("up_topics")
    if f_res is None or f_top is None:
        ss["upload_error"] = "Загрузите оба файла: results.csv и topics.csv."
        return
    results, err1 = read_csv_bytes(f_res.getvalue())
    topics, err2 = read_csv_bytes(f_top.getvalue())
    errs = [f"results.csv: {err1}"] * bool(err1) + [f"topics.csv: {err2}"] * bool(err2)
    if errs:
        ss["upload_error"] = " ".join(errs)
        return
    _set_source(_make_source("upload", f"{f_res.name} + {f_top.name}", False, results, topics))


def cb_goto(section: str, focus: str | None = None) -> None:
    """Переход между разделами. Источник, формы, фиксации, сценарии и утверждённый план не трогаем."""
    ss["nav"] = section
    if focus:
        ss["topic_focus"] = focus


def cb_choose_form(topic_id: str) -> None:
    value = ss.get(f"tc_{topic_id}")
    choice = dict(ss.get("test_choice", {}))
    if value and value != PLACEHOLDER:
        choice[topic_id] = value
    else:
        choice.pop(topic_id, None)
    ss["test_choice"] = choice


def cb_pair() -> None:
    ss["pair"] = ss.get("pair_sel")


def cb_cmp_topic() -> None:
    ss["cmp_topic"] = ss.get("topic_sel")


def _set_draft(hours: dict) -> None:
    ss["draft"] = {k: int(v) for k, v in hours.items()}
    for tid, v in ss["draft"].items():
        ss[f"hrs_{tid}"] = int(v)


def _set_locked(locked) -> None:
    ss["locked"] = sorted(locked)
    _clear_keys(("lock_",))


def cb_draft_changed(topic_id: str) -> None:
    ss["draft"][topic_id] = ss[f"hrs_{topic_id}"]
    ss["apply_msg"] = None


def cb_lock(topic_id: str) -> None:
    """«Не изменять тему»: фиксирует ИСХОДНЫЕ часы. Предложение пересчитывается, черновик — к нему."""
    locked = set(ss.get("locked", []))
    if ss.get(f"lock_{topic_id}"):
        locked.add(topic_id)
    else:
        locked.discard(topic_id)
    ss["locked"] = sorted(locked)
    ss["_reseed_draft"] = True
    ss["apply_msg"] = None


def cb_reset_draft() -> None:
    _set_draft(ss.get("_proposed", {}))
    ss["apply_msg"] = None


def _approve(name: str, hours: dict, locked: list) -> None:
    topics = ss.get("_topics")
    if topics is None:
        return
    if ss.get("_unresolved"):        # нерешённая форма теста не должна превращаться в утверждённое решение
        ss["apply_msg"] = ("error", ["Не выбрана форма теста для тем: " + ", ".join(ss["_unresolved"])
                                     + ". Выберите форму в боковой панели, затем утверждайте план."])
        return
    errors = validate_plan(topics, hours, locked)
    if errors:       # невалидный план не принимается; сохраняется последний валидный
        ss["apply_msg"] = ("error", errors)
        return
    ss["applied"] = {k: int(v) for k, v in hours.items()}
    ss["applied_name"] = name
    ss["applied_locked"] = sorted(locked)
    ss["apply_msg"] = ("success", [f"Утверждён план «{name}» (только в этой сессии)."])
    ss["reset_note"] = None


def cb_apply() -> None:
    _approve(DRAFT_NAME, dict(ss.get("draft", {})), list(ss.get("locked", [])))


def cb_save_scenario() -> None:
    name = ss.get("sc_name", "")
    sc = Scenario(name, dict(ss.get("draft", {})), list(ss.get("locked", [])), ss.get("_ctx", ""))
    errs = validate_plan(ss["_topics"], sc.hours, sc.locked)
    if errs:
        ss["sc_msg"] = ("error", ["Сохранить нельзя: черновик некорректен."] + errs)
        return
    new, err = add_scenario(ss.get("scenarios", []), sc)
    if err:
        ss["sc_msg"] = ("error", [err])
        return
    ss["scenarios"] = new
    ss["sc_msg"] = ("success", [f"Сценарий «{sc.name.strip()}» сохранён."])
    ss["sc_name"] = ""


def _scenario(i: int) -> Scenario | None:
    scs = ss.get("scenarios", [])
    return scs[i] if 0 <= i < len(scs) else None


def cb_load_scenario(i: int) -> None:
    sc = _scenario(i)
    if sc is None:
        return
    _set_locked(sc.locked)
    _set_draft(sc.hours)
    ss["sc_msg"] = ("success", [f"Сценарий «{sc.name}» загружен в черновик."])


def cb_recalc_scenario(i: int) -> None:
    """Сценарий другого контекста: заново считаем предложение с ТЕМИ ЖЕ фиксациями на текущих данных."""
    sc, a = _scenario(i), ss.get("_analysis")
    if sc is None or a is None:
        return
    locked = [t for t in sc.locked if t in set(a.topics["topic_id"])]
    hours = a.plan_for(locked).proposed()
    scs = list(ss["scenarios"])
    scs[i] = Scenario(sc.name, hours, locked, ss["_ctx"], "algorithm")
    ss["scenarios"] = scs
    ss["sc_msg"] = ("success", [f"Сценарий «{sc.name}» пересчитан на текущих данных: ручные правки в нём "
                                "заменены предложением алгоритма с теми же фиксациями."])


def cb_approve_scenario(i: int) -> None:
    sc = _scenario(i)
    if sc is None:
        return
    probs = scenario_problems(sc, ss["_topics"], ss.get("_ctx", ""))
    if probs and not ss.get("_unresolved"):
        ss["apply_msg"] = ("error", probs)
        return
    _approve(sc.name, sc.hours, sc.locked)
    if ss.get("applied") == sc.hours and ss.get("applied_name") == sc.name:
        _set_locked(sc.locked)
        _set_draft(sc.hours)


def cb_delete_scenario(i: int) -> None:
    sc = _scenario(i)
    if sc is not None:
        ss["scenarios"] = [x for j, x in enumerate(ss["scenarios"]) if j != i]
        ss["sc_msg"] = ("success", [f"Сценарий «{sc.name}» удалён."])
        _clear_keys((), ("cmp_a", "cmp_b"))


def cb_restore_plan() -> None:
    """Восстановление из JSON: проверка → настройки → сценарии → черновик. Утверждение не восстанавливается."""
    f, src = ss.get("up_plan"), ss.get("source")
    if f is None or src is None:
        ss["restore_msg"] = ("error", ["Сначала откройте те же исходные данные, затем выберите файл плана JSON."])
        return
    topics = ss.get("_topics")
    if topics is None:
        ss["restore_msg"] = ("error", ["Анализ заблокирован ошибками данных: восстановление невозможно."])
        return
    doc, errs = parse_plan(f.getvalue(), source_sig=src["sig"], topics=topics)
    if errs:
        ss["restore_msg"] = ("error", errs)
        return
    rest, errs = restore_plan(doc, topics=topics)
    if errs:
        ss["restore_msg"] = ("error", errs)
        return
    check = analyze(src["results"], src["topics"], rest.params, rest.test_choice, rest.locked)
    ctx = context_id(context_parts(src["sig"], rest.params, check.form_key)) if not check.blocked else None
    if ctx != rest.context:
        ss["restore_msg"] = ("error", ["Контекст плана не совпадает с текущими данными (формы теста или параметры "
                                       "дают другой набор измерений). Восстановление отменено."])
        return
    for name, *_ in PARAM_UI:                       # параметры методики
        ss[f"p_{name}"] = getattr(rest.params, name)
    ss["test_choice"] = dict(rest.test_choice)      # формы теста
    _clear_keys(("tc_",))
    _set_locked(rest.locked)
    ss["scenarios"] = list(rest.scenarios)
    ss["applied"] = None
    ss["applied_name"] = None
    ss["state_ctx"] = rest.context                  # контекст совпал — черновик не сбрасывать
    ss["state_parts"] = context_parts(src["sig"], rest.params, check.form_key)
    _set_draft(rest.draft or check.plan.proposed())
    ss["restore_msg"] = ("success", [f"План восстановлен как черновик: сценариев {len(rest.scenarios)}, "
                                     f"фиксаций {len(rest.locked)}. Утверждение из файла не переносится — "
                                     "утвердите план заново."])


def sync_state(analysis, ctx: str, parts: dict) -> None:
    """Смена данных, параметров или формы теста делает черновик и утверждение недействительными."""
    prev = ss.get("state_ctx")
    prev_parts = ss.get("state_parts")
    if ss.pop("_reseed_draft", False) and prev == ctx:
        _clear_keys(("hrs_",))
        ss["draft"] = dict(analysis.plan.proposed())
        ss["reset_note"] = "Фиксации изменены: черновик пересчитан по предложению алгоритма."
    if prev == ctx:
        return
    _clear_keys(("hrs_",))
    ss["draft"] = dict(analysis.plan.proposed())
    had_plan = ss.get("applied") is not None
    ss["applied"] = None
    ss["applied_name"] = None
    ss["apply_msg"] = None
    tail = ((", прежнее утверждение недействительно" if had_plan else "")
            + ". Сохранённые сценарии прежнего контекста помечены и требуют пересчёта.")
    if prev is None or prev_parts is None or prev_parts["source"] != parts["source"]:
        ss["reset_note"] = None
    elif prev_parts["forms"] != parts["forms"]:
        ss["reset_note"] = ("Форма теста изменена: черновик пересчитан, потому что прежний опирался на другой "
                            "набор измерений" + tail)
    else:
        ss["reset_note"] = "Параметры изменены: черновик пересчитан по новому предложению алгоритма" + tail
    ss["state_ctx"] = ctx
    ss["state_parts"] = parts


# ============================================================================
# Помощники показа
# ============================================================================
def f1(x) -> str:
    return "—" if pd.isna(x) else f"{x:.1f}"


def fi(x) -> str:
    return "—" if pd.isna(x) else f"{int(x)}"


def table(df: pd.DataFrame) -> None:
    st.dataframe(df, hide_index=True, width="stretch")


def card(key: str):
    """Контейнер-карточка. Ключ начинается с card_ — на него опираются стили."""
    return st.container(border=True, key=f"card_{key}")


STATUS_BADGE = {STATUS_HARD: ("Трудная", "red"), STATUS_OK: ("Без флага", "green"),
                STATUS_LOW: ("Мало данных", "gray"), STATUS_CHOOSE: ("Выберите форму", "orange")}


def status_badge(status: str) -> str:
    text, color = STATUS_BADGE.get(status, (status, "gray"))
    return f":{color}-badge[{text}]"


def tb_badge(name: str, colors: dict) -> str:
    safe = name.replace("[", "(").replace("]", ")")
    return f":{colors[name][1]}-badge[{safe}]"


def style_fig(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15)),
        height=height, margin=dict(l=10, r=10, t=78, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=TEXT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)))
    fig.update_xaxes(gridcolor=BORDER, automargin=True)
    fig.update_yaxes(gridcolor=BORDER, automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def banner(source: dict, details: bool = False) -> None:
    """Компактная пометка источника. Для демо — видна на каждом экране."""
    if source["synthetic"]:
        st.markdown(":orange-badge[Синтетические демо-данные] не результаты реальных школ")
    else:
        st.markdown(":gray-badge[Загруженные данные] Источник и качество измерения не проверялись.")
    if details:
        with st.expander("О происхождении данных и ограничениях"):
            if source["synthetic"]:
                st.write("Данные созданы генератором с фиксированным seed. Сценарий задан заранее: у «Учебник Б» "
                         "выше входной и итоговый результат, у «Учебник А» больше средний прирост. Это условие "
                         "демонстрации, а не результат исследования. Темы и часы — условный демонстрационный план "
                         "математики 6 класса, а не официальная учебная программа.")
            else:
                st.write("Файлы загружены пользователем. Приложение проверяет формат и корректность значений, "
                         "но не проверяет происхождение данных и качество измерения.")
            st.write("Различие результатов групп не доказывает влияние учебника: группы не сформированы случайно, "
                     "исходный уровень не корректируется. Пороги — настраиваемые допущения, не нормативы. "
                     "Перенос часов — сценарий, не прогноз роста баллов.")


def unresolved_notice(analysis) -> None:
    if not analysis.unresolved:
        return
    labels = topic_labels(analysis.topics)
    st.warning("**Выберите форму теста** (боковая панель) для тем: "
               + ", ".join(labels[t] for t in analysis.unresolved)
               + ". Эти темы пока не оцениваются, часы по ним не меняются, а весь план нельзя утвердить, "
               "пока выбор не завершён. Остальные темы анализируются как обычно.")


def blocked_notice(analysis) -> None:
    st.error("Анализ заблокирован: " + " ".join(analysis.block_reasons))
    st.caption("Исправьте файл и загрузите снова на экране «Данные» либо откройте демо.")


def sidebar_forms(analysis, box) -> None:
    """Постоянный выбор формы теста для тем, где форм несколько (виден на всех экранах)."""
    multi = {t: i for t, i in analysis.forms.items() if len(i.forms) > 1}
    if not multi:
        return
    labels = topic_labels(analysis.topics)
    with box:
        st.subheader("Формы теста")
        st.caption("В этих темах несколько форм теста. Все расчёты идут по одной выбранной форме. "
                   "Темы с единственной формой выбираются автоматически.")
        for tid, info in multi.items():
            key = f"tc_{tid}"
            if key not in ss:
                ss[key] = ss.get("test_choice", {}).get(tid, PLACEHOLDER)
            st.selectbox(f"{labels[tid]}", [PLACEHOLDER] + list(info.forms), key=key,
                         on_change=cb_choose_form, args=(tid,))


def sidebar_params() -> Params:
    defaults = Params()
    values = {}
    with st.sidebar.expander("Параметры методики", expanded=False):
        st.caption("Редактируемые параметры демонстрации, а не научные критерии и не государственные нормативы. "
                   "Смена параметров сбрасывает черновик и утверждённый план.")
        for name, label, kind, lo, hi, step, hint in PARAM_UI:
            default = getattr(defaults, name)
            if f"p_{name}" not in ss:           # значение задаётся через состояние: так его можно восстановить из JSON
                ss[f"p_{name}"] = kind(default)
            values[name] = st.number_input(label, min_value=kind(lo), max_value=kind(hi),
                                           step=kind(step), key=f"p_{name}", help=hint or None)
    return Params(**values)



# ============================================================================
# Обзор
# ============================================================================
def screen_overview(source: dict | None, analysis) -> None:
    if source is None:
        st.markdown(f'<div class="olshem-logo">Өлш<span>е</span>м</div>', unsafe_allow_html=True)
        st.subheader("Найти слабые темы, проверить основания и утвердить план учебных часов")
        st.write("Загрузите результаты входного и итогового тестов по темам. Өлшем покажет, где результаты ниже "
                 "порогов, как различаются учебники и как перераспределить часы при неизменном бюджете — "
                 "с фиксацией тем, сравнением вариантов и вашим утверждением.")
        c1, c2, _ = st.columns([1, 1, 2])
        c1.button("Открыть демо", type="primary", on_click=cb_open_demo, width="stretch")
        c2.button("Загрузить свои данные", on_click=cb_goto, args=("Данные",), width="stretch")
        st.caption("Демо использует синтетические данные, сгенерированные программой.")
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.markdown("**1 · Найти**  \nТемы с низким итоговым результатом и достаточным числом учеников.")
        c2.markdown("**2 · Проверить**  \nФорма теста, n, учебники: на чём держится вывод.")
        c3.markdown("**3 · Решить**  \nСценарии часов с фиксациями, сравнение и утверждение.")
        return

    st.header("Обзор")
    banner(source, details=True)
    if analysis.blocked:
        blocked_notice(analysis)
        st.button("Перейти к проверкам данных", on_click=cb_goto, args=("Данные",))
        return
    unresolved_notice(analysis)

    s, tt, plan = analysis.summary, analysis.topic_table, analysis.plan
    lab = topic_labels(analysis.topics)
    hard = tt[tt["status"] == STATUS_HARD].sort_values(["mean_post", "topic_id"])
    assessed = int(tt["status"].isin([STATUS_HARD, STATUS_OK]).sum())

    # --- главное: самая слабая трудная тема и следующий шаг
    with card("focus"):
        if len(hard):
            r = hard.iloc[0]
            left, right = st.columns([3, 2], gap="large")
            with left:
                st.caption("Требует внимания в первую очередь")
                st.subheader(lab[r.topic_id])
                st.markdown(f"Итоговый тест: **{r.mean_post:.1f}%** · ниже {params_cut()}: **{r.share_below:.1f}%** "
                            f"учеников · n = **{int(r.n_students)}** · форма {r.selected_test}")
                st.caption(r.reason)
            with right:
                gets = [(f, n) for f, t, n in plan.flows() if t == r.topic_id]
                st.caption("Доступное предложение по часам")
                if gets:
                    st.markdown("  \n".join(f"**+{n} ч** из «{lab[f]}»" for f, n in gets))
                else:
                    st.markdown("Для этой темы перенос не предложен.")
                st.button("Открыть план часов", type="primary", on_click=cb_goto, args=("Сценарии",),
                          width="stretch")
                st.button("Проверить основания", on_click=cb_goto, args=("Темы", r.topic_id), width="stretch")
        elif analysis.unresolved:
            st.markdown("Трудные темы будут видны после выбора формы теста.")
        else:
            st.markdown("Тем, достигших порогов трудности, не выявлено.")
            st.button("Открыть план часов", type="primary", on_click=cb_goto, args=("Сценарии",))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Учеников в анализе", s["students_used"],
              help="Ученики, чьи записи выбранных форм вошли в расчёт.")
    k2.metric("Трудных тем", f"{len(hard)} из {assessed}",
              help="Считаются только темы, где данных достаточно и форма теста выбрана.")
    k3.metric("Бюджет часов", f"{plan.budget} ч", help="Сумма исходных часов; при переносе не меняется.")
    k4.metric("Часов к переносу", f"{plan.hours_moved()} ч",
              help="Сумма положительных изменений по темам: каждый переносимый час учтён один раз.")
    st.caption(f"В файле: {s['students']} учеников и {s['rows']} валидных записей; в расчёте — "
               f"{s['students_used']} учеников и {s['rows_used']} записей (выбранные формы теста).")

    left, right = st.columns(2)
    with left:
        st.markdown("**Все трудные темы**")
        if len(hard):
            for r in hard.itertuples():
                st.markdown(f"{status_badge(STATUS_HARD)} **{lab[r.topic_id]}** — {r.mean_post:.1f}% · "
                            f"ниже {params_cut()}: {r.share_below:.1f}% · n = {r.n_students}")
        elif not analysis.unresolved:
            st.success("Тем, достигших порогов трудности, не выявлено.")
        low = tt[tt["status"] == STATUS_LOW]
        if len(low):
            st.caption("Мало данных, оценка не выполняется: " + ", ".join(lab[t] for t in low["topic_id"]) + ".")
    with right:
        st.markdown("**Предложение по часам**")
        if plan.moves:
            for f, t, n in plan.flows():
                st.markdown(f"«{lab[f]}» → «{lab[t]}»: **{n} ч**")
        else:
            for m in plan.messages:
                st.info(m)
            if not plan.messages:
                st.info("Переносов нет.")
        st.caption(f"Бюджет {plan.budget} ч до и после. Сценарий распределения, не прогноз роста баллов."
                   + (" План нельзя утвердить, пока не выбраны формы теста." if analysis.unresolved else ""))
        st.button("Сравнить учебники", on_click=cb_goto, args=("Учебники",))


def params_cut() -> str:
    return f"{ss.get('_cut', 50):g}%"


def analysis_cut(_summary) -> str:
    return params_cut()


# ============================================================================
# Темы
# ============================================================================
def screen_topics(analysis, params: Params) -> None:
    topic_focus(analysis, params)
    st.caption("Результат — средний процент итогового теста (post) от максимума. Это результат конкретного теста, "
               "а не доказанный уровень освоения знаний.")
    unresolved_notice(analysis)
    tt = analysis.topic_table
    labels = topic_labels(analysis.topics)

    with card("topics_summary"):
        heads = st.columns([3, 1.4, 1.6, 1, 1.8])
        for h, text in zip(heads, ("Тема", "Результат post", f"Ниже {params.low_score_cut:g}%", "n", "Статус")):
            h.caption(text)
        for r in tt.itertuples():
            c = st.columns([3, 1.4, 1.6, 1, 1.8])
            c[0].markdown(f"**{labels[r.topic_id]}**")
            if r.status == STATUS_CHOOSE:
                c[1].write("—"); c[2].write("—"); c[3].write("—")
            else:
                c[1].write(f"{r.mean_post:.1f}%" if r.n_students else "—")
                c[2].write(f"{r.share_below:.1f}%" if r.n_students else "—")
                c[3].write(str(r.n_students))
            c[4].markdown(status_badge(r.status))
    st.caption(f"Трудная тема: средний post < {params.hard_mean_below:g}% или доля ниже {params.low_score_cut:g}% "
               f"не меньше {params.hard_share_at_least:g}% при n ≥ {params.min_post_topic}. "
               "При меньшем n флаг не ставится.")

    tb = analysis.topic_tb
    tids = list(analysis.topics["topic_id"])          # темы — по topic_id, а не по названию
    books = sorted(tb["textbook"].unique())
    z, text = [], []
    for tid in tids:
        zr, tr = [], []
        for b in books:
            c = tb[(tb["topic_id"] == tid) & (tb["textbook"] == b)].iloc[0]
            zr.append(None if pd.isna(c.mean_post) else float(c.mean_post))
            if c.needs_choice:
                tr.append("выберите<br>форму теста")
            elif pd.isna(c.mean_post):
                tr.append("нет данных")
            else:
                tr.append(f"{c.mean_post:.0f}%<br>n={int(c.n_students)}" + ("<br>мало" if not c.enough else ""))
        z.append(zr)
        text.append(tr)
    fig = go.Figure(go.Heatmap(z=z, x=books, y=[labels[t] for t in tids], text=text, texttemplate="%{text}",
                               colorscale=[[0, "#EAF0FE"], [1, ACCENT]], zmin=0, zmax=100,
                               colorbar=dict(title="post, %")))
    fig.update_yaxes(autorange="reversed")
    style_fig(fig, "Карта тем: результат post, %", height=130 + 60 * len(tids))
    show(fig)
    st.caption("В ячейке — средний post и число учеников. Пометка «мало» — учеников меньше порога оценки темы. "
               "Цвет показывает уровень результата, а не «лучший» учебник.")

    with st.expander("Подробности: таблица и график"):
        view = pd.DataFrame({
            "Тема": tt["topic_id"].map(labels),
            "Форма теста": tt["selected_test"].replace("", "не выбрана"),
            "Записей post в файле": tt["n_post_file"],
            "Использовано в расчёте": tt["n_post_rows"],
            "Учеников (post)": tt["n_students"],
            "Классов": tt["n_classes"],
            "Средний post, %": tt["mean_post"].map(f1),
            f"Доля ниже {params.low_score_cut:g}%, %": tt["share_below"].map(f1),
            "Статус": tt["status"],
            "Почему": tt["reason"],
        })
        table(view)
        st.caption("«В файле» — валидные записи post по теме во всех формах; «использовано» — только выбранная форма.")
        colors = {STATUS_HARD: "#B42318", STATUS_LOW: "#98A2B3", STATUS_CHOOSE: "#98A2B3"}
        bar = go.Figure(go.Bar(x=list(tt["topic_id"].map(labels)), y=tt["share_below"],
                               marker_color=[colors.get(x, ACCENT) for x in tt["status"]],
                               text=tt["share_below"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}%"),
                               textposition="outside"))
        bar.add_hline(y=params.hard_share_at_least, line_dash="dash",
                      annotation_text=f"порог {params.hard_share_at_least:g}%")
        bar.update_layout(yaxis_range=[0, 100], yaxis_title="% учеников", xaxis_title="Тема")
        style_fig(bar, f"Доля ниже {params.low_score_cut:g}%, %")
        show(bar)
        st.caption("Красный — тема отмечена трудной, серый — мало данных или не выбрана форма теста.")



def topic_focus(analysis, params: Params) -> None:
    """«Проверить основания»: одна тема — форма теста, n, причина статуса, результат по учебникам."""
    lab = topic_labels(analysis.topics)
    tids = list(analysis.topics["topic_id"])
    tt = analysis.topic_table.set_index("topic_id")
    if ss.get("topic_focus") not in tids:
        hard = analysis.topic_table[analysis.topic_table["status"] == STATUS_HARD].sort_values(["mean_post", "topic_id"])
        ss["topic_focus"] = hard["topic_id"].iloc[0] if len(hard) else tids[0]
    with card("topic_focus"):
        st.selectbox("Проверить основания по теме", tids, key="topic_focus", format_func=lab.get)
        t = tt.loc[ss["topic_focus"]]
        info = analysis.forms[ss["topic_focus"]]
        st.markdown(f"**Статус:** {t['status']} · {t['reason']}")
        st.caption(f"Форма теста: {info.selected or 'не выбрана'} (в данных: {', '.join(info.forms) or '—'}) · "
                   f"записей итогового теста: в файле {int(t['n_post_file'])}, в расчёте {int(t['n_post_rows'])} · "
                   f"классов: {int(t['n_classes'])}")
        tb = analysis.topic_tb[analysis.topic_tb["topic_id"] == ss["topic_focus"]]
        cols = st.columns(max(len(tb), 1))
        colors = textbook_colors(analysis.textbooks)
        for col, r in zip(cols, tb.itertuples()):
            with col:
                st.markdown(tb_badge(r.textbook, colors))
                if r.needs_choice or pd.isna(r.mean_post):
                    st.write("нет данных по выбранной форме")
                else:
                    st.metric("Итоговый тест, %", f"{r.mean_post:.1f}")
                    st.caption(f"n = {int(r.n_students)}" + ("" if r.enough else " · мало данных"))
        st.caption("Это описание результата теста, а не доказанный уровень освоения и не эффект учебника.")


# ============================================================================
# Учебники
# ============================================================================
def _grouped_bars(title: str, xs, series, y_title: str, digits: int = 0) -> go.Figure:
    """Подписи короткие (целые для процентов), за ось не обрезаются; точные значения — в подсказке и таблице."""
    fig = go.Figure()
    top = 0.0
    for name, color, ys, ns in series:
        ys = list(ys)
        top = max([top] + [float(v) for v in ys if not pd.isna(v)])
        fig.add_trace(go.Bar(
            name=name, x=list(xs), y=ys, marker_color=color, customdata=list(ns),
            text=["" if pd.isna(v) else f"{v:.{digits}f}" for v in ys], textposition="outside",
            textfont=dict(size=11), cliponaxis=False,
            hovertemplate="%{x}<br>%{y:.1f}<br>n = %{customdata}<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group", yaxis_title=y_title, bargap=0.28, bargroupgap=0.08,
                      uniformtext=dict(minsize=10, mode="show"),
                      yaxis_range=[0, top * 1.18 if top > 0 else 1])
    return style_fig(fig, title, height=380)


def screen_textbooks(analysis, params: Params) -> None:
    unresolved_notice(analysis)
    cmp = analysis.comparison
    for m in cmp.messages:
        st.info(m)
    if cmp.table.empty:
        return
    lab = topic_labels(analysis.topics)
    colors = textbook_colors(analysis.textbooks)

    # --- 1. выбор пары
    t_all = cmp.table
    books = sorted(set(t_all["textbook_a"]) | set(t_all["textbook_b"]))
    pair_options = {f"«{a}» и «{b}»": (a, b) for a, b in combinations(books, 2)}
    options = list(pair_options)
    if ss.get("pair") not in pair_options:
        ss["pair"] = options[0]
    if ss.get("pair_sel") not in pair_options:
        ss["pair_sel"] = ss["pair"]
    st.selectbox("Пара учебников для сравнения", options, key="pair_sel", on_change=cb_pair,
                 help="Графики и таблицы ниже относятся только к выбранной паре.")
    a_name, b_name = pair_options[ss["pair_sel"]]
    st.markdown(f"{tb_badge(a_name, colors)} {tb_badge(b_name, colors)}")
    t = t_all[(t_all["textbook_a"] == a_name) & (t_all["textbook_b"] == b_name)]   # одна строка на тему
    st.info("**Описательное сравнение:** различие не доказывает влияние учебника. Группы не случайны, "
            "исходный уровень не корректируется.")

    # --- 2. итоговый результат и прирост рядом
    ok = t[t["status"] == ST_OK]
    xs_all = [lab[x] for x in t["topic_id"]]
    c1, c2 = st.columns(2)
    with c1:
        show(_grouped_bars("Итоговый результат post, %", xs_all, [
            (a_name, colors[a_name][0], t["post_a"], t["n_post_a"]),
            (b_name, colors[b_name][0], t["post_b"], t["n_post_b"])], "%"))
    with c2:
        if len(ok):
            xs_ok = [lab[x] for x in ok["topic_id"]]                      # каждая тема — один раз
            show(_grouped_bars("Средний прирост, п.п.", xs_ok, [
                (a_name, colors[a_name][0], ok["gain_a"], ok["n_pairs_a"]),
                (b_name, colors[b_name][0], ok["gain_b"], ok["n_pairs_b"])], "п.п.", digits=1))
        else:
            st.info("Нет тем, где данных достаточно для сравнения приростов этой пары, поэтому график прироста "
                    "не строится. Итоговые результаты слева показаны с n.")
    st.caption("Итоговый тест (post) — все ученики с результатом. Прирост = итоговый − входной (post − pre) у тех же "
               "учеников, только для тем с достаточным числом пар и классов. Подписи округлены; точные значения — "
               "в подсказке и в «Подробных таблицах».")

    # --- 3. выбранная тема: pre → post
    tids = list(analysis.topics["topic_id"])
    if ss.get("cmp_topic") not in tids:
        ss["cmp_topic"] = ok["topic_id"].iloc[0] if len(ok) else tids[0]
    if ss.get("topic_sel") not in tids:
        ss["topic_sel"] = ss["cmp_topic"]
    with card("topic_detail"):
        st.selectbox("Тема для подробного сравнения", tids, key="topic_sel", format_func=lab.get,
                     on_change=cb_cmp_topic)
        r = t[t["topic_id"] == ss["topic_sel"]].iloc[0]
        st.markdown(f"**{lab[r.topic_id]}** · форма теста: {r.test_id or 'не выбрана'} · {STATUS_LABELS[r.status]}")
        cols = st.columns(2)
        for col, side, name in ((cols[0], "a", a_name), (cols[1], "b", b_name)):
            with col:
                st.markdown(tb_badge(name, colors))
                if r.status == ST_OK:
                    pre, post, gain = getattr(r, f"pre_{side}"), getattr(r, f"post_pair_{side}"), getattr(r, f"gain_{side}")
                    st.metric("Входной → итоговый, те же пары", f"{pre:.1f}% → {post:.1f}%", delta=f"{gain:+.1f} п.п.",
                              delta_color="off")
                    st.caption(f"n пар: {fi(getattr(r, f'n_pairs_{side}'))} · классов: {fi(getattr(r, f'classes_{side}'))}"
                               f" · доля пар: {100 * getattr(r, f'pair_share_{side}'):.0f}%")
                else:
                    st.metric("Итоговый результат post, все ученики", f"{f1(getattr(r, f'post_{side}'))}%"
                              if not pd.isna(getattr(r, f"post_{side}")) else "—")
                    st.caption(f"n с post: {fi(getattr(r, f'n_post_{side}'))} · пар pre/post: "
                               f"{fi(getattr(r, f'n_pairs_{side}'))} · классов среди пар: {fi(getattr(r, f'classes_{side}'))}")
        if r.status == ST_OK:
            st.markdown(f"**Разница приростов: {f1(r.gain_diff)} п.п.** («{a_name}» − «{b_name}»). {r.direction}")
            fig = go.Figure()
            for name, side in ((a_name, "a"), (b_name, "b")):
                fig.add_trace(go.Bar(
                    name=name, x=["входной (pre)", "итоговый (post)"], marker_color=colors[name][0],
                    y=[getattr(r, f"pre_{side}"), getattr(r, f"post_pair_{side}")],
                    text=[f"{getattr(r, f'pre_{side}'):.1f}", f"{getattr(r, f'post_pair_{side}'):.1f}"],
                    textposition="outside"))
            fig.update_layout(barmode="group", yaxis_title="%", yaxis_range=[0, 105])
            show(style_fig(fig, "Входной → итоговый, % (те же пары)", height=320))
            for w in r.warnings:
                st.info(w)
        else:
            st.markdown(f"**{r.direction}**")
            for w in r.warnings:          # существенные проблемы выбранной темы не прячем
                st.warning(w)

    # --- 4. сводка предупреждений о входном уровне
    pre_warn = [(x.topic_id, w) for x in t.itertuples() for w in x.warnings if "Входные уровни" in w]
    if len(ok):
        st.markdown(f"**Входной уровень групп.** Различие pre больше порога ({params.pre_gap_warn_pp:g} п.п.) "
                    f"в {len(pre_warn)} из {len(ok)} тем с прямым сравнением.")
        with st.expander("Подробности по темам"):
            if pre_warn:
                for tid, w in pre_warn:
                    st.markdown(f"- **{lab[tid]}.** {w}")
            else:
                st.write("Превышений порога нет. Это не доказывает сопоставимость групп.")

    # --- 5. подробные таблицы
    with st.expander("Подробные таблицы"):
        long_rows = []
        for x in t.itertuples(index=False):
            okx = x.status == ST_OK
            for side, tb in (("a", x.textbook_a), ("b", x.textbook_b)):
                share = getattr(x, f"pair_share_{side}")
                long_rows.append({
                    "Тема": lab[x.topic_id], "Учебник": tb, "Форма теста": x.test_id or "—",
                    "Учеников с post": fi(getattr(x, f"n_post_{side}")),
                    "Средний post, % (все ученики)": f1(getattr(x, f"post_{side}")),
                    "Пар pre/post": fi(getattr(x, f"n_pairs_{side}")),
                    "Доля пар, %": "—" if pd.isna(share) else f"{100 * share:.0f}",
                    "Классов (среди пар)": fi(getattr(x, f"classes_{side}")),
                    "pre, % (пары)": f1(getattr(x, f"pre_{side}")) if okx else "—",
                    "post, % (те же пары)": f1(getattr(x, f"post_pair_{side}")) if okx else "—",
                    "Прирост, п.п.": f1(getattr(x, f"gain_{side}")) if okx else "—",
                })
        table(pd.DataFrame(long_rows))
        table(pd.DataFrame([{
            "Тема": lab[x.topic_id], "Пара": f"«{x.textbook_a}» и «{x.textbook_b}»",
            "Статус": STATUS_LABELS[x.status],
            "Разница приростов, п.п.": f1(x.gain_diff), "Разница pre, п.п.": f1(x.pre_gap),
            "Вывод (описательный)": x.direction,
        } for x in t.itertuples(index=False)]))
        st.caption(f"Разница = «{a_name}» − «{b_name}», в процентных пунктах (п.п.). Доля пар — какая часть учеников "
                   "с post имеет пару pre/post. Общего рейтинга учебников нет: темы не объединяются.")
        for x in t.itertuples(index=False):
            for w in x.warnings:
                st.markdown(f"- **{lab[x.topic_id]}.** {w}")



# ============================================================================
# Сценарии плана часов
# ============================================================================
def plan_status(analysis, errors: list[str]) -> str:
    applied, draft = ss.get("applied"), ss.get("draft", {})
    if analysis.unresolved:
        return "Нельзя утвердить: не выбрана форма теста"
    if applied is not None:
        same = applied == {k: int(v) for k, v in draft.items()} and ss.get("applied_locked", []) == ss.get("locked", [])
        return "Утверждён" if same else "Утверждён; есть неприменённые правки"
    if errors:
        return "Черновик некорректен"
    if draft == analysis.plan.proposed():
        return "Предложение алгоритма, не утверждено"
    return "Черновик изменён, не утверждён"


def _msg(key: str, err_prefix: str = "") -> None:
    msg = ss.get(key)
    if msg:
        kind, lines = msg
        if kind == "error":
            st.error(err_prefix + "\n".join(f"- {e}" for e in lines))
        else:
            st.success(lines[0] if len(lines) == 1 else "\n".join(lines))


def screen_scenarios(analysis, params: Params) -> None:
    st.caption("Часы относятся к общему плану группы. **Сценарий распределения, не прогноз роста баллов.** "
               "Бюджет не меняется: часы перераспределяются между темами.")
    if ss.get("reset_note"):
        st.info(ss["reset_note"])

    plan, lab, topics = analysis.plan, topic_labels(analysis.topics), analysis.topics
    applied, draft, locked = ss.get("applied"), ss["draft"], ss.get("locked", [])
    errors = validate_plan(topics, draft, locked)
    total = sum(v for v in draft.values() if isinstance(v, (int, float)))
    tt = analysis.topic_table.set_index("topic_id")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Исходный бюджет", f"{plan.budget} ч")
    k2.metric("Сумма черновика", f"{total:g} ч",
              delta=(f"{total - plan.budget:+g} к бюджету" if total != plan.budget else None), delta_color="off")
    k3.metric("Часов к переносу", f"{plan.hours_moved()} ч", help="Сумма положительных изменений предложения.")
    with k4.container(border=True, key="card_plan_status"):
        st.caption("Статус плана")
        st.markdown(f"**{plan_status(analysis, errors)}**")
        if applied is not None:
            st.caption(f"Утверждён: «{ss.get('applied_name')}»")

    for m in plan.messages:
        if m.startswith("Допустимых переносов нет"):
            st.success(m + " Это нормальный результат, а не ошибка.")
        else:
            st.caption(m)

    # --- черновик: исходно / предложено / черновик / фиксация
    with card("draft"):
        st.markdown("**Черновик плана**")
        W = [3, 1.6, 0.9, 1.1, 1.4, 1.3]
        for c, h in zip(st.columns(W), ("Тема", "Итоговый тест", "Исходно", "Предложено", "Черновик, ч",
                                        "Не изменять")):
            c.caption(h)
        for r in plan.table.itertuples(index=False):
            t = tt.loc[r.topic_id]
            c = st.columns(W, vertical_alignment="center")
            c[0].markdown(f"**{lab[r.topic_id]}**  \n:gray[{r.note}]")
            c[1].markdown("—" if pd.isna(t["mean_post"]) else f"{t['mean_post']:.1f}% · n {int(t['n_students'])}")
            c[2].markdown(f"{r.hours_before}")
            d = int(r.delta)
            c[3].markdown(f"**{r.hours_proposed}**" + (f" :blue[({d:+d})]" if d else ""))
            key = f"hrs_{r.topic_id}"
            if key not in ss:
                ss[key] = int(draft.get(r.topic_id, r.hours_proposed))
            c[4].number_input(f"{lab[r.topic_id]}: часы (допустимо {r.hours_low}–{r.hours_high})", min_value=0,
                              step=1, key=key, on_change=cb_draft_changed, args=(r.topic_id,),
                              disabled=r.topic_id in locked, label_visibility="collapsed",
                              help=f"Допустимо {r.hours_low}–{r.hours_high} ч")
            lk = f"lock_{r.topic_id}"
            if lk not in ss:
                ss[lk] = r.topic_id in locked
            c[5].checkbox(f"Не изменять тему «{lab[r.topic_id]}»", key=lk, on_change=cb_lock, args=(r.topic_id,),
                          label_visibility="collapsed")
        st.caption(f"Сумма черновика: **{total:g}** из бюджета **{plan.budget}**. "
                   "«Не изменять» фиксирует исходные часы: зафиксированная тема не отдаёт и не получает часы.")
        if analysis.unresolved:
            st.warning("Утвердить план нельзя: выберите форму теста для тем «"
                       + "», «".join(lab[t] for t in analysis.unresolved) + "» (боковая панель).")
        elif errors:
            st.warning("Черновик пока нельзя утвердить или сохранить:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            st.success("Черновик корректен: его можно утвердить или сохранить как сценарий.")
        b1, b2 = st.columns(2)
        b1.button("Утвердить черновик", on_click=cb_apply, type="primary", width="stretch")
        b2.button("Сбросить черновик к предложению алгоритма", on_click=cb_reset_draft, width="stretch")
        _msg("apply_msg", "План не принят — сохранён предыдущий валидный план:\n\n")

    pt = plan.table
    xs = list(pt["topic_id"].map(lab))
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Исходно", x=xs, y=list(pt["hours_before"]), marker_color="#B8C0CE"))
    fig.add_trace(go.Bar(name="Предложено", x=xs, y=list(pt["hours_proposed"]), marker_color=ACCENT))
    fig.add_trace(go.Bar(name="Черновик", x=xs, y=[draft.get(t) for t in pt["topic_id"]], marker_color=TEXT))
    fig.update_layout(barmode="group", yaxis_title="часов", bargap=0.3)
    show(style_fig(fig, "Часы по темам, ч", height=320))

    # --- сохранение сценариев
    scs = ss.get("scenarios", [])
    ctx = ss["_ctx"]
    st.subheader(f"Сценарии ({len(scs)} из {MAX_SCENARIOS})")
    n1, n2 = st.columns([3, 1], vertical_alignment="bottom")
    n1.text_input("Название сценария", key="sc_name", max_chars=60,
                  placeholder="например, «Без изменения дробей»")
    n2.button("Сохранить сценарий", on_click=cb_save_scenario, width="stretch")
    _msg("sc_msg")
    for i, sc in enumerate(scs):
        probs = scenario_problems(sc, topics, ctx)
        stale = sc.context != ctx
        state = (":orange-badge[Другой контекст данных]" if stale
                 else ":red-badge[Некорректен]" if probs else ":blue-badge[Актуален]")
        if applied is not None and ss.get("applied_name") == sc.name and applied == sc.hours:
            state += " :green-badge[Утверждён]"
        with card(f"sc_{i}"):
            a, b = st.columns([1, 1], vertical_alignment="center")
            a.markdown(f"**{sc.name}** {state}  \n:gray[сумма {sum(sc.hours.values())} ч · фиксации: "
                       f"{', '.join(lab.get(t, t) for t in sc.locked) or 'нет'} · контекст {sc.context}]")
            bc = b.columns(3)
            if stale:
                bc[0].button("Пересчитать", key=f"sc_recalc_{i}", on_click=cb_recalc_scenario, args=(i,),
                             width="stretch")
            else:
                bc[0].button("В черновик", key=f"sc_load_{i}", on_click=cb_load_scenario, args=(i,),
                             width="stretch")
            bc[1].button("Утвердить", key=f"sc_ok_{i}", on_click=cb_approve_scenario, args=(i,), type="primary",
                         disabled=bool(probs) or bool(analysis.unresolved), width="stretch")
            bc[2].button("Удалить", key=f"sc_del_{i}", on_click=cb_delete_scenario, args=(i,), width="stretch")
            if stale:
                st.caption("Создан на других данных, параметрах или формах теста. Пересчитайте, чтобы применить.")

    # --- сравнение двух сценариев
    if len(scs) >= 2:
        names = [s.name for s in scs]
        for k, default in (("cmp_a", names[0]), ("cmp_b", names[1])):
            if ss.get(k) not in names:
                ss[k] = default
        st.markdown("**Сравнить два сценария**")
        c1, c2 = st.columns(2)
        c1.selectbox("Сценарий А", names, key="cmp_a")
        c2.selectbox("Сценарий Б", names, key="cmp_b")
        sa, sb = scs[names.index(ss["cmp_a"])], scs[names.index(ss["cmp_b"])]
        if sa.context != ctx or sb.context != ctx:
            st.warning("Один из сценариев создан в другом контексте данных: сравнение на одних исходных данных "
                       "невозможно. Пересчитайте его.")
        else:
            cmp = compare_scenarios(topics, sa, sb, lab)
            table(cmp)
            moved = int(cmp["Разница Б − А, ч"].clip(lower=0).sum())
            st.caption(f"Суммы: «{sa.name}» {sum(sa.hours.values())} ч, «{sb.name}» {sum(sb.hours.values())} ч. "
                       f"Различаются на {moved} ч. Сравниваются только часы и фиксации: прогноз баллов не строится.")
    elif scs:
        st.caption("Сохраните ещё один сценарий, чтобы сравнить два варианта.")

    with st.expander("Подробности: журнал переносов и правила"):
        if plan.moves:
            table(plan.moves_table())
        else:
            st.write("Переносов нет.")
        for m in plan.messages:
            st.write(m)
        st.write(f"Цель по итоговому тесту — {params.target:g}%; донор — тема с результатом от {params.donor_from:g}%; "
                 f"получатель — ниже {params.target:g}%. За шаг переносится 1 час от темы с наибольшим запасом к теме "
                 "с наибольшим дефицитом по ИСХОДНЫМ результатам (при равенстве — порядок topic_id). Границы темы: "
                 "от max(min_hours, hours − max_delta) до hours + max_delta. Зафиксированные темы и темы без "
                 "достаточного числа результатов итогового теста не меняются. Пороги — допущения демо, не нормативы.")


# ============================================================================
# Решение: статус, экспорт, восстановление
# ============================================================================
def decision_now(analysis, params: Params, source: dict):
    return build_decision(analysis, params, source=source, context=ss["_ctx"], context_parts=ss["_ctx_parts"],
                          test_choice=ss.get("test_choice", {}), scenarios=ss.get("scenarios", []),
                          applied=ss.get("applied"), applied_name=ss.get("applied_name"),
                          applied_locked=ss.get("applied_locked", []), draft=ss.get("draft", {}),
                          locked=ss.get("locked", []))


def screen_decision(analysis, params: Params, source: dict) -> None:
    unresolved_notice(analysis)
    lab = topic_labels(analysis.topics)
    tt, s, rv = analysis.topic_table, analysis.summary, analysis.results_val
    d = decision_now(analysis, params, source)
    name, hours, locked, approved, errors = d.name, d.hours, d.locked, d.approved, d.errors

    with card("decision"):
        if approved:
            st.markdown(f":green-badge[Утверждён] **План «{name}»** — утверждён методистом в этой сессии.")
        else:
            st.markdown(f":orange-badge[Не утверждён] **{name}** — в экспорт попадёт как неутверждённый черновик.")
            if errors:
                st.caption("Черновик некорректен: " + " ".join(errors)
                           + " План (JSON) недоступен, пока черновик не исправлен.")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Скачать паспорт решения (HTML)", d.html.encode("utf-8"), "olshem_passport.html",
                           "text/html", on_click="ignore", key="dl_passport", type="primary", width="stretch")
        # JSON — то же показанное решение и его фиксации; после восстановления это неутверждённый черновик.
        c2.download_button("Скачать план (JSON)", d.plan_json or b"", "olshem_plan.json", "application/json",
                           on_click="ignore", key="dl_plan", width="stretch", disabled=d.plan_json is None,
                           help=None if d.plan_json else "Черновик некорректен: исправьте его в «Сценариях».")
        c3.download_button("Скачать отчёт (CSV)", d.csv, "olshem_report.csv", "text/csv",
                           on_click="ignore", key="dl_report", width="stretch")
        st.caption("Паспорт, план и отчёт относятся к показанному здесь варианту. PDF — печатью паспорта в браузере. "
                   "JSON хранит настройки, фиксации и сценарии без поученичных данных: для восстановления нужны "
                   "те же исходные CSV, а утверждение после восстановления нужно повторить.")

    with st.expander("Восстановить план из JSON"):
        st.caption("Откройте те же исходные данные (для демо — то же демо), затем загрузите файл olshem_plan.json. "
                   "Восстановленный план станет черновиком: утверждение из файла не переносится.")
        st.file_uploader("Файл плана (JSON)", type=["json"], key="up_plan")
        st.button("Восстановить план", on_click=cb_restore_plan)
        _msg("restore_msg")

    st.subheader("Основные находки")
    hard = tt[tt["status"] == STATUS_HARD]
    for r in hard.itertuples():
        st.markdown(f"- **{lab[r.topic_id]}** — трудная тема: итоговый тест {r.mean_post:.1f}%, ниже "
                    f"{params.low_score_cut:g}%: {r.share_below:.1f}%, n = {r.n_students} (форма {r.selected_test}).")
    if not len(hard) and not analysis.unresolved:
        st.markdown("- Тем, достигших порогов трудности, не выявлено.")
    low = tt[tt["status"] == STATUS_LOW]
    if len(low):
        st.markdown("- Мало данных для оценки: " + ", ".join(lab[t] for t in low["topic_id"]) + ".")
    st.markdown(f"- Данные: в файле {s['rows']} валидных записей, в расчёте {s['rows_used']}; "
                f"исключено при проверке: {rv.n_excluded_rows}.")

    st.subheader("План часов")
    pt = analysis.plan_for(locked).table
    table(pd.DataFrame({"Тема": pt["topic_id"].map(lab), "Исходно, ч": pt["hours_before"],
                        "Предложено, ч": pt["hours_proposed"],
                        ("Утверждено, ч" if approved else "Черновик, не утверждено, ч"): [hours.get(t, "—") for t in pt["topic_id"]],
                        "Не изменять": ["да" if t in locked else "—" for t in pt["topic_id"]],
                        "Пояснение": pt["note"]}))
    st.caption("Сценарий распределения, не прогноз роста баллов.")

    st.subheader("Сравнение учебников по темам")
    cmp = analysis.comparison.table
    if cmp.empty:
        st.write("; ".join(analysis.comparison.messages))
    else:
        table(pd.DataFrame({
            "Тема": cmp["topic_id"].map(lab),
            "Пара": cmp.apply(lambda x: f"«{x.textbook_a}» и «{x.textbook_b}»", axis=1),
            "Статус": cmp["status"].map(STATUS_LABELS),
            "Разница приростов, п.п.": cmp["gain_diff"].map(f1),
            "Вывод": cmp["direction"],
        }))
    st.caption("Описательное сравнение: различие не доказывает влияние учебника.")
    with st.expander("Методика и ограничения"):
        st.write("Результат — средний процент итогового теста (post) от максимума, по выбранной форме теста каждой "
                 "темы. Прирост = post − pre у одних и тех же учеников — описательное изменение результата. "
                 "Исходный уровень не корректируется, доверительных интервалов и общего рейтинга учебников нет. "
                 "Пороги — настраиваемые допущения демо, не научные критерии и не нормативы. Перенос часов — "
                 "сценарий, не прогноз; план утверждает методист. Если Excel откроет CSV в одном столбце, "
                 "используйте «Данные → Из текста/CSV» и разделитель «запятая».")


# ============================================================================
# Данные
# ============================================================================
def screen_data(source: dict | None, analysis) -> None:
    st.header("Данные")
    with card("source"):
        if source is None:
            st.write("Данные ещё не выбраны. Откройте демо или загрузите два CSV-файла.")
        else:
            banner(source, details=True)
            st.caption(f"Источник: {source['label']}")
        c1, c2 = st.columns([1, 3])
        c1.button("Открыть демо", on_click=cb_open_demo, type="primary", width="stretch",
                  help="Синтетические данные: 2 условных учебника, 8 классов, 6 тем.")
        c2.caption("Демо не требует загрузки файлов.")

    with st.expander("Загрузить свои CSV и скачать шаблоны", expanded=source is None):
        u1, u2 = st.columns(2)
        u1.file_uploader("results.csv", type=["csv"], key="up_results")
        u2.file_uploader("topics.csv", type=["csv"], key="up_topics")
        st.button("Использовать загруженные файлы", on_click=cb_use_uploads)
        if ss.get("upload_error"):
            st.error(ss["upload_error"])
        st.caption("results.csv: student_id, class_id, textbook, topic_id, stage (pre/post), test_id, score, max_score. "
                   "topics.csv: topic_id, name, hours, min_hours, max_delta. Используйте обезличенные идентификаторы.")
        d1, d2 = st.columns(2)
        d1.download_button("Шаблон results.csv", demo.results_template_csv().encode("utf-8-sig"),
                           "template_results.csv", "text/csv", on_click="ignore", key="dl_tpl_res")
        d2.download_button("Шаблон topics.csv", demo.topics_template_csv().encode("utf-8-sig"),
                           "template_topics.csv", "text/csv", on_click="ignore", key="dl_tpl_top")

    with st.expander("Проверить ограничения демо"):
        st.caption("Два готовых примера, чтобы увидеть, как приложение ведёт себя при нехватке данных и при ошибках.")
        e1, e2 = st.columns(2)
        e1.button("Демо с малой выборкой", on_click=cb_open_demo_small, width="stretch",
                  help="Тем хватает для оценки, а для сравнения учебников — нет.")
        e2.button("Пример плохого CSV", on_click=cb_open_bad, width="stretch",
                  help="Намеренно испорченный файл: смотрите, как приложение сообщает об ошибках.")

    if source is None:
        return

    st.subheader("Проверки данных")
    tv, rv = analysis.topics_val, analysis.results_val
    for e in tv.errors:
        st.error("topics.csv: " + e)
    for e in rv.errors:
        st.error("results.csv: " + e)
    for e in analysis.param_errors:
        st.error("Параметры: " + e)
    if tv.ok and rv.ok and not analysis.param_errors:
        st.success(f"Структура файлов корректна. Тем: {len(tv.clean)}.")
    for w in tv.warnings:
        st.warning("topics.csv: " + w)
    for w in rv.warnings:
        st.warning("results.csv: " + w)

    m1, m2, m3 = st.columns(3)
    m1.metric("Записей в файле", rv.n_input_rows)
    m2.metric("Валидных записей", rv.n_valid_rows)
    m3.metric("Исключено записей", rv.n_excluded_rows)
    if rv.exclusion_counts:
        st.markdown("**Исключённые записи** (пропуски не превращаются в нули):")
        table(pd.DataFrame([{"Причина": k, "Записей": v} for k, v in rv.exclusion_counts.items()]))
        with st.expander("Показать исключённые записи (до 100)"):
            table(rv.excluded.head(100).rename(columns={"record_no": "№ записи", "reason": "Причина"}))

    if analysis.blocked:
        blocked_notice(analysis)
        return

    s = analysis.summary
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Учеников", s["students"])
    n2.metric("Классов", s["classes"])
    n3.metric("Учебников", s["textbooks"])
    n4.metric("Использовано в расчёте", f"{s['rows_used']} из {s['rows']}",
              help="Записи выбранных форм теста от всех валидных записей файла.")
    unresolved_notice(analysis)

    st.markdown("**Достаточность данных по темам** — три независимые проверки:")
    table(analysis.sufficiency)
    st.caption("Данных может хватать для оценки темы, но не хватать для сравнения учебников. "
               "Тогда блокируется только сильный сравнительный вывод, а описательные числа остаются.")
    with st.expander("Формы теста по темам"):
        lab = topic_labels(analysis.topics)
        table(pd.DataFrame([{
            "Тема": lab[t], "Формы в данных": ", ".join(i.forms) or "—",
            "Выбрана": i.selected or "не выбрана",
            "Как выбрана": {"auto": "автоматически (форма одна)", "chosen": "вручную",
                            "needs_choice": "нужно выбрать", "none": "нет данных"}[i.mode],
        } for t, i in analysis.forms.items()]))
        st.caption("Выбор формы меняется в боковой панели без повторной загрузки файла. "
                   "Одинаковый test_id — проверка формата, а не доказательство эквивалентности измерения.")



# ============================================================================
# Точка входа
# ============================================================================
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="olshem-logo">Өлш<span>е</span>м</div>', unsafe_allow_html=True)
    st.sidebar.caption(TAGLINE)
    if "nav" not in ss:
        ss["nav"] = SECTIONS[0]
    st.sidebar.radio("Раздел", SECTIONS, key="nav", label_visibility="collapsed")
    forms_box = st.sidebar.container()          # сюда ниже попадёт выбор форм теста
    params = sidebar_params()
    ss["_cut"] = params.low_score_cut
    ss.setdefault("locked", [])
    ss.setdefault("scenarios", [])

    source = ss.get("source")
    analysis = None
    if source is not None:
        analysis = analyze(source["results"], source["topics"], params, ss.get("test_choice", {}), ss["locked"])
        if not analysis.blocked:
            parts = context_parts(source["sig"], params, analysis.form_key)
            ctx = context_id(parts)
            ss["_ctx"], ss["_ctx_parts"], ss["_analysis"] = ctx, parts, analysis
            sync_state(analysis, ctx, parts)
            lab = topic_labels(analysis.topics)
            ss["_topics"] = analysis.topics
            ss["_proposed"] = analysis.plan.proposed()
            ss["_unresolved"] = [lab[t] for t in analysis.unresolved]
            sidebar_forms(analysis, forms_box)
            st.sidebar.caption(("Синтетическое демо" if source["synthetic"] else "Загруженные данные")
                               + f" · контекст {ctx}")
        else:
            ss["_topics"] = None
            ss["_unresolved"] = []

    section = ss["nav"]
    if section == "Обзор":
        screen_overview(source, analysis)
        return
    if section == "Данные":
        screen_data(source, analysis)
        return
    if source is None:
        st.header(section)
        st.info("Сначала откройте демо или загрузите файлы на экране «Данные» или «Обзор».")
        st.button("Открыть демо", type="primary", on_click=cb_open_demo)
        return
    st.header(section)
    banner(source, details=True)
    if analysis.blocked:
        blocked_notice(analysis)
        st.button("Перейти к проверкам данных", on_click=cb_goto, args=("Данные",))
        return
    if section == "Темы":
        screen_topics(analysis, params)
    elif section == "Учебники":
        screen_textbooks(analysis, params)
    elif section == "Сценарии":
        screen_scenarios(analysis, params)
    else:
        screen_decision(analysis, params, source)


main()
