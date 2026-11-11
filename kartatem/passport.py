"""Паспорт решения: самостоятельный HTML для чтения и печати в браузере (без Streamlit).

Все значения из данных и пользовательские названия экранируются (html.escape).
Документ показывает только текущий сценарий; неутверждённый план явно помечен.
"""
from __future__ import annotations

import html
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from .config import PARAM_UI, Params
from .hours import validate_plan
from .labels import topic_labels

E_ = html.escape

LIMITS = [
    "Результат — средний процент итогового теста (post) от максимума; это результат теста, а не доказанный уровень освоения.",
    "Прирост = post − pre у одних и тех же учеников (парная выборка) — описательное изменение, не причинный эффект.",
    "Сравнение учебников описательное: группы не случайны, исходный уровень не корректируется, доверительных интервалов нет.",
    "Пороги — настраиваемые допущения, не нормативы и не научные критерии.",
    "Распределение часов — сценарий, а не прогноз роста баллов. Решение принимает методист.",
]

CSS = """
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif;color:#14213D;background:#F7F8FA;margin:0}
main{max-width:960px;margin:0 auto;padding:32px 28px;background:#fff}
h1{font-size:26px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid #E3E7EE}
.muted{color:#5B6478;font-size:13px} .logo{font-weight:700;letter-spacing:.02em;color:#2F5BEA}
.status{padding:10px 14px;border-radius:6px;font-weight:600;margin:14px 0}
.ok{background:#EAF0FE;color:#1E3A8A} .no{background:#FDF2E9;color:#8A3B00}
.syn{display:inline-block;background:#FDF2E9;color:#8A3B00;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{border-bottom:1px solid #E3E7EE;padding:6px 8px;text-align:left;vertical-align:top}
th{color:#5B6478;font-weight:600} td.n{text-align:right;font-variant-numeric:tabular-nums}
ul{margin:6px 0 0 18px;padding:0} li{margin:3px 0}
@media print{body{background:#fff} main{padding:0} .noprint{display:none}}
"""


def _table(headers, rows, numeric=()) -> str:
    h = "".join(f"<th>{E_(str(x))}</th>" for x in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td class='n'>{E_(str(v))}</td>" if i in numeric else f"<td>{E_(str(v))}</td>"
            for i, v in enumerate(r)) + "</tr>"
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def build_passport(analysis, params: Params, *, source_text: str, synthetic: bool,
                   scenario_name: str, hours: dict, locked: list, approved: bool,
                   context: str, generated: datetime | None = None) -> str:
    topics = analysis.topics
    lab = topic_labels(topics)
    plan = analysis.plan_for(locked)                       # предложение алгоритма при этих фиксациях
    errors = validate_plan(topics, hours, locked)
    if analysis.unresolved:
        approved = False
    when = (generated or datetime.now()).strftime("%d.%m.%Y %H:%M")

    if approved:
        status = f"<div class='status ok'>Утверждён методистом: сценарий «{E_(scenario_name)}»</div>"
    else:
        why = ""
        if analysis.unresolved:
            why = " Не выбрана форма теста для тем: " + E_(", ".join(lab[t] for t in analysis.unresolved)) + "."
        if errors:
            why += " Черновик некорректен: " + E_(" ".join(errors))
        status = (f"<div class='status no'>НЕ УТВЕРЖДЁН — черновик «{E_(scenario_name)}».{why}</div>")

    tt = analysis.topic_table.set_index("topic_id")
    topic_rows = []
    for tid in topics["topic_id"]:
        t = tt.loc[tid]
        topic_rows.append([lab[tid], t["selected_test"] or "не выбрана",
                           "—" if pd.isna(t["mean_post"]) else f"{t['mean_post']:.1f}",
                           "—" if pd.isna(t["share_below"]) else f"{t['share_below']:.1f}",
                           int(t["n_students"]), t["status"], t["reason"]])
    pt = plan.table
    hour_rows = []
    for r in pt.itertuples(index=False):
        h = int(hours.get(r.topic_id, r.hours_before))
        note = r.note
        if h != int(r.hours_proposed):
            note += "; часы изменены методистом вручную"
        hour_rows.append([lab[r.topic_id], int(r.hours_before), int(r.hours_proposed), h,
                          f"{h - int(r.hours_before):+d}" if h != r.hours_before else "0",
                          "да" if r.topic_id in locked else "—", f"{r.hours_low}–{r.hours_high}", note])
    total = sum(int(hours.get(t, 0)) for t in topics["topic_id"])
    moves = "".join(f"<li>Шаг {m.step}: {E_(m.reason)}</li>" for m in plan.moves) or "<li>Переносов нет.</li>"
    msgs = "".join(f"<li>{E_(m)}</li>" for m in plan.messages)
    pvals = asdict(params)
    prow = [[label, f"{pvals[name]:g}"] for name, label, *_ in PARAM_UI]
    s = analysis.summary
    syn = "<span class='syn'>Синтетические демо-данные</span> " if synthetic else ""
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Өлшем — паспорт решения</title>
<style>{CSS}</style></head><body><main>
<div class="logo">Өлшем</div>
<h1>Паспорт решения по учебным часам</h1>
<div class="muted">Сформирован {E_(when)} · контекст данных {E_(context)}</div>
{status}
<p>{syn}{E_(source_text)}</p>
<p class="muted">В файле {s['rows']} валидных записей и {s['students']} учеников; в расчёте {s['rows_used']} записей
(выбранные формы теста). Исключено при проверке: {analysis.results_val.n_excluded_rows}.</p>
<p class="noprint muted">Чтобы сохранить PDF, используйте печать браузера (Cmd/Ctrl + P → «Сохранить как PDF»).</p>
<h2>Часы: исходно, предложение, сценарий</h2>
<p>Бюджет: <b>{int(plan.budget)} ч</b>; сумма сценария: <b>{total} ч</b>.
Фиксации: {E_(", ".join(lab[t] for t in topics["topic_id"] if t in locked) or "нет")}.</p>
{_table(["Тема", "Исходно, ч", "Предложено, ч", "Сценарий, ч", "Изменение", "Не изменять", "Допустимо", "Пояснение"], hour_rows, numeric=(1, 2, 3, 4))}
<h2>Причины переносов</h2><ul>{moves}</ul>
{f"<h2>Почему других переносов нет</h2><ul>{msgs}</ul>" if msgs else ""}
<h2>Показатели тем</h2>
{_table(["Тема", "Форма теста", "Средний post, %", f"Доля ниже {params.low_score_cut:g}%, %", "n учеников", "Статус", "Причина"], topic_rows, numeric=(2, 3, 4))}
<h2>Параметры методики</h2>{_table(["Параметр", "Значение"], prow, numeric=(1,))}
<h2>Ограничения</h2><ul>{"".join(f"<li>{E_(x)}</li>" for x in LIMITS)}</ul>
</main></body></html>"""
