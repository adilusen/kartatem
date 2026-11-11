"""Сохранение и восстановление плана в JSON (без поученичных данных, без eval/pickle).

Файл хранит версию формата, контекст данных, настройки (параметры и выбор форм),
фиксации, черновик и сценарии. Поле утверждения сохраняется только для сведения:
при восстановлении ему НЕ доверяют — восстановленный план становится черновиком.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields

from .config import Params, validate_params
from .hours import validate_plan
from .scenarios import MAX_NAME, MAX_SCENARIOS, Scenario, clean_name

FORMAT = "olshem-plan"
VERSION = 1
MAX_BYTES = 200_000


def plan_to_json(*, context: str, context_parts: dict, params: Params, test_choice: dict,
                 locked: list, draft: dict, scenarios: list[Scenario], approved: dict | None,
                 topics) -> bytes:
    doc = {
        "format": FORMAT, "version": VERSION,
        "product": "Өлшем",
        "context": {"id": context, "source": context_parts["source"],
                    "forms": context_parts["forms"],
                    "topics": [{"topic_id": r.topic_id, "hours": int(r.hours)} for r in topics.itertuples(index=False)]},
        "settings": {"params": asdict(params), "test_choice": dict(test_choice)},
        "locked": sorted(locked),
        "draft": {k: int(v) for k, v in draft.items()},
        "scenarios": [s.to_dict() for s in scenarios],
        "approved_info": approved,   # только для сведения, не восстанавливается как утверждение
        "note": "Синтетические или загруженные данные в файл не входят: нужны те же исходные CSV.",
    }
    return json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")


@dataclass
class Restored:
    params: Params
    test_choice: dict
    locked: list
    draft: dict
    scenarios: list
    context: str


MAX_HOURS = 10_000          # верхняя граница «разумных» часов: большие числа — ошибка, а не OverflowError
CTX_RE = re.compile(r"^[0-9a-f]{12}$")


def _is_int_value(v) -> bool:
    """Целое конечное число в разумных пределах (bool — не число). Без float(v) для огромных int."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return abs(v) <= MAX_HOURS
    if isinstance(v, float):
        return math.isfinite(v) and abs(v) <= MAX_HOURS and v == int(v)
    return False


def _hours(obj, where: str, known: set, errors: list, *, complete: bool) -> dict | None:
    if not isinstance(obj, dict):
        errors.append(f"{where}: ожидается объект «тема → часы».")
        return None
    out, ok = {}, True
    for k, v in obj.items():
        if k not in known:
            errors.append(f"{where}: неизвестная тема «{k}».")
            ok = False
        elif not _is_int_value(v):
            errors.append(f"{where}: тема «{k}» — часы должны быть целым числом от 0 до {MAX_HOURS}, "
                          f"получено {str(v)[:30]!r}.")
            ok = False
        else:
            out[k] = int(v)
    if complete and ok and set(out) != known:
        errors.append(f"{where}: указаны не все темы (нет: {', '.join(sorted(known - set(out)))}).")
        ok = False
    return out if ok else None


def _str_list(obj, where: str, errors: list) -> list | None:
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        errors.append(f"{where}: ожидается список строковых topic_id.")
        return None
    if len(set(obj)) != len(obj):
        errors.append(f"{where}: темы повторяются.")
        return None
    return list(obj)


def _reject_constant(name):
    raise ValueError(f"недопустимое значение {name}")


def parse_plan(raw: bytes, *, source_sig: str, topics) -> tuple[dict | None, list[str]]:
    """Шаг 1: разбор и проверка структуры, совпадения исходных данных и тем."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_BYTES:
        return None, [f"Файл больше {MAX_BYTES // 1000} КБ — это не файл плана Өлшем."]
    try:
        doc = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
    except UnicodeDecodeError:
        return None, ["Файл не в кодировке UTF-8: это не файл плана Өлшем."]
    except RecursionError:
        return None, ["Слишком глубокая вложенность JSON: это не файл плана Өлшем."]
    except ValueError as e:          # json.JSONDecodeError — подкласс ValueError; NaN/Infinity отклоняются здесь же
        return None, [f"Файл не является корректным JSON: {e}."]
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        return None, ["Это не файл плана Өлшем (нет признака формата «olshem-plan»)."]
    if doc.get("version") != VERSION or isinstance(doc.get("version"), bool):
        return None, [f"Неподдерживаемая версия формата: {str(doc.get('version'))[:20]!r} (ожидается {VERSION})."]
    ctx = doc.get("context")
    if not isinstance(ctx, dict):
        return None, ["В файле нет раздела context."]
    errors: list[str] = []
    if not isinstance(ctx.get("id"), str) or not CTX_RE.match(ctx["id"]):
        errors.append("context.id отсутствует или записан неверно.")
    if not isinstance(ctx.get("source"), str):
        errors.append("context.source отсутствует или записан неверно.")
    elif ctx["source"] != source_sig:
        errors.append("План сохранён для других исходных данных. Загрузите те же results.csv и topics.csv "
                      "(для демо — откройте то же демо), затем восстановите план.")
    file_topics = ctx.get("topics")
    if (not isinstance(file_topics, list)
            or not all(isinstance(t, dict) and isinstance(t.get("topic_id"), str) for t in file_topics)):
        errors.append("context.topics: ожидается список объектов со строковым topic_id.")
    else:
        ids = [t["topic_id"] for t in file_topics]
        if len(set(ids)) != len(ids):
            errors.append("context.topics: topic_id повторяются.")
        elif set(ids) != set(topics["topic_id"]):
            errors.append("Набор тем в плане не совпадает с текущим topics.csv.")
    return (doc if not errors else None), errors


def _params(raw_p, errors: list) -> Params | None:
    if not isinstance(raw_p, dict):
        errors.append("В файле нет параметров методики.")
        return None
    defaults = Params()
    names = {f.name for f in fields(Params)}
    extra = sorted(str(k) for k in raw_p if k not in names)
    if extra:
        errors.append("Неизвестные параметры: " + ", ".join(extra) + ".")
        return None
    vals = {}
    for name in names:
        v, d = raw_p.get(name, getattr(defaults, name)), getattr(defaults, name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"Параметры: {name} должен быть числом.")
            continue
        if isinstance(v, int) and abs(v) > 10**9 or isinstance(v, float) and not math.isfinite(v):
            errors.append(f"Параметры: {name} вне допустимого диапазона.")
            continue
        if isinstance(d, int):
            if v != int(v):
                errors.append(f"Параметры: {name} должен быть целым.")
                continue
            vals[name] = int(v)
        else:
            vals[name] = float(v)
    if len(vals) != len(names):
        return None
    p = Params(**vals)
    probs = validate_params(p)
    errors += [f"Параметры: {e}" for e in probs]
    return None if probs else p


def restore_plan(doc: dict, *, topics) -> tuple[Restored | None, list[str]]:
    """Шаг 2: настройки, фиксации, черновик и сценарии. Утверждение не восстанавливается.

    Каждый сценарий сохраняет СВОЙ контекст: сценарий другого контекста остаётся «старым»
    и после восстановления — восстановление не является пересчётом."""
    errors: list[str] = []
    known = set(topics["topic_id"])
    st = doc.get("settings")
    if not isinstance(st, dict):
        return None, ["В файле нет раздела settings."]
    params = _params(st.get("params"), errors)
    tc = st.get("test_choice", {})
    if not isinstance(tc, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in tc.items()):
        errors.append("Выбор форм теста записан неверно.")
        tc = {}
    elif set(tc) - known:
        errors.append("Выбор формы для неизвестных тем: " + ", ".join(sorted(set(tc) - known)) + ".")
    locked = _str_list(doc.get("locked", []), "Фиксации", errors)
    if locked is not None and set(locked) - known:
        errors.append("Фиксация для неизвестных тем: " + ", ".join(sorted(set(locked) - known)) + ".")
        locked = None
    draft = _hours(doc.get("draft"), "Черновик", known, errors, complete=True)
    if draft is not None and locked is not None:
        errors += [f"Черновик: {e}" for e in validate_plan(topics, draft, locked)]
    scs = []
    raw_s = doc.get("scenarios", [])
    if not isinstance(raw_s, list) or len(raw_s) > MAX_SCENARIOS:
        errors.append(f"Сценарии записаны неверно (ожидается список не больше {MAX_SCENARIOS}).")
        raw_s = []
    seen = set()
    for i, s in enumerate(raw_s, 1):
        if not isinstance(s, dict):
            errors.append(f"Сценарий {i}: ожидается объект.")
            continue
        name = clean_name(s["name"]) if isinstance(s.get("name"), str) else ""
        if not name:
            errors.append(f"Сценарий {i}: нет названия.")
            continue
        if name in seen:
            errors.append(f"Сценарий «{name}»: название повторяется — сценарии должны различаться по имени.")
            continue
        seen.add(name)
        ctx = s.get("context")
        if not isinstance(ctx, str) or not CTX_RE.match(ctx):
            errors.append(f"Сценарий «{name}»: контекст данных отсутствует или записан неверно.")
            continue
        h = _hours(s.get("hours"), f"Сценарий «{name}»", known, errors, complete=True)
        lk = _str_list(s.get("locked", []), f"Сценарий «{name}»: фиксации", errors)
        if h is None or lk is None:
            continue
        if set(lk) - known:
            errors.append(f"Сценарий «{name}»: фиксация для неизвестных тем.")
            continue
        errors += [f"Сценарий «{name}»: {e}" for e in validate_plan(topics, h, lk)]
        origin = s.get("origin") if s.get("origin") in ("manual", "algorithm", "restored") else "restored"
        scs.append(Scenario(name, h, sorted(lk), ctx, origin))
    if errors or params is None or draft is None or locked is None:
        return None, errors or ["План записан неверно."]
    return Restored(params, dict(tc), sorted(locked), draft, scs, str(doc["context"]["id"])), []
