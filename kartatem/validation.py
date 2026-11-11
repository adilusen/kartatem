"""Чтение и проверка CSV. Здесь нет Streamlit — только pandas.

Правила (коротко):
* Структурные ошибки (нет столбцов, пустые идентификаторы, неизвестный stage или тема,
  конфликтующие дубли) БЛОКИРУЮТ анализ: clean будет пустым, errors — заполнен.
* Пропущенные/невалидные баллы НЕ блокируют: такие строки исключаются и считаются
  в отчёте. Пропуск никогда не превращается в ноль.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RESULTS_COLUMNS = ["student_id", "class_id", "textbook", "topic_id", "stage",
                   "test_id", "score", "max_score"]
TOPICS_COLUMNS = ["topic_id", "name", "hours", "min_hours", "max_delta"]

ID_COLUMNS = ["student_id", "class_id", "textbook", "topic_id", "test_id"]

STAGE_MAP = {
    "pre": "pre", "pretest": "pre", "pre-test": "pre", "pre_test": "pre",
    "вход": "pre", "входной": "pre", "до": "pre",
    "post": "post", "posttest": "post", "post-test": "post", "post_test": "post",
    "итог": "post", "итоговый": "post", "после": "post",
}

REASON_DUP_EXACT = "Точный дубль (оставлена одна копия)"
REASON_SCORE_EMPTY = "Пропущен score"
REASON_SCORE_BAD = "score не является конечным числом"
REASON_MAX_EMPTY = "Пропущен max_score"
REASON_MAX_BAD = "max_score не является конечным числом"
REASON_MAX_NONPOS = "max_score ≤ 0"
REASON_SCORE_NEG = "score < 0"
REASON_SCORE_OVER = "score > max_score"


# ----------------------------------------------------------------------------
# Чтение файла
# ----------------------------------------------------------------------------
def read_csv_bytes(data: bytes | None) -> tuple[pd.DataFrame | None, str | None]:
    """Безопасно читает CSV из байтов. Возвращает (таблица, текст_ошибки)."""
    if data is None or len(data.strip()) == 0:
        return None, "Файл пуст."
    text = None
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return None, "Не удалось определить кодировку. Сохраните файл как CSV в UTF-8."
    if not any(ln.strip() for ln in text.splitlines()):
        return None, "Файл пуст."
    sep, header = _detect_header(text)
    dup = duplicate_header_errors(header, "файле")
    if dup:
        return None, " ".join(dup)
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return None, "В файле нет данных."
    except (pd.errors.ParserError, ValueError) as exc:
        return None, f"Не удалось разобрать CSV: {exc}"
    df.columns = [_norm_name(c) for c in df.columns]
    return df, None


# ----------------------------------------------------------------------------
# Вспомогательное
# ----------------------------------------------------------------------------

def _norm_name(c) -> str:
    """Нормализованное имя столбца: без BOM и пробелов по краям, в нижнем регистре."""
    return str(c).replace("\ufeff", "").strip().lower()


def _header_record(text: str, sep: str) -> list[str]:
    """Первая непустая запись CSV, разобранная с учётом кавычек."""
    for row in csv.reader(io.StringIO(text), delimiter=sep):
        if any(cell.strip() for cell in row):
            return row
    return []


def _detect_header(text: str) -> tuple[str, list[str]]:
    """Разделитель — тот, при котором заголовок (с учётом кавычек) даёт больше всего столбцов."""
    best_sep, best = ",", _header_record(text, ",")
    for sep in (";", "\t"):
        rec = _header_record(text, sep)
        if len(rec) > len(best):
            best_sep, best = sep, rec
    return best_sep, best


def duplicate_header_errors(columns, file_label: str) -> list[str]:
    """Блокирующие ошибки, если после нормализации (регистр, пробелы) имена столбцов повторяются.

    Проверка выполняется ДО обращения к столбцам и до того, как pandas сам переименует повторы
    (score → score.1). Пустые имена не считаются.
    """
    groups: dict[str, list[str]] = {}
    for c in columns:
        n = _norm_name(c)
        if n:
            groups.setdefault(n, []).append(str(c))
    return [
        f"В {file_label} повторяется заголовок столбца «{n}»: " + ", ".join(f"«{o}»" for o in originals)
        + " — после приведения к нижнему регистру и удаления пробелов это один и тот же столбец. "
        "Оставьте один."
        for n, originals in groups.items() if len(originals) > 1
    ]


def make_uid(class_id: str, student_id: str) -> str:
    """Однозначный ключ ученика (class_id + student_id).

    JSON-сериализация пары не даёт коллизий: пары («X::Y», «Z») и («X», «Y::Z») —
    разные ключи, в отличие от простой склейки через разделитель.
    """
    return json.dumps([class_id, student_id], ensure_ascii=False)

def _clean_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _parse_number(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(число как float, признак «пусто»). Запятая допускается как десятичный разделитель."""
    s = _clean_str(raw)
    empty = s == ""
    num = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce").astype(float)
    return num, empty


def _sample(values, limit: int = 5) -> str:
    vals = [str(v) for v in list(values)[:limit]]
    extra = len(list(values)) - len(vals)
    return ", ".join(vals) + (f" и ещё {extra}" if extra > 0 else "")


def _norm_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_norm_name(c) for c in out.columns]
    return out


# ----------------------------------------------------------------------------
# topics.csv
# ----------------------------------------------------------------------------
@dataclass
class TopicsValidation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clean: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TOPICS_COLUMNS))

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_topics(df: pd.DataFrame | None) -> TopicsValidation:
    res = TopicsValidation()
    if df is None:
        res.errors.append("Файл topics.csv не прочитан.")
        return res
    dup = duplicate_header_errors(df.columns, "topics.csv")
    if dup:
        res.errors.extend(dup)
        return res
    df = _norm_headers(df)
    missing = [c for c in TOPICS_COLUMNS if c not in df.columns]
    if missing:
        res.errors.append("В topics.csv нет столбцов: " + ", ".join(missing) + ".")
        return res
    if len(df) == 0:
        res.errors.append("В topics.csv нет ни одной темы.")
        return res

    d = df[TOPICS_COLUMNS].copy()
    for c in TOPICS_COLUMNS:
        d[c] = _clean_str(d[c])

    empty_ids = d.index[d["topic_id"] == ""]
    if len(empty_ids):
        res.errors.append(f"Пустой topic_id в записях №{_sample([i + 1 for i in empty_ids])}.")
    dups = d.loc[(d["topic_id"] != "") & d["topic_id"].duplicated(keep=False), "topic_id"].unique()
    if len(dups):
        res.errors.append("topic_id должен быть уникальным. Повторяются: " + _sample(dups) + ".")

    nums = {}
    for c in ("hours", "min_hours", "max_delta"):
        num, empty = _parse_number(d[c])
        nums[c] = num
        bad = ~np.isfinite(num.to_numpy())
        if bad.any():
            res.errors.append(f"Столбец {c}: нужны конечные числа. Проблема в темах: "
                              + _sample(d.loc[bad, "topic_id"].replace("", "(пусто)")) + ".")
            continue
        non_int = (num != np.floor(num)).to_numpy()
        if non_int.any():
            res.errors.append(f"Столбец {c}: часы должны быть целыми. Проблема в темах: "
                              + _sample(d.loc[non_int, "topic_id"]) + ".")
        neg = (num < 0).to_numpy()
        if neg.any():
            res.errors.append(f"Столбец {c}: значения не могут быть отрицательными. "
                              "Проблема в темах: " + _sample(d.loc[neg, "topic_id"]) + ".")

    if not res.errors:
        over = (nums["min_hours"] > nums["hours"]).to_numpy()
        if over.any():
            res.errors.append("min_hours больше hours в темах: " + _sample(d.loc[over, "topic_id"]) + ".")

    noname = d["name"] == ""
    if noname.any():
        res.warnings.append("У некоторых тем пустое название — будет показан topic_id.")
        d.loc[noname, "name"] = d.loc[noname, "topic_id"]

    if res.errors:
        return res
    clean = d.copy()
    for c in ("hours", "min_hours", "max_delta"):
        clean[c] = nums[c].astype(int)
    res.clean = clean.reset_index(drop=True)
    return res


# ----------------------------------------------------------------------------
# results.csv
# ----------------------------------------------------------------------------
@dataclass
class ResultsValidation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clean: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    excluded: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["record_no", "reason"] + RESULTS_COLUMNS))
    exclusion_counts: dict[str, int] = field(default_factory=dict)
    n_input_rows: int = 0
    changed_textbook_uids: set = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def n_valid_rows(self) -> int:
        return int(len(self.clean))

    @property
    def n_excluded_rows(self) -> int:
        return int(sum(self.exclusion_counts.values()))


def validate_results(df: pd.DataFrame | None,
                     known_topic_ids: set[str] | None = None) -> ResultsValidation:
    res = ResultsValidation()
    if df is None:
        res.errors.append("Файл results.csv не прочитан.")
        return res
    dup = duplicate_header_errors(df.columns, "results.csv")
    if dup:
        res.errors.extend(dup)
        return res
    df = _norm_headers(df)
    missing = [c for c in RESULTS_COLUMNS if c not in df.columns]
    if missing:
        res.errors.append("В results.csv нет столбцов: " + ", ".join(missing) + ".")
        return res
    extra = [c for c in df.columns if c not in RESULTS_COLUMNS]
    if extra:
        res.warnings.append("Лишние столбцы проигнорированы: " + ", ".join(extra) + ".")
    res.n_input_rows = int(len(df))
    if len(df) == 0:
        res.errors.append("В results.csv нет ни одной записи (есть только заголовок).")
        return res

    d = df[RESULTS_COLUMNS].copy().reset_index(drop=True)
    for c in RESULTS_COLUMNS:
        d[c] = _clean_str(d[c])
    d["record_no"] = np.arange(1, len(d) + 1)

    # --- stage ---
    d["stage_norm"] = d["stage"].str.lower().map(STAGE_MAP).fillna("")
    d["stage_norm"] = d["stage_norm"].astype(str)

    # --- блокирующие ошибки: идентификаторы, stage, темы ---
    for c in ID_COLUMNS:
        empty = d["record_no"][d[c] == ""]
        if len(empty):
            res.errors.append(f"Пустое значение {c} в записях №{_sample(empty)}.")
    bad_stage = d[d["stage_norm"] == ""]
    if len(bad_stage):
        shown = bad_stage["stage"].replace("", "(пусто)").unique()
        res.errors.append(
            f"Недопустимый stage ({_sample(shown)}) в записях №{_sample(bad_stage['record_no'])}. "
            "Допустимо: pre или post.")
    if known_topic_ids is not None:
        unknown = d[(d["topic_id"] != "") & ~d["topic_id"].isin(known_topic_ids)]
        if len(unknown):
            res.errors.append(
                f"Неизвестные темы ({_sample(unknown['topic_id'].unique())}) в записях "
                f"№{_sample(unknown['record_no'])}: их нет в topics.csv.")

    # --- числа ---
    score, score_empty = _parse_number(d["score"])
    maxs, max_empty = _parse_number(d["max_score"])
    d["score_num"], d["max_num"] = score, maxs
    s_fin, m_fin = np.isfinite(score.to_numpy()), np.isfinite(maxs.to_numpy())
    both = s_fin & m_fin
    reason = np.select(
        [
            score_empty.to_numpy(),
            ~score_empty.to_numpy() & ~s_fin,
            max_empty.to_numpy(),
            ~max_empty.to_numpy() & ~m_fin,
            m_fin & (maxs.to_numpy() <= 0),
            both & (score.to_numpy() < 0),
            both & (score.to_numpy() > maxs.to_numpy()),
        ],
        [REASON_SCORE_EMPTY, REASON_SCORE_BAD, REASON_MAX_EMPTY, REASON_MAX_BAD,
         REASON_MAX_NONPOS, REASON_SCORE_NEG, REASON_SCORE_OVER],
        default="",
    )
    d["reason"] = reason

    # --- дубли: ключ = class_id + student_id + topic_id + stage ---
    key = ["class_id", "student_id", "topic_id", "stage_norm"]
    keyed = d[(d[ID_COLUMNS] != "").all(axis=1) & (d["stage_norm"] != "")].copy()

    def _cmp(num: pd.Series, raw: pd.Series) -> list[str]:
        return [repr(float(n)) if np.isfinite(n) else r for n, r in zip(num, raw)]

    keyed["score_cmp"] = _cmp(keyed["score_num"], keyed["score"])
    keyed["max_cmp"] = _cmp(keyed["max_num"], keyed["max_score"])
    payload = key + ["textbook", "test_id", "score_cmp", "max_cmp"]
    exact_dup_mask = keyed.duplicated(subset=payload, keep="first")
    exact_dup_records = set(keyed.loc[exact_dup_mask, "record_no"])
    deduped = keyed[~exact_dup_mask]
    conflicts = deduped[deduped.duplicated(subset=key, keep=False)]
    if len(conflicts):
        examples = (conflicts.groupby(key).size().reset_index()
                    .head(5).apply(lambda r: f"{r['class_id']}/{r['student_id']}/{r['topic_id']}/{r['stage_norm']}",
                                   axis=1).tolist())
        res.errors.append(
            f"Конфликтующие дубли: одна и та же запись (класс + ученик + тема + stage) встречается "
            f"с разными значениями ({_sample(examples)}). Мы не усредняем их молча — исправьте файл.")
    if exact_dup_records:
        res.warnings.append(f"Удалено точных дублей: {len(exact_dup_records)} (оставлена одна копия).")

    # --- ученики, сменившие учебник ---
    d["uid"] = [make_uid(c, st) for c, st in zip(d["class_id"], d["student_id"])]
    ident_ok = d[(d[ID_COLUMNS] != "").all(axis=1) & ~d["record_no"].isin(exact_dup_records)]
    tb_counts = ident_ok.groupby("uid")["textbook"].nunique()
    res.changed_textbook_uids = set(tb_counts[tb_counts > 1].index)
    if res.changed_textbook_uids:
        res.warnings.append(
            f"Учеников, у которых в данных меняется учебник: {len(res.changed_textbook_uids)}. "
            "Они не включаются в парное сравнение учебников.")
    cls_tb = ident_ok.groupby("class_id")["textbook"].nunique()
    if (cls_tb > 1).any():
        res.warnings.append("Есть классы, где встречаются разные учебники: "
                            + _sample(cls_tb[cls_tb > 1].index) + ".")

    # --- исключённые записи (для отчёта) ---
    excl_rows = d[(d["reason"] != "") | d["record_no"].isin(exact_dup_records)].copy()
    excl_rows.loc[excl_rows["record_no"].isin(exact_dup_records), "reason"] = REASON_DUP_EXACT
    res.excluded = excl_rows[["record_no", "reason"] + RESULTS_COLUMNS].reset_index(drop=True)
    res.exclusion_counts = excl_rows["reason"].value_counts().to_dict()
    score_reasons = {k: v for k, v in res.exclusion_counts.items() if k != REASON_DUP_EXACT}
    if score_reasons:
        res.warnings.append(
            f"Исключено записей с невалидным баллом: {sum(score_reasons.values())} "
            "(пропуски не заменяются нулями).")

    if res.errors:
        return res

    clean = d[(d["reason"] == "") & ~d["record_no"].isin(exact_dup_records)].copy()
    if clean.empty:
        res.errors.append("В файле нет ни одной валидной записи: после проверок не осталось строк для анализа.")
        return res
    clean["stage"] = clean["stage_norm"]
    clean["score"] = clean["score_num"]
    clean["max_score"] = clean["max_num"]
    clean["pct"] = 100.0 * clean["score"] / clean["max_score"]
    res.clean = clean[["record_no", "uid", "student_id", "class_id", "textbook", "topic_id",
                       "stage", "test_id", "score", "max_score", "pct"]].reset_index(drop=True)
    return res
