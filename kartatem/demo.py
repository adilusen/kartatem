"""Синтетические демо-данные (генератор с фиксированным seed).

ЭТО НЕ РЕАЛЬНЫЕ РЕЗУЛЬТАТЫ ШКОЛ. Сценарий задан заранее: у «Учебник Б» выше входной
и итоговый результат, а у «Учебник А» больше средний прирост. Это условие демонстрации,
а не результат исследования. Темы — условный демонстрационный план, не официальная программа.

Запуск как скрипт перезаписывает CSV в папке data/:
    python -m kartatem.demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .validation import RESULTS_COLUMNS, TOPICS_COLUMNS

SEED = 42
TEXTBOOKS = ("Учебник А", "Учебник Б")

DEMO_TOPICS = [
    # topic_id, название, часы, min_hours, max_delta
    ("T1", "Обыкновенные дроби", 5, 3, 2),
    ("T2", "Десятичные дроби", 5, 3, 2),
    ("T3", "Проценты", 5, 3, 2),
    ("T4", "Уравнения", 6, 3, 2),
    ("T5", "Пропорции", 4, 3, 2),
    ("T6", "Геометрия: площади", 5, 3, 2),
]
MAX_SCORES = {"T1": 20, "T2": 25, "T3": 20, "T4": 30, "T5": 20, "T6": 25}
TEST_IDS = {t[0]: f"M6-{t[0]}-F1" for t in DEMO_TOPICS}   # одна форма теста на тему (pre, post, обе группы)

# Заданный сценарий: (центр post, post Б − post А, прирост А − прирост Б), п.п.
SCENARIO = {
    "T1": (80, 5, 5),
    "T2": (72, 5, 8),
    "T3": (50, 5, 9),
    "T4": (62, 5, 4),
    "T5": (79, 8, 6),
    "T6": (56, 6, 8),
}
GAIN_A = 16.0          # средний прирост А, п.п.
ABILITY_SD = 9.0       # разброс способностей учеников
NOISE_SD = 6.0         # шум теста
CLASS_SD = 2.0         # различия между классами
BLANK_RATE = 0.02      # доля пропущенных баллов (имитация пропусков в выгрузке)

FULL_SIZES = {"А": [21, 20, 22, 19], "Б": [20, 22, 19, 21]}
SMALL_SIZES = {"А": [8, 8], "Б": [9, 9]}


def topics_frame() -> pd.DataFrame:
    return pd.DataFrame(DEMO_TOPICS, columns=TOPICS_COLUMNS).astype(str)


def _means(topic: str, tb_index: int) -> tuple[float, float]:
    """(средний pre, средний прирост) для учебника (0 = А, 1 = Б) в теме."""
    center, post_gap, gain_diff = SCENARIO[topic]
    post_mean = center - post_gap / 2 if tb_index == 0 else center + post_gap / 2
    gain = GAIN_A if tb_index == 0 else GAIN_A - gain_diff
    return post_mean - gain, gain


def generate_results(seed: int = SEED, small: bool = False) -> pd.DataFrame:
    """Строки results.csv (все столбцы — строки, как после чтения CSV)."""
    rng = np.random.default_rng(seed)
    sizes = SMALL_SIZES if small else FULL_SIZES
    rows = []
    class_no = 0
    for tb_index, letter in enumerate(("А", "Б")):
        textbook = TEXTBOOKS[tb_index]
        for n_students in sizes[letter]:
            class_no += 1
            class_id = f"K{class_no:02d}"
            class_effect = rng.normal(0, CLASS_SD)
            first_class_of_tb = (class_no == 1) or (class_no == len(sizes["А"]) + 1)
            for s in range(1, n_students + 1):
                student_id = f"S{s:02d}"
                ability = rng.normal(0, 1)
                for topic_id, *_ in DEMO_TOPICS:
                    # В малом варианте тест по теме T6 писали только 5 учеников первого класса группы.
                    if small and topic_id == "T6" and not (first_class_of_tb and s <= 5):
                        continue
                    pre_mean, gain_mean = _means(topic_id, tb_index)
                    pre = pre_mean + class_effect + ABILITY_SD * ability + rng.normal(0, NOISE_SD)
                    post = pre + gain_mean + rng.normal(0, NOISE_SD)
                    mx = MAX_SCORES[topic_id]
                    for stage, pct in (("pre", pre), ("post", post)):
                        score = int(np.clip(round(pct / 100 * mx), 0, mx))
                        rows.append([student_id, class_id, textbook, topic_id, stage,
                                     TEST_IDS[topic_id], str(score), str(mx)])
    df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
    blank = rng.random(len(df)) < BLANK_RATE
    df.loc[blank, "score"] = ""      # пропуск остаётся пропуском, а не нулём
    return df


BAD_RESULTS_CSV = """student_id,class_id,textbook,topic_id,stage,test_id,score,max_score
S01,K01,Учебник А,T1,pre,M6-T1-F1,8,20
S01,K01,Учебник А,T1,post,M6-T1-F1,15,20
S02,K01,Учебник А,T1,pre,M6-T1-F1,10,20
S02,K01,Учебник А,T1,post,M6-T1-F1,abc,20
S03,K01,Учебник А,T1,pre,M6-T1-F1,12,0
S03,K01,Учебник А,T1,post,M6-T1-F1,25,20
,K01,Учебник А,T1,pre,M6-T1-F1,9,20
S04,K01,Учебник А,T1,midterm,M6-T1-F1,11,20
S04,K01,Учебник А,T9,post,M6-T9-F1,14,20
S05,K02,Учебник Б,T1,pre,M6-T1-F1,13,20
S05,K02,Учебник Б,T1,pre,M6-T1-F1,17,20
S05,K02,Учебник Б,T1,post,M6-T1-F1,,20
"""
"""Пример испорченного файла: пустой student_id, неизвестный stage и тема, конфликтующий
дубль (S05/K02/T1/pre), а также исключаемые записи (score=abc, max_score=0, score>max, пропуск)."""


def bad_results_frame() -> pd.DataFrame:
    import io
    return pd.read_csv(io.StringIO(BAD_RESULTS_CSV), dtype=str, keep_default_na=False)


def results_template_csv() -> str:
    return ("student_id,class_id,textbook,topic_id,stage,test_id,score,max_score\n"
            "S01,K01,Учебник 1,T1,pre,TEST-T1-F1,8,20\n"
            "S01,K01,Учебник 1,T1,post,TEST-T1-F1,15,20\n")


def topics_template_csv() -> str:
    return ("topic_id,name,hours,min_hours,max_delta\n"
            "T1,Тема 1,5,3,2\n"
            "T2,Тема 2,5,3,2\n")


def write_data_files(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []

    def put(name: str, text: str) -> None:
        p = folder / name
        p.write_text(text, encoding="utf-8")
        out.append(p)

    put("demo_results.csv", generate_results().to_csv(index=False))
    put("demo_results_small.csv", generate_results(small=True).to_csv(index=False))
    put("demo_topics.csv", topics_frame().to_csv(index=False))
    put("bad_results_example.csv", BAD_RESULTS_CSV)
    put("template_results.csv", results_template_csv())
    put("template_topics.csv", topics_template_csv())
    return out


if __name__ == "__main__":
    for path in write_data_files(Path(__file__).resolve().parent.parent / "data"):
        print("записан", path)
