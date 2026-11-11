"""Цвета и подписи оформления (без Streamlit).

Цвета учебников постоянны во всех разделах: определяются по положению названия в
отсортированном списке ВСЕХ учебников набора данных, а не по порядку внутри выбранной пары.
Красный используется только для точечных индикаторов трудности и ошибок.
"""
from __future__ import annotations

from typing import Iterable

BG = "#F7F8FA"
SURFACE = "#FFFFFF"
TEXT = "#14213D"
MUTED = "#5B6478"
ACCENT = "#2F5BEA"
DANGER = "#B42318"
BORDER = "#E3E7EE"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# (цвет для графиков, название цвета для st.badge)
TEXTBOOK_PALETTE = [
    ("#D97706", "orange"),     # кобальт #2F5BEA оставлен акценту интерфейса
    ("#0F9488", "green"),
    ("#7C3AED", "violet"),
    ("#475569", "gray"),
    ("#B45309", "yellow"),
]


def textbook_colors(textbooks: Iterable[str]) -> dict[str, tuple[str, str]]:
    """{учебник: (hex, цвет значка)}. Устойчиво к порядку в паре и к выбору пары."""
    names = sorted(set(textbooks))
    return {n: TEXTBOOK_PALETTE[i % len(TEXTBOOK_PALETTE)] for i, n in enumerate(names)}
