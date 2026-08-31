from __future__ import annotations

import os
import re

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")

MATERIAL_RE = re.compile(r"\b(?:" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(?:" + "|".join(COLORS) + r")\b", re.I)

BM25_WEIGHTS = (0.0, 6.0, 4.0, 4.0, 3.5, 1.5, 0.3)

ASK_STRATEGY_ORDER = [
    "other", "other", "feature", "material", "other",
    "feature", "material", "color", "style", "other",
]

ATTRIBUTE_MESSAGES = {
    "other": "What specific features or requirements matter most to you?",
    "feature": "Are there particular features you need?",
    "material": "Do you have a material preference?",
    "color": "Is there a particular color you're looking for?",
    "style": "Do you have a style preference?",
    "size": "What size are you looking for?",
    "brand": "Do you have a brand preference?",
    "budget": "What's your budget range?",
    "use_case": "What will you be using this for?",
    "category": "What type of product are you looking for?",
}

LLM_ENABLED = os.environ.get("LLM_ENABLED", "false").lower() == "true"
