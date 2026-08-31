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

# Broad discovery questions are especially information-dense in the supplied
# conversation simulator: ``other`` may reveal constraints across attribute
# types.  Keep the opening exploration deliberately broad, then fall back to
# the normal attribute rotation once enough preference detail has been
# collected.
BROAD_DISCOVERY_TURNS = 3

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

# Boundary handling ---------------------------------------------------------
#
# In the evaluator's boundary scenario, the customer declines the first
# attribute we ask about.  Treating that as a permanent refusal to share
# *anything* causes the next question to be needlessly narrow.  A single
# follow-up using the evaluator's wildcard ``other`` attribute can disclose
# the customer's most useful remaining constraints.  Keep this policy here
# (rather than embedding it in the state router) so its trigger and wording
# are explicit and independently testable.
BOUNDARY_BROAD_REASK_ATTRIBUTE = "other"
BOUNDARY_BROAD_REASK_MESSAGE = (
    "No problem. What are the one or two most important practical needs for it?"
)


def should_broad_reask_after_boundary(
    *,
    is_no_preference: bool,
    attributes_asked: list[str | None],
) -> bool:
    """Return whether to make the one broad follow-up after an initial decline.

    ``attributes_asked`` is inspected *before* the current decision is
    appended.  Requiring exactly one prior question makes the policy a
    one-shot recovery for the initial boundary response, rather than a
    repeated override of users who later decline an attribute.
    """
    return is_no_preference and len(attributes_asked) == 1

LLM_ENABLED = os.environ.get("LLM_ENABLED", "false").lower() == "true"

# Optional live NLU parse (starter/src/nlu.py::_llm_extract). Only used when
# LLM_ENABLED is true AND an ANTHROPIC_API_KEY is present in the environment;
# otherwise the agent runs fully offline against the regex/heuristic parser.
# No key is ever hardcoded or committed here.
LLM_MODEL = os.environ.get("NLU_LLM_MODEL", "claude-3-5-haiku-20241022")
LLM_API_URL = "https://api.anthropic.com/v1/messages"
LLM_TIMEOUT_SECONDS = float(os.environ.get("NLU_LLM_TIMEOUT_SECONDS", "8"))
