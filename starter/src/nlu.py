from __future__ import annotations

import re

from starter.src.config import MATERIALS, COLORS
from starter.src.interfaces import Constraint, NLUResult, SessionState

_MATTERS_RE = re.compile(r"what matters is:\s*(.+)\.", re.I | re.DOTALL)
_KEY_REQ_RE = re.compile(r"key requirement is:\s*(.+)\.", re.I)
_OVERRIDE_RE = re.compile(
    r"(?:ignore my earlier preference|forget what I said).*?"
    r"(?:What I need is|now I want|instead):\s*(.+)\.",
    re.I | re.DOTALL,
)
_BOUNDARY_RE = re.compile(r"I don't have a preference for (\w+)", re.I)
_EXHAUSTED_RE = re.compile(r"I don't have an additional preference for (\w+)", re.I)
_LOOKING_FOR_RE = re.compile(r"I'm looking for (.+?)(?:\.|,)", re.I)
_WASTED_RE = re.compile(r"Ask me about one specific attribute", re.I)


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(m in lowered for m in MATERIALS):
        return "material"
    if any(w in lowered for w in ("color",) + COLORS):
        return "color"
    if any(w in lowered for w in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(w in lowered for w in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(w in lowered for w in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _make_constraint(raw: str, turn: int) -> Constraint:
    return Constraint(
        raw_text=raw,
        attribute_type=classify_constraint(raw),
        value=raw,
        turn_received=turn,
    )


class NLUModule:

    def parse(self, user_message: str, turn: int, state: SessionState) -> NLUResult:
        constraints: list[Constraint] = []
        is_override = False
        is_no_preference = False
        is_exhausted = False
        exhausted_attribute = None
        category_text = ""

        override_m = _OVERRIDE_RE.search(user_message)
        if override_m:
            is_override = True
            raw = override_m.group(1).strip().rstrip(".")
            if raw:
                constraints.append(_make_constraint(raw, turn))

        if _BOUNDARY_RE.search(user_message):
            is_no_preference = True

        exhausted_m = _EXHAUSTED_RE.search(user_message)
        if exhausted_m:
            is_exhausted = True
            exhausted_attribute = exhausted_m.group(1).strip().lower()

        if _WASTED_RE.search(user_message):
            is_exhausted = True

        if not is_override:
            matters_m = _MATTERS_RE.search(user_message)
            if matters_m:
                for part in matters_m.group(1).split(";"):
                    raw = part.strip().rstrip(".")
                    if raw:
                        constraints.append(_make_constraint(raw, turn))

            key_req_m = _KEY_REQ_RE.search(user_message)
            if key_req_m:
                raw = key_req_m.group(1).strip().rstrip(".")
                if raw:
                    constraints.append(_make_constraint(raw, turn))

        looking_m = _LOOKING_FOR_RE.search(user_message)
        if looking_m:
            category_text = looking_m.group(1).strip()

        intent = "unknown"
        if turn == 1:
            if "key requirement" in user_message.lower():
                intent = "buying"
            elif "still exploring" in user_message.lower():
                intent = "browsing"
            else:
                intent = "browsing"
        if is_override:
            intent = "override"

        from starter.src.catalog_index import tokenize
        all_text = " ".join(c.value for c in constraints)
        if category_text:
            all_text = category_text + " " + all_text
        raw_terms = tokenize(all_text)

        return NLUResult(
            new_constraints=constraints,
            detected_intent=intent,
            is_override=is_override,
            is_no_preference=is_no_preference,
            is_exhausted=is_exhausted,
            exhausted_attribute=exhausted_attribute,
            raw_query_terms=raw_terms,
            category_text=category_text,
        )

    def generate_message(
        self, attribute: str | None, state: SessionState
    ) -> str:
        if attribute is None:
            return self._recommend_message(state)

        category = state.category_text or "what you're looking for"
        tags = state.user_profile.get("preference_tags", [])
        n_constraints = len(state.constraints)

        if n_constraints == 0:
            prefix = f"To help narrow down {category}"
        elif n_constraints <= 2:
            prefix = "Thanks for that detail"
        else:
            prefix = "That's helpful"

        tag_hint = ""
        if tags and attribute in ("feature", "other"):
            relevant = [t for t in tags if t not in ("fit",)]
            if relevant:
                tag_hint = f" I see you value {relevant[0]}, so"

        questions = {
            "other": f"{prefix}.{tag_hint} what specific features or requirements matter most to you?",
            "feature": f"{prefix}.{tag_hint} are there particular features you need?",
            "material": f"{prefix}. Do you have a material preference?",
            "color": f"{prefix}. Is there a particular color you're looking for?",
            "style": f"{prefix}. Do you have a style or fit preference?",
            "size": f"{prefix}. What size are you looking for?",
            "brand": f"{prefix}. Do you have a brand preference?",
            "budget": f"{prefix}. What's your budget range?",
            "use_case": f"{prefix}. What will you primarily use this for?",
            "category": f"{prefix}. What type of product are you looking for?",
        }
        return questions.get(attribute, f"{prefix}. What else matters to you?")

    def _recommend_message(self, state: SessionState) -> str:
        n = len(state.constraints)
        if n == 0:
            return "Here are some options to get started."
        if n <= 2:
            return "Based on what you've shared, here are my top picks."
        return "I've refined the results based on everything you've told me. Here are my best matches."
