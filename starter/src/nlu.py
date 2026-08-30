from __future__ import annotations

import re

from starter.src.config import MATERIALS
from starter.src.interfaces import Constraint, NLUResult, SessionState

_MATTERS_RE = re.compile(r"what matters is:\s*(.+)\.", re.I)
_KEY_REQ_RE = re.compile(r"key requirement is:\s*(.+)\.", re.I)
_OVERRIDE_RE = re.compile(r"(?:ignore my earlier preference|forget what I said).*?(?:What I need is|now I want|instead):\s*(.+)\.", re.I)
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
    if any(w in lowered for w in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(w in lowered for w in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(w in lowered for w in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(w in lowered for w in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


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
            constraints.append(Constraint(
                raw_text=raw,
                attribute_type=classify_constraint(raw),
                value=raw,
                turn_received=turn,
            ))

        boundary_m = _BOUNDARY_RE.search(user_message)
        if boundary_m:
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
                parts = matters_m.group(1).split(";")
                for part in parts:
                    raw = part.strip().rstrip(".")
                    if raw:
                        constraints.append(Constraint(
                            raw_text=raw,
                            attribute_type=classify_constraint(raw),
                            value=raw,
                            turn_received=turn,
                        ))

            key_req_m = _KEY_REQ_RE.search(user_message)
            if key_req_m:
                raw = key_req_m.group(1).strip().rstrip(".")
                if raw:
                    constraints.append(Constraint(
                        raw_text=raw,
                        attribute_type=classify_constraint(raw),
                        value=raw,
                        turn_received=turn,
                    ))

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
