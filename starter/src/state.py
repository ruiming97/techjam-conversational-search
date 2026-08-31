from __future__ import annotations

import re
from typing import Optional

from starter.src.config import (
    ASK_STRATEGY_ORDER,
    ATTRIBUTE_MESSAGES,
    BOUNDARY_BROAD_REASK_ATTRIBUTE,
    BOUNDARY_BROAD_REASK_MESSAGE,
    should_broad_reask_after_boundary,
)
from starter.src.interfaces import (
    ALLOWED_ATTRIBUTES,
    Constraint,
    NLUResult,
    SessionState,
    StrategyDecision,
)


class StateModule:

    # Attribute labels are not values: removing them means a rephrasing such
    # as "color: black" / "black color" remains compatible, while black/red
    # does not merely match on the word "color".
    _NON_VALUE_WORDS = frozenset({
        "a", "an", "and", "around", "at", "budget", "by", "color", "for",
        "from", "in", "is", "it", "material", "of", "on", "or", "the",
        "to", "under", "with",
    })

    def decide(self, state: SessionState, turn: int, nlu_result: NLUResult, user_message: str = "") -> StrategyDecision:
        if nlu_result.is_override:
            state.override_detected = True
            # Keep stable session context in structured state.  The earlier
            # free-form transcript may contradict the new intent, and query
            # construction should never need to recover constraints from it.
            state.messages.clear()
            state.constraints = [
                c for c in state.constraints
                if self._is_compatible_with_override(c, nlu_result.new_constraints)
            ]
            state.exhausted_attributes -= {
                c.attribute_type for c in nlu_result.new_constraints
            }

        if nlu_result.is_no_preference:
            state.boundary_detected = True
            if state.attributes_asked:
                last_asked = state.attributes_asked[-1]
                if last_asked:
                    state.exhausted_attributes.add(last_asked)

        if nlu_result.is_exhausted and nlu_result.exhausted_attribute:
            state.exhausted_attributes.add(nlu_result.exhausted_attribute)

        for c in nlu_result.new_constraints:
            if not any(ex.raw_text == c.raw_text for ex in state.constraints):
                state.constraints.append(c)

        state.messages.append({"role": "user", "text": user_message, "turn": turn})

        if nlu_result.category_text and not state.category_text:
            state.category_text = nlu_result.category_text

        if nlu_result.detected_intent != "unknown" and turn == 1:
            state.scenario_guess = nlu_result.detected_intent

        should_broad_reask = should_broad_reask_after_boundary(
            is_no_preference=nlu_result.is_no_preference,
            attributes_asked=state.attributes_asked,
        )
        attr = BOUNDARY_BROAD_REASK_ATTRIBUTE if should_broad_reask else self._pick_attribute(state, turn)
        if should_broad_reask:
            msg = BOUNDARY_BROAD_REASK_MESSAGE
        elif attr is None:
            msg = "Here are some options based on what you've told me so far."
        else:
            msg = ATTRIBUTE_MESSAGES.get(attr, "What else matters to you?")

        state.attributes_asked.append(attr)

        return StrategyDecision(
            ask_attribute=attr,
            message_template=msg,
            should_recommend=True,
            strategy_note=f"turn={turn} attr={attr} constraints={len(state.constraints)}",
        )

    @classmethod
    def _is_compatible_with_override(
        cls,
        existing: Constraint,
        replacements: list[Constraint],
    ) -> bool:
        """Return whether an existing structured constraint survives an override."""
        same_type = [
            replacement
            for replacement in replacements
            if replacement.attribute_type == existing.attribute_type
        ]
        if not same_type:
            # The override addresses a different attribute, so this remains
            # useful, compatible intent context.
            return True
        return any(cls._values_are_compatible(existing.value, replacement.value) for replacement in same_type)

    @classmethod
    def _values_are_compatible(cls, existing_value: str, replacement_value: str) -> bool:
        existing = cls._normalize_value(existing_value)
        replacement = cls._normalize_value(replacement_value)
        if not existing or not replacement:
            # Do not discard an otherwise useful parsed constraint because an
            # incoming value was unexpectedly empty.
            return True
        if existing in replacement or replacement in existing:
            return True
        return bool(cls._value_tokens(existing) & cls._value_tokens(replacement))

    @staticmethod
    def _normalize_value(value: str) -> str:
        return " ".join(value.lower().split())

    @classmethod
    def _value_tokens(cls, value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.lower())
            if token not in cls._NON_VALUE_WORDS
        }

    def _pick_attribute(self, state: SessionState, turn: int) -> Optional[str]:
        known_types = {c.attribute_type for c in state.constraints}
        blocked = state.exhausted_attributes | known_types

        idx = turn - 1
        order = ASK_STRATEGY_ORDER

        for offset in range(len(order)):
            candidate = order[(idx + offset) % len(order)]
            if candidate not in blocked:
                return candidate

        for attr in ALLOWED_ATTRIBUTES:
            if attr not in blocked:
                return attr

        return None
