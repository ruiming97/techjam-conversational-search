from __future__ import annotations

from typing import Optional

from starter.src.config import ASK_STRATEGY_ORDER, ATTRIBUTE_MESSAGES
from starter.src.interfaces import (
    ALLOWED_ATTRIBUTES,
    NLUResult,
    SessionState,
    StrategyDecision,
)


class StateModule:

    def decide(self, state: SessionState, turn: int, nlu_result: NLUResult, user_message: str = "") -> StrategyDecision:
        if nlu_result.is_override:
            state.override_detected = True
            overridden_types = {c.attribute_type for c in nlu_result.new_constraints}
            state.constraints = [
                c for c in state.constraints if c.attribute_type not in overridden_types
            ]
            state.exhausted_attributes -= overridden_types

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

        attr = self._pick_attribute(state, turn)
        if attr is None:
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
