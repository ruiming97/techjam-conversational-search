from __future__ import annotations

import unittest

from starter.src.config import ASK_STRATEGY_ORDER
from starter.src.interfaces import Constraint, NLUResult, SessionState
from starter.src.state import StateModule


def make_state() -> SessionState:
    return SessionState(session_id="s1", user_profile={})


def make_nlu(**overrides) -> NLUResult:
    defaults = dict(
        new_constraints=[],
        detected_intent="unknown",
        is_override=False,
        is_no_preference=False,
        is_exhausted=False,
        exhausted_attribute=None,
        raw_query_terms=[],
        category_text="",
    )
    defaults.update(overrides)
    return NLUResult(**defaults)


class StateModuleTest(unittest.TestCase):
    def test_known_attribute_is_never_asked_again(self) -> None:
        state = make_state()
        state.constraints.append(
            Constraint(raw_text="cotton", attribute_type="material", value="cotton", turn_received=1)
        )
        mod = StateModule()
        for turn in range(1, 8):
            decision = mod.decide(state, turn, make_nlu(), "hello")
            self.assertNotEqual(decision.ask_attribute, "material")

    def test_override_only_removes_overridden_attribute_type(self) -> None:
        state = make_state()
        state.constraints.append(
            Constraint(raw_text="cotton", attribute_type="material", value="cotton", turn_received=1)
        )
        state.constraints.append(
            Constraint(raw_text="under $50", attribute_type="budget", value="under $50", turn_received=2)
        )
        mod = StateModule()
        override_nlu = make_nlu(
            is_override=True,
            new_constraints=[
                Constraint(raw_text="leather", attribute_type="material", value="leather", turn_received=3)
            ],
        )
        mod.decide(state, 3, override_nlu, "ignore my earlier preference, What I need is: leather.")

        attribute_types = {c.attribute_type for c in state.constraints}
        self.assertIn("budget", attribute_types)
        self.assertIn("material", attribute_types)
        material_values = [c.value for c in state.constraints if c.attribute_type == "material"]
        self.assertEqual(material_values, ["leather"])

    def test_boundary_decline_is_remembered_and_never_asked_again(self) -> None:
        state = make_state()
        mod = StateModule()
        decision1 = mod.decide(state, 1, make_nlu(), "I'm looking for shoes")
        declined_attr = decision1.ask_attribute
        self.assertIsNotNone(declined_attr)

        mod.decide(
            state, 2,
            make_nlu(is_no_preference=True),
            f"I don't have a preference for {declined_attr}",
        )
        self.assertIn(declined_attr, state.exhausted_attributes)

        for turn in range(3, 8):
            decision = mod.decide(state, turn, make_nlu(), "ok")
            self.assertNotEqual(decision.ask_attribute, declined_attr)

    def test_router_stops_after_ask_cap(self) -> None:
        state = make_state()
        mod = StateModule()
        asked_count = 0
        for turn in range(1, 8):
            decision = mod.decide(state, turn, make_nlu(), "ok")
            if decision.ask_attribute is not None:
                asked_count += 1
        self.assertLessEqual(asked_count, 3)
        self.assertIsNone(state.attributes_asked[-1])

    def test_router_never_asks_on_turn_ten(self) -> None:
        state = make_state()
        mod = StateModule()
        decision = mod.decide(state, 10, make_nlu(), "ok")
        self.assertIsNone(decision.ask_attribute)

    def test_ask_strategy_order_surfaces_budget_and_size(self) -> None:
        self.assertIn("budget", ASK_STRATEGY_ORDER)
        self.assertIn("size", ASK_STRATEGY_ORDER)


if __name__ == "__main__":
    unittest.main()
