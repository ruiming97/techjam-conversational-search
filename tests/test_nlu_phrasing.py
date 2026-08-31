from __future__ import annotations

import unittest

from starter.src.config import BOUNDARY_BROAD_REASK_MESSAGE
from starter.src.interfaces import Constraint, SessionState, StrategyDecision
from starter.src.nlu import NLUModule, _join_natural, _recent_unique_constraint_values, _truncate_value


def make_state(profile: dict | None = None, constraints: list[Constraint] | None = None) -> SessionState:
    state = SessionState(session_id="s1", user_profile=profile if profile is not None else {})
    if constraints:
        state.constraints = constraints
    return state


def make_decision(ask_attribute: str | None = "material", message: str = "Do you have a material preference?") -> StrategyDecision:
    return StrategyDecision(ask_attribute=ask_attribute, message_template=message)


class PhraseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nlu = NLUModule()

    def test_constraints_present_take_priority_over_tags(self) -> None:
        state = make_state(
            profile={"preference_tags": ["fit", "comfort"]},
            constraints=[Constraint(raw_text="leather", attribute_type="material", value="leather", turn_received=1)],
        )
        decision = make_decision()
        message = self.nlu.phrase(decision, state, num_recommendations=5)
        self.assertIn("leather", message)
        self.assertIn(decision.message_template, message)
        self.assertNotIn("fit", message)

    def test_no_recommendations_suppresses_constraint_explanation(self) -> None:
        state = make_state(
            profile={},
            constraints=[Constraint(raw_text="leather", attribute_type="material", value="leather", turn_received=1)],
        )
        decision = make_decision()
        message = self.nlu.phrase(decision, state, num_recommendations=0)
        self.assertEqual(message, decision.message_template)

    def test_ask_attribute_none_suppresses_constraint_explanation(self) -> None:
        state = make_state(
            profile={},
            constraints=[Constraint(raw_text="leather", attribute_type="material", value="leather", turn_received=1)],
        )
        decision = make_decision(ask_attribute=None, message="Here are some options based on what you've told me so far.")
        message = self.nlu.phrase(decision, state, num_recommendations=5)
        self.assertEqual(message, decision.message_template)

    def test_no_constraints_uses_preference_tags(self) -> None:
        state = make_state(profile={"preference_tags": ["fit", "comfort", "durability"]})
        decision = make_decision()
        message = self.nlu.phrase(decision, state, num_recommendations=0)
        self.assertIn("fit", message)
        self.assertIn("comfort", message)
        self.assertIn("durability", message)
        self.assertIn(decision.message_template, message)

    def test_empty_tags_falls_back_to_summary(self) -> None:
        state = make_state(profile={"preference_tags": [], "summary": "Prior purchases emphasize fit and comfort."})
        decision = make_decision()
        message = self.nlu.phrase(decision, state, num_recommendations=0)
        self.assertIn("Prior purchases emphasize fit and comfort.", message)

    def test_no_tags_no_summary_returns_bare_template(self) -> None:
        state = make_state(profile={"preference_tags": [], "summary": ""})
        decision = make_decision()
        message = self.nlu.phrase(decision, state, num_recommendations=0)
        self.assertEqual(message, decision.message_template)

    def test_missing_profile_keys_returns_bare_template(self) -> None:
        state = make_state(profile={})
        decision = make_decision()
        message = self.nlu.phrase(decision, state, num_recommendations=0)
        self.assertEqual(message, decision.message_template)

    def test_boundary_reask_suppresses_tag_opener(self) -> None:
        state = make_state(profile={"preference_tags": ["fit", "comfort"]})
        decision = make_decision(ask_attribute="other", message=BOUNDARY_BROAD_REASK_MESSAGE)
        message = self.nlu.phrase(decision, state, num_recommendations=0)
        self.assertEqual(message, BOUNDARY_BROAD_REASK_MESSAGE)

    def test_malformed_profile_never_raises(self) -> None:
        decision = make_decision()
        for bad_profile in (
            None,
            "not a dict",
            {"preference_tags": "not a list"},
            {"preference_tags": [None, 123, "  ", "fit"]},
            {"summary": 12345},
        ):
            state = make_state(profile=bad_profile)  # type: ignore[arg-type]
            message = self.nlu.phrase(decision, state, num_recommendations=0)
            self.assertIsInstance(message, str)
            self.assertTrue(message)

    def test_state_constraints_not_mutated(self) -> None:
        constraints = [
            Constraint(raw_text="a", attribute_type="feature", value="a value", turn_received=1),
            Constraint(raw_text="b", attribute_type="material", value="b value", turn_received=2),
            Constraint(raw_text="c", attribute_type="color", value="c value", turn_received=3),
        ]
        state = make_state(profile={}, constraints=list(constraints))
        decision = make_decision()
        self.nlu.phrase(decision, state, num_recommendations=5)
        self.assertEqual([c.value for c in state.constraints], [c.value for c in constraints])


class HelperFunctionTest(unittest.TestCase):
    def test_join_natural(self) -> None:
        self.assertEqual(_join_natural([]), "")
        self.assertEqual(_join_natural(["a"]), "a")
        self.assertEqual(_join_natural(["a", "b"]), "a and b")
        self.assertEqual(_join_natural(["a", "b", "c"]), "a, b, and c")

    def test_truncate_value_short_string_unchanged(self) -> None:
        self.assertEqual(_truncate_value("leather"), "leather")

    def test_truncate_value_cuts_at_word_boundary(self) -> None:
        long_value = "a " * 40 + "final"
        result = _truncate_value(long_value, limit=20)
        self.assertTrue(result.endswith("…"))
        self.assertNotIn(",…", result)
        self.assertFalse(result[:-1].endswith(" "))

    def test_recent_unique_values_dedupes_on_truncated_display(self) -> None:
        shared_prefix = "75% Polyester 20% Rayon 5% Spandex extremely long fabric description"
        constraints = [
            Constraint(raw_text="x", attribute_type="material", value=shared_prefix + " variant one", turn_received=1),
            Constraint(raw_text="y", attribute_type="material", value=shared_prefix + " variant two", turn_received=2),
        ]
        values = _recent_unique_constraint_values(constraints, limit=3)
        self.assertEqual(len(values), 1)

    def test_recent_unique_values_preserves_chronological_order(self) -> None:
        constraints = [
            Constraint(raw_text="a", attribute_type="feature", value="alpha", turn_received=1),
            Constraint(raw_text="b", attribute_type="material", value="beta", turn_received=2),
            Constraint(raw_text="c", attribute_type="color", value="gamma", turn_received=3),
        ]
        values = _recent_unique_constraint_values(constraints, limit=3)
        self.assertEqual(values, ["alpha", "beta", "gamma"])

    def test_recent_unique_values_does_not_mutate_input(self) -> None:
        constraints = [
            Constraint(raw_text="a", attribute_type="feature", value="alpha", turn_received=1),
            Constraint(raw_text="b", attribute_type="material", value="beta", turn_received=2),
        ]
        original_order = list(constraints)
        _recent_unique_constraint_values(constraints, limit=1)
        self.assertEqual(constraints, original_order)


if __name__ == "__main__":
    unittest.main()
