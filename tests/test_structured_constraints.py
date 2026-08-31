from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.src.catalog_index import CatalogIndex
from starter.src.interfaces import Constraint, NLUResult, SessionState
from starter.src.nlu import NLUModule
from starter.src.retrieval import RetrievalModule
from starter.src.state import StateModule


def _catalog(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


def _product(asin: str, price: float, features: list[str]) -> dict:
    return {
        "parent_asin": asin,
        "title": "Everyday trail belt",
        "categories": ["Accessories", "Belts"],
        "features": features,
        "details": {},
        "store": "Example",
        "description": [],
        "price": price,
    }


class StructuredConstraintTest(unittest.TestCase):
    def test_budget_around_price_promotes_nearest_catalog_item(self) -> None:
        path = _catalog([
            _product("LOW", 24.0, ["durable casual belt"]),
            _product("TARGET", 50.0, ["durable casual belt"]),
            _product("HIGH", 99.0, ["durable casual belt"]),
        ])
        try:
            result = RetrievalModule(CatalogIndex(path)).search(
                "everyday trail belt",
                [Constraint("budget around $50", "budget", "50", 1)],
                top_k=3,
            )
            self.assertEqual(result.ranked_asins[0], "TARGET")
        finally:
            path.unlink(missing_ok=True)

    def test_negative_material_is_demoted_from_recommendations(self) -> None:
        path = _catalog([
            _product("LEATHER", 50.0, ["genuine leather everyday belt"]),
            _product("COTTON", 50.0, ["cotton canvas everyday belt"]),
        ])
        try:
            result = RetrievalModule(CatalogIndex(path)).search(
                "everyday trail belt",
                [],
                top_k=2,
                negative_constraints=[Constraint("leather", "material", "leather", 1)],
            )
            self.assertEqual(result.ranked_asins[0], "COTTON")
        finally:
            path.unlink(missing_ok=True)

    def test_color_value_matches_when_disclosure_includes_attribute_label(self) -> None:
        path = _catalog([
            _product("GREY", 50.0, ["soft grey everyday belt"]),
            _product("RED", 50.0, ["soft red everyday belt"]),
        ])
        try:
            result = RetrievalModule(CatalogIndex(path)).search(
                "everyday belt",
                [Constraint("color: grey", "color", "color: grey", 1)],
                top_k=2,
            )
            self.assertEqual(result.ranked_asins[0], "GREY")
        finally:
            path.unlink(missing_ok=True)

    def test_nlu_preserves_rejected_material_as_negative_constraint(self) -> None:
        state = SessionState(session_id="negative", user_profile={})
        result = NLUModule().parse("I don't want leather or wool.", 2, state)

        self.assertEqual(result.new_constraints, [])
        self.assertEqual(
            {(constraint.attribute_type, constraint.value) for constraint in result.negative_constraints},
            {("material", "leather"), ("material", "wool")},
        )

    def test_nlu_does_not_turn_strictly_disclosed_rejection_into_positive_constraint(self) -> None:
        state = SessionState(session_id="strict-negative", user_profile={})
        result = NLUModule().parse("For that, what matters is: avoid leather.", 2, state)

        self.assertEqual(result.new_constraints, [])
        self.assertEqual([constraint.value for constraint in result.negative_constraints], ["leather"])

    def test_negative_constraint_removes_conflicting_prior_positive_value(self) -> None:
        state = SessionState(
            session_id="negative-state",
            user_profile={},
            constraints=[Constraint("leather", "material", "leather", 1)],
        )
        StateModule().decide(
            state,
            2,
            NLUResult(
                new_constraints=[],
                negative_constraints=[Constraint("leather", "material", "leather", 2)],
            ),
            "I don't want leather.",
        )

        self.assertEqual(state.constraints, [])
        self.assertEqual([constraint.value for constraint in state.negative_constraints], ["leather"])


if __name__ == "__main__":
    unittest.main()
