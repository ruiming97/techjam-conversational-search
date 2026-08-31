from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.src.catalog_index import CatalogIndex
from starter.src.interfaces import Constraint, SessionState
from starter.src.retrieval import RetrievalModule


def _write_catalog(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


def _row(asin: str, features: list[str]) -> dict:
    return {
        "parent_asin": asin,
        "title": "Trail backpack",
        "categories": ["Outdoors"],
        "features": features,
        "details": {},
        "store": "Example",
        "description": [],
    }


class RetrievalBehaviorTest(unittest.TestCase):
    def test_products_matching_every_disclosed_clause_outrank_partial_matches(self) -> None:
        """Guard rank quality: an AND-style match beats generic BM25 overlap."""
        path = _write_catalog([
            _row("ALL_CLAUSES", ["water resistant shell", "lightweight construction"]),
            _row("WATER_ONLY", ["water resistant shell", "heavy construction"]),
            _row("LIGHT_ONLY", ["lightweight construction", "does not repel rain"]),
        ])
        try:
            retrieval = RetrievalModule(CatalogIndex(path))
            result = retrieval.search(
                "trail backpack",
                [
                    Constraint("water resistant", "feature", "water resistant", 1),
                    Constraint("lightweight", "feature", "lightweight", 2),
                ],
                top_k=3,
            )
            self.assertEqual(result.method, "clause_aware_bm25")
            self.assertEqual(result.ranked_asins[0], "ALL_CLAUSES")
            self.assertGreater(result.scores[0], result.scores[1])
        finally:
            path.unlink(missing_ok=True)

    def test_query_uses_active_constraints_not_stale_transcript_or_profile_tags(self) -> None:
        """Overrides must not leak old preferences back into retrieval text."""
        state = SessionState(
            session_id="override",
            user_profile={"preference_tags": ["comfort", "obsolete tag"]},
            category_text="running shoes",
            constraints=[
                Constraint("red color", "color", "red color", 3),
                Constraint("under 50 dollars", "budget", "under 50 dollars", 4),
            ],
            messages=[
                {"role": "user", "text": "I need blue leather boots", "turn": 1},
            ],
        )

        query = Agent._build_query(Agent.__new__(Agent), state, "ignored category")

        self.assertIn("running shoes", query)
        self.assertIn("red color", query)
        self.assertIn("under 50 dollars", query)
        self.assertNotIn("blue leather boots", query)
        self.assertNotIn("comfort", query)
        self.assertNotIn("obsolete tag", query)

    def test_profile_tags_are_only_a_first_turn_fallback(self) -> None:
        state = SessionState(
            session_id="first-turn",
            user_profile={"preference_tags": ["durable", "comfortable"]},
        )

        query = Agent._build_query(Agent.__new__(Agent), state, "")

        self.assertEqual(query, "durable comfortable")


if __name__ == "__main__":
    unittest.main()
