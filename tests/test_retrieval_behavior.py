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


def _row(asin: str, features: list[str], rating_number: int | None = None) -> dict:
    return {
        "parent_asin": asin,
        "title": "Trail backpack",
        "categories": ["Outdoors"],
        "features": features,
        "details": {},
        "store": "Example",
        "description": [],
        "rating_number": rating_number,
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

    def test_popularity_breaks_ties_among_fully_matching_near_duplicates(self) -> None:
        """Among products satisfying every disclosed clause identically (a
        common failure mode when a disclosed constraint is boilerplate
        shared by a whole product line), the more-reviewed listing should
        be preferred over an obscure sibling."""
        path = _write_catalog([
            _row("OBSCURE", ["water resistant shell", "lightweight construction"], rating_number=3),
            _row("POPULAR", ["water resistant shell", "lightweight construction"], rating_number=9000),
        ])
        try:
            result = RetrievalModule(CatalogIndex(path)).search(
                "trail backpack",
                [
                    Constraint("water resistant", "feature", "water resistant", 1),
                    Constraint("lightweight", "feature", "lightweight", 2),
                ],
                top_k=2,
            )
            self.assertEqual(result.ranked_asins[0], "POPULAR")
        finally:
            path.unlink(missing_ok=True)

    def test_popularity_never_overrides_clause_coverage(self) -> None:
        """A hugely popular product that only satisfies one of two disclosed
        clauses must still rank behind a far less popular product that
        satisfies both -- coverage stays lexicographically dominant."""
        path = _write_catalog([
            _row("PARTIAL_BUT_POPULAR", ["water resistant shell", "heavy construction"], rating_number=50000),
            _row("FULL_BUT_OBSCURE", ["water resistant shell", "lightweight construction"], rating_number=1),
        ])
        try:
            result = RetrievalModule(CatalogIndex(path)).search(
                "trail backpack",
                [
                    Constraint("water resistant", "feature", "water resistant", 1),
                    Constraint("lightweight", "feature", "lightweight", 2),
                ],
                top_k=2,
            )
            self.assertEqual(result.ranked_asins[0], "FULL_BUT_OBSCURE")
        finally:
            path.unlink(missing_ok=True)

    def test_popularity_does_not_apply_with_no_disclosed_constraints(self) -> None:
        """Before any constraint is disclosed, a plain exploratory query must
        keep ranking by text relevance -- popularity must not hijack the
        first-turn recommendation for an unrelated but well-reviewed item."""
        path = _write_catalog([
            _row("RELEVANT_OBSCURE", ["trail backpack essential gear"], rating_number=2),
            _row("POPULAR_UNRELATED", ["kitchen blender accessory"], rating_number=100000),
        ])
        try:
            result = RetrievalModule(CatalogIndex(path)).search(
                "trail backpack",
                [],
                top_k=2,
            )
            self.assertEqual(result.ranked_asins[0], "RELEVANT_OBSCURE")
        finally:
            path.unlink(missing_ok=True)

    def test_profile_tags_are_only_a_first_turn_fallback(self) -> None:
        state = SessionState(
            session_id="first-turn",
            user_profile={"preference_tags": ["durable", "comfortable"]},
        )

        query = Agent._build_query(Agent.__new__(Agent), state, "")

        self.assertEqual(query, "durable comfortable")


if __name__ == "__main__":
    unittest.main()
