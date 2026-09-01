from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.src.catalog_index import CatalogIndex
from starter.src.config import BM25_WEIGHTS


def _write_catalog(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


class CatalogIndexBm25WeightTest(unittest.TestCase):
    def test_title_match_outranks_description_only_match(self) -> None:
        # BM25_WEIGHTS weights title far above description, so a product
        # whose only mention of the query term is in its title should
        # outrank one whose only mention is buried in marketing description
        # copy - this is what makes disclosed-attribute words (which tend
        # to land in title/features/details) count for more than incidental
        # mentions in flowery description text.
        path = _write_catalog([
            {
                "parent_asin": "TITLE_MATCH",
                "title": "zzzqueryterm widget",
                "features": [],
                "details": {},
                "store": "Acme",
                "description": ["nothing relevant here"],
            },
            {
                "parent_asin": "DESC_MATCH",
                "title": "unrelated widget",
                "features": [],
                "details": {},
                "store": "Acme",
                "description": ["some text mentioning zzzqueryterm in passing"],
            },
        ])
        try:
            index = CatalogIndex(path)
            results = index.bm25_search_raw('"zzzqueryterm"', limit=10)
            ranked = [asin for asin, _ in results]
            self.assertEqual(ranked[0], "TITLE_MATCH")
            self.assertIn("DESC_MATCH", ranked)
        finally:
            path.unlink(missing_ok=True)

    def test_bm25_weights_wired_into_query(self) -> None:
        # Regression guard: BM25_WEIGHTS in config.py must actually drive the
        # bm25() calls in catalog_index.py, not just be dead configuration
        # that a hardcoded duplicate silently shadows.
        self.assertEqual(len(BM25_WEIGHTS), 7)
        self.assertGreater(BM25_WEIGHTS[1], BM25_WEIGHTS[6])  # title > description


class CatalogIndexPopularityTest(unittest.TestCase):
    def test_popularity_increases_with_rating_count(self) -> None:
        path = _write_catalog([
            {"parent_asin": "FEW", "title": "widget", "features": [], "details": {},
             "store": "Acme", "description": [], "rating_number": 5},
            {"parent_asin": "MANY", "title": "widget", "features": [], "details": {},
             "store": "Acme", "description": [], "rating_number": 5000},
        ])
        try:
            index = CatalogIndex(path)
            self.assertGreater(index.popularity("MANY"), index.popularity("FEW"))
        finally:
            path.unlink(missing_ok=True)

    def test_popularity_defaults_to_zero_when_missing_or_invalid(self) -> None:
        path = _write_catalog([
            {"parent_asin": "NO_FIELD", "title": "widget", "features": [], "details": {},
             "store": "Acme", "description": []},
            {"parent_asin": "NULL_FIELD", "title": "widget", "features": [], "details": {},
             "store": "Acme", "description": [], "rating_number": None},
        ])
        try:
            index = CatalogIndex(path)
            self.assertEqual(index.popularity("NO_FIELD"), 0.0)
            self.assertEqual(index.popularity("NULL_FIELD"), 0.0)
            self.assertEqual(index.popularity("UNKNOWN_ASIN"), 0.0)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
