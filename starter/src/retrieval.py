from __future__ import annotations

from starter.src.catalog_index import CatalogIndex, tokenize
from starter.src.interfaces import Constraint, RetrievalResult


_PHRASE_BOOST = 3


class RetrievalModule:

    def __init__(self, catalog: CatalogIndex) -> None:
        self._catalog = catalog

    def search(
        self,
        query_text: str,
        constraints: list[Constraint],
        top_k: int = 50,
    ) -> RetrievalResult:
        phrases: list[str] = []
        constraint_terms: list[str] = []
        for c in constraints:
            words = tokenize(c.value)
            if words:
                phrases.append(" ".join(words[:6]))
                constraint_terms.extend(words)

        terms = list(dict.fromkeys(tokenize(query_text) + constraint_terms))

        if not terms and not phrases:
            return RetrievalResult(ranked_asins=[], scores=[])

        parts: list[str] = []
        # SQLite FTS5's bm25() sums a term's contribution per occurrence in an
        # OR expression, so repeating a constraint phrase inflates its weight
        # relative to the single-occurrence terms below (verified empirically:
        # 3 repeats -> ~3x that phrase's score contribution).
        for phrase in phrases:
            parts.extend([f'"{phrase}"'] * _PHRASE_BOOST)
        for t in terms:
            parts.append(f'"{t}"')

        expression = " OR ".join(parts)
        if not expression:
            return RetrievalResult(ranked_asins=[], scores=[])

        results = self._catalog.bm25_search_raw(expression, limit=top_k)
        return RetrievalResult(
            ranked_asins=[asin for asin, _ in results],
            scores=[score for _, score in results],
            method="bm25_boosted",
        )
