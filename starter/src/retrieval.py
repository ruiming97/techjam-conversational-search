from __future__ import annotations

from starter.src.catalog_index import CatalogIndex, tokenize
from starter.src.interfaces import Constraint, RetrievalResult


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
        for c in constraints:
            words = tokenize(c.value)
            if len(words) >= 2:
                phrases.append(" ".join(words[:6]))

        terms = tokenize(query_text)
        for c in constraints:
            terms.extend(tokenize(c.value))
        terms = list(dict.fromkeys(terms))

        if not terms and not phrases:
            return RetrievalResult(ranked_asins=[], scores=[])

        parts: list[str] = []
        for phrase in phrases:
            parts.append(f'"{phrase}"')
        for t in terms:
            parts.append(f'"{t}"')

        expression = " OR ".join(parts)
        if not expression:
            return RetrievalResult(ranked_asins=[], scores=[])

        results = self._catalog.bm25_search_raw(expression, limit=top_k)
        return RetrievalResult(
            ranked_asins=[asin for asin, _ in results],
            scores=[score for _, score in results],
            method="bm25",
        )
