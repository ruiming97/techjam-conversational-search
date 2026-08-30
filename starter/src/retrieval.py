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
        constraint_terms: list[str] = []

        # 1. Extract exact phrases and individual terms from constraints
        for c in constraints:
            words = tokenize(c.value)
            if len(words) >= 1:
                phrases.append(" ".join(words[:6]))
                constraint_terms.extend(words)

        # 2. Extract base query terms
        base_terms = tokenize(query_text)
        all_unique_terms = list(dict.fromkeys(base_terms + constraint_terms))

        parts: list[str] = []
        
        # 3. Add exact constraint phrases multiple times
        # This artificially skews the FTS5 math so exact phrase matches skyrocket to the top
        for phrase in phrases:
            parts.append(f'"{phrase}"')
            parts.append(f'"{phrase}"') # 2x multiplier
            parts.append(f'"{phrase}"') # 3x multiplier
            
        # 4. The Safety Net: Add all individual terms once
        for t in all_unique_terms:
            parts.append(f'"{t}"')

        expression = " OR ".join(parts)
        if not expression and not parts:
            return RetrievalResult(ranked_asins=[], scores=[])

        # 5. Execute standard BM25 with the heavily weighted expression
        results = self._catalog.bm25_search_raw(expression, limit=top_k)
        
        return RetrievalResult(
            ranked_asins=[asin for asin, _ in results],
            scores=[score for _, score in results],
            method="bm25_boosted",
        )
