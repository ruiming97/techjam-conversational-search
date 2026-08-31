from __future__ import annotations

import math

from starter.src.catalog_index import CatalogIndex, tokenize
from starter.src.interfaces import Constraint, RetrievalResult


_CLAUSE_CANDIDATE_LIMIT = 250
_FALLBACK_CANDIDATE_MULTIPLIER = 5


class RetrievalModule:

    def __init__(self, catalog: CatalogIndex) -> None:
        self._catalog = catalog

    def search(
        self,
        query_text: str,
        constraints: list[Constraint],
        top_k: int = 50,
    ) -> RetrievalResult:
        # Keep each user statement as a clause.  Combining all words into one
        # OR expression makes a product that matches "comfortable" look as
        # relevant as one matching a whole disclosed product feature.
        clauses = [(tokenize(c.value), c.value) for c in constraints]
        clauses = [(terms, raw) for terms, raw in clauses if terms]
        base_terms = tokenize(query_text)
        all_unique_terms = list(dict.fromkeys(base_terms + [
            term for clause, _raw in clauses for term in clause
        ]))
        if not all_unique_terms:
            return RetrievalResult(ranked_asins=[], scores=[])

        # Broad BM25 remains a recall-oriented fallback and a stable tie
        # breaker.  We deliberately retrieve deeper than top_k so a candidate
        # found by one precise clause can be promoted when it also satisfies
        # another clause.
        fallback = self._catalog.bm25_search(
            all_unique_terms,
            limit=max(top_k * _FALLBACK_CANDIDATE_MULTIPLIER, top_k),
        )

        # Weight a complete clause by its inverse catalog frequency. A rare,
        # specific product detail is stronger evidence than a common term
        # such as "black". The +1 keeps common clauses useful.
        clause_weights = [
            1.0 + math.log(
                (len(self._catalog.asin_set) + 1)
                / (self._catalog.clause_document_frequency(clause) + 1)
            )
            for clause, _raw in clauses
        ]
        total_clause_weight = sum(clause_weights)

        # asin -> [completed clause weight, exact phrase count, BM25 evidence]
        evidence: dict[str, list[float]] = {}
        for rank, (asin, raw_score) in enumerate(fallback):
            # SQLite's bm25 is negative and lower is better.  A small
            # rank-based component avoids depending on score scale across
            # different FTS queries.
            evidence[asin] = [0.0, 0.0, 1.0 / (rank + 1)]

        for (clause, raw_clause), clause_weight in zip(clauses, clause_weights):
            complete, phrase_asins = self._catalog.clause_search(
                clause,
                limit=_CLAUSE_CANDIDATE_LIMIT,
                phrase_text=raw_clause,
            )
            for rank, (asin, _raw_score) in enumerate(complete):
                values = evidence.setdefault(asin, [0.0, 0.0, 0.0])
                values[0] += clause_weight
                values[2] += 1.0 / (rank + 1)
                if asin in phrase_asins:
                    values[1] += 1.0

        clause_count = len(clauses)

        # A product satisfying the category/query vocabulary *and* every
        # disclosed term is an especially high-precision candidate.  This
        # avoids a common failure mode of independently scored generic clauses
        # (for example, "leather" and "imported") drowning out the correct
        # item even when its category and all constraints agree.
        if clause_count:
            combined, _phrase_asins = self._catalog.clause_search(
                all_unique_terms,
                limit=_CLAUSE_CANDIDATE_LIMIT,
            )
            for rank, (asin, _raw_score) in enumerate(combined):
                values = evidence.setdefault(asin, [0.0, 0.0, 0.0])
                values[0] = total_clause_weight
                values[2] += 2.0 / (rank + 1)

        def score(item: tuple[str, list[float]]) -> tuple[float, float, float, str]:
            asin, (complete_count, phrase_count, bm25_evidence) = item
            # Complete-clause coverage is lexicographically dominant.  This
            # implements AND-style preference across multiple statements;
            # phrase evidence then separates feature text from coincidental
            # bag-of-words matches, and BM25 only resolves remaining ties.
            coverage = complete_count / total_clause_weight if total_clause_weight else 0.0
            return (coverage, phrase_count, bm25_evidence, asin)

        ranked = sorted(evidence.items(), key=score, reverse=True)[:top_k]
        return RetrievalResult(
            ranked_asins=[asin for asin, _ in ranked],
            scores=[score(item)[0] for item in ranked],
            method="clause_aware_bm25",
        )
