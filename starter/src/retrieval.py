from __future__ import annotations

import math
import re
from dataclasses import dataclass

from starter.src.catalog_index import CatalogIndex, tokenize
from starter.src.interfaces import Constraint, RetrievalResult


_CLAUSE_CANDIDATE_LIMIT = 250
_FALLBACK_CANDIDATE_MULTIPLIER = 5
_PRICE_CANDIDATE_LIMIT = 250
_OVERRIDE_CATEGORY_WEIGHT = 2.0
_BUDGET_AMOUNT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ATTRIBUTE_LABEL_TERMS = frozenset({
    "brand", "budget", "category", "color", "feature", "material",
    "size", "style", "use", "case",
})


@dataclass(frozen=True)
class _BudgetPreference:
    amount: float
    mode: str


def _budget_preference(constraint: Constraint) -> _BudgetPreference | None:
    """Parse a budget phrase while preserving its directional meaning."""
    if constraint.attribute_type != "budget":
        return None
    source = f"{constraint.raw_text} {constraint.value}".lower()
    match = _BUDGET_AMOUNT_RE.search(source)
    if not match:
        return None
    try:
        amount = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    if amount <= 0:
        return None
    if re.search(r"\b(?:under|below|less than|no more than|at most|maximum)\b", source):
        mode = "maximum"
    elif re.search(r"\b(?:over|above|more than|at least|minimum)\b", source):
        mode = "minimum"
    else:
        # A bare dollar value is normally conversational shorthand for an
        # approximate budget, not an exact hard filter.
        mode = "around"
    return _BudgetPreference(amount=amount, mode=mode)


def _budget_score(price: float, preference: _BudgetPreference) -> float:
    if preference.mode == "maximum":
        if price > preference.amount:
            return 0.0
        return 0.5 + 0.5 * (price / preference.amount)
    if preference.mode == "minimum":
        if price < preference.amount:
            return 0.0
        return 0.5 + 0.5 * (preference.amount / price)

    # A 20% tolerance (with a $5 floor) rewards prices near the stated amount
    # without forcing an unrealistically exact retail-price match.
    tolerance = max(5.0, preference.amount * 0.20)
    return max(0.0, 1.0 - abs(price - preference.amount) / tolerance)


def _constraint_terms(constraint: Constraint) -> list[str]:
    """Tokenize a constraint without requiring its attribute label in FTS.

    Customer-facing disclosures often include labels such as ``color: grey``.
    Catalog text typically contains the value (``grey``), not the label, so
    retaining it in an AND-style clause turns a correct match into a miss.
    """
    terms = tokenize(constraint.value)
    if constraint.attribute_type in {
        "brand", "color", "material", "size", "style", "use_case",
    }:
        value_terms = [term for term in terms if term not in _ATTRIBUTE_LABEL_TERMS]
        return value_terms or terms
    return terms


class RetrievalModule:

    def __init__(self, catalog: CatalogIndex) -> None:
        self._catalog = catalog

    def search(
        self,
        query_text: str,
        constraints: list[Constraint],
        top_k: int = 50,
        negative_constraints: list[Constraint] | None = None,
        override_category_text: str = "",
    ) -> RetrievalResult:
        # Keep each user statement as a clause.  Combining all words into one
        # OR expression makes a product that matches "comfortable" look as
        # relevant as one matching a whole disclosed product feature.
        budget_preferences = [
            preference
            for constraint in constraints
            if (preference := _budget_preference(constraint)) is not None
        ]
        text_constraints = [c for c in constraints if c.attribute_type != "budget"]
        clauses = [(_constraint_terms(c), c.value, 1.0) for c in text_constraints]
        # Once the shopper overrides earlier intent, a known category is a
        # useful guardrail against generic products that happen to share the
        # new material or color.  Promote it to a weighted clause only then:
        # before an override, broad category recall remains more important.
        category_terms = tokenize(override_category_text)
        if category_terms:
            clauses.append((category_terms, override_category_text, _OVERRIDE_CATEGORY_WEIGHT))
        clauses = [(terms, raw, weight) for terms, raw, weight in clauses if terms]
        budget_terms = {
            term
            for constraint in constraints
            if constraint.attribute_type == "budget"
            for term in tokenize(constraint.value)
        }
        base_terms = [term for term in tokenize(query_text) if term not in budget_terms]
        all_unique_terms = list(dict.fromkeys(base_terms + [
            term for clause, _raw, _weight in clauses for term in clause
        ]))
        if not all_unique_terms and not budget_preferences:
            return RetrievalResult(ranked_asins=[], scores=[])

        # Broad BM25 remains a recall-oriented fallback and a stable tie
        # breaker.  We deliberately retrieve deeper than top_k so a candidate
        # found by one precise clause can be promoted when it also satisfies
        # another clause.
        fallback = self._catalog.bm25_search(
            all_unique_terms,
            limit=max(top_k * _FALLBACK_CANDIDATE_MULTIPLIER, top_k),
        ) if all_unique_terms else []

        # Weight a complete clause by its inverse catalog frequency. A rare,
        # specific product detail is stronger evidence than a common term
        # such as "black". The +1 keeps common clauses useful.
        clause_weights = [
            1.0 + math.log(
                (len(self._catalog.asin_set) + 1)
                / (self._catalog.clause_document_frequency(clause) + 1)
            ) * weight
            for clause, _raw, weight in clauses
        ]
        total_clause_weight = sum(clause_weights)

        # asin -> [completed clause weight, exact phrase count, BM25 evidence, budget evidence]
        evidence: dict[str, list[float]] = {}
        for rank, (asin, raw_score) in enumerate(fallback):
            # SQLite's bm25 is negative and lower is better.  A small
            # rank-based component avoids depending on score scale across
            # different FTS queries.
            evidence[asin] = [0.0, 0.0, 1.0 / (rank + 1), 0.0]

        for (clause, raw_clause, _weight), clause_weight in zip(clauses, clause_weights):
            complete, phrase_asins = self._catalog.clause_search(
                clause,
                limit=_CLAUSE_CANDIDATE_LIMIT,
                phrase_text=raw_clause,
            )
            for rank, (asin, _raw_score) in enumerate(complete):
                values = evidence.setdefault(asin, [0.0, 0.0, 0.0, 0.0])
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
                values = evidence.setdefault(asin, [0.0, 0.0, 0.0, 0.0])
                values[0] = total_clause_weight
                values[2] += 2.0 / (rank + 1)

        # Inject the nearest-price candidates even when their numeric value
        # never appears in full-text fields.  This gives a disclosed budget a
        # meaningful recall path rather than treating it as an FTS keyword.
        for preference in budget_preferences:
            for asin, price in self._catalog.nearest_price_matches(
                preference.amount,
                limit=_PRICE_CANDIDATE_LIMIT,
            ):
                values = evidence.setdefault(asin, [0.0, 0.0, 0.0, 0.0])
                values[3] = max(values[3], _budget_score(price, preference))

        excluded_asins: set[str] = set()
        for constraint in negative_constraints or []:
            terms = _constraint_terms(constraint)
            if not terms:
                continue
            matching, _phrase_asins = self._catalog.clause_search(
                terms,
                limit=_CLAUSE_CANDIDATE_LIMIT,
                phrase_text=constraint.value,
            )
            excluded_asins.update(asin for asin, _score in matching)

        def score(item: tuple[str, list[float]]) -> tuple[float, float, float, float, float, str]:
            asin, (complete_count, phrase_count, bm25_evidence, budget_evidence) = item
            # Complete-clause coverage is lexicographically dominant.  This
            # implements AND-style preference across multiple statements;
            # phrase evidence then separates feature text from coincidental
            # bag-of-words matches.  A disclosed constraint is frequently
            # boilerplate shared by an entire product line (a common
            # material plus one or two generic care/closure phrases), so
            # once every disclosed clause is fully satisfied, coverage and
            # phrase evidence alone often leave many sibling listings
            # genuinely tied, and BM25 field-length overlap among such
            # near-duplicates is itself close to arbitrary.  Review count
            # breaks that tie toward the more established listing.  Gate
            # this strictly on full coverage: with no disclosed constraint
            # yet, or only a partial match, BM25 relevance to the
            # customer's own words remains the best available signal, and
            # popularity must not override it.
            coverage = complete_count / total_clause_weight if total_clause_weight else 0.0
            popularity = self._catalog.popularity(asin) if coverage >= 0.999 else 0.0
            return (coverage, budget_evidence, phrase_count, popularity, bm25_evidence, asin)

        ordered = sorted(evidence.items(), key=score, reverse=True)
        # Negative preferences are a strong exclusion rather than a weak text
        # penalty.  Keep matching fallbacks at the end only when the catalog
        # cannot otherwise supply the requested number of recommendations.
        ranked = [item for item in ordered if item[0] not in excluded_asins]
        ranked.extend(item for item in ordered if item[0] in excluded_asins)
        ranked = ranked[:top_k]
        return RetrievalResult(
            ranked_asins=[asin for asin, _ in ranked],
            scores=[score(item)[0] for item in ranked],
            method="clause_aware_bm25",
        )
