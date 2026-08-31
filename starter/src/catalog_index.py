from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from starter.src.config import BM25_WEIGHTS

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def tokenize(text: str) -> list[str]:
    return [
        tok.lower()
        for tok in TOKEN_RE.findall(text)
        if len(tok) > 1 and tok.lower() not in STOPWORDS
    ]


class CatalogIndex:

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.asin_set: set[str] = set()
        self.connection = sqlite3.connect(":memory:")
        self._clause_cache: dict[
            tuple[tuple[str, ...], int, str], tuple[tuple[tuple[str, float], ...], frozenset[str]]
        ] = {}
        self._clause_document_frequency_cache: dict[tuple[str, ...], int] = {}
        self._build()

    def _build(self) -> None:
        cur = self.connection.cursor()
        cur.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as fh:
            for line in fh:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.asin_set.add(asin)
                batch.append((
                    asin,
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 1000:
                    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def is_valid(self, asin: str) -> bool:
        return asin in self.asin_set

    def bm25_search(self, query_terms: list[str], limit: int = 50) -> list[tuple[str, float]]:
        unique = list(dict.fromkeys(query_terms))[:60]
        expression = " OR ".join(f'"{t}"' for t in unique)
        return self.bm25_search_raw(expression, limit) if expression else []

    def bm25_search_raw(self, expression: str, limit: int = 50) -> list[tuple[str, float]]:
        if not expression:
            return []
        weights = ", ".join(str(w) for w in BM25_WEIGHTS)
        rows = self.connection.execute(
            f"SELECT parent_asin, bm25(products, {weights}) "
            f"FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {weights}) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]

    def clause_search(
        self,
        terms: list[str],
        limit: int = 250,
        phrase_text: str | None = None,
    ) -> tuple[list[tuple[str, float]], set[str]]:
        """Return products that satisfy a complete constraint clause.

        The normal BM25 query is intentionally broad: it uses OR so a useful
        first-turn recommendation is possible even with little information.
        A disclosed constraint is different.  It normally originates from a
        product feature/details card, and products containing *every* salient
        word are much stronger candidates than products that happen to contain
        one generic word such as ``comfort``.  FTS5 lets us obtain this small
        high-precision candidate set without loading the full catalog into
        Python on every turn.

        The first return value contains AND-term matches ordered by BM25.  The
        second identifies the subset which also contains the complete phrase
        in order.  Both use normalized tokens, so punctuation in the source
        catalog does not prevent an otherwise exact match.
        """
        unique = list(dict.fromkeys(terms))[:24]
        if not unique:
            return [], set()
        cache_key = (tuple(unique), limit, (phrase_text or "").casefold())
        cached = self._clause_cache.get(cache_key)
        if cached is not None:
            rows, phrase_asins = cached
            return list(rows), set(phrase_asins)

        # Quotes are safe because tokenize() only returns alphanumeric terms.
        conjunction = " AND ".join(f'"{term}"' for term in unique)
        complete = self.bm25_search_raw(conjunction, limit=limit)

        # Phrase matching is particularly useful for feature sentences.  Do
        # not issue it for a one-word clause: the conjunction already means
        # exactly the same thing in that case.
        phrase_asins: set[str] = set()
        if len(unique) > 1:
            # Keep stopwords for phrase matching.  ``tokenize`` drops them
            # for a useful AND query, but removing words such as "with" or
            # "for" changes FTS token positions and would make an otherwise
            # verbatim feature sentence fail its phrase match.
            phrase_tokens = TOKEN_RE.findall(phrase_text or " ".join(unique))
            phrase = '"' + " ".join(phrase_tokens) + '"'
            phrase_asins = {
                asin for asin, _ in self.bm25_search_raw(phrase, limit=limit)
            }
        self._clause_cache[cache_key] = (tuple(complete), frozenset(phrase_asins))
        return complete, phrase_asins

    def clause_document_frequency(self, terms: list[str]) -> int:
        """Return the catalog frequency of a complete constraint clause."""
        unique = tuple(dict.fromkeys(terms))[:24]
        if not unique:
            return 0
        cached = self._clause_document_frequency_cache.get(unique)
        if cached is not None:
            return cached
        conjunction = " AND ".join(f'"{term}"' for term in unique)
        count = int(self.connection.execute(
            "SELECT count(*) FROM products WHERE products MATCH ?", (conjunction,)
        ).fetchone()[0])
        self._clause_document_frequency_cache[unique] = count
        return count
