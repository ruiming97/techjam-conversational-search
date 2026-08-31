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
