from __future__ import annotations

from starter.src.catalog_index import CatalogIndex, tokenize


def safe_bm25_response(
    catalog: CatalogIndex,
    user_message: str,
    top_k: int = 10,
) -> dict:
    terms = tokenize(user_message)
    recs: list[dict] = []
    if terms:
        results = catalog.bm25_search(terms, limit=top_k)
        recs = [{"parent_asin": asin} for asin, _ in results]
    return {
        "message": "Could you tell me more about what you're looking for?",
        "ask_attribute": "other",
        "recommendations": recs,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def empty_response() -> dict:
    return {
        "message": "I'd love to help. What features matter most?",
        "ask_attribute": "other",
        "recommendations": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }
