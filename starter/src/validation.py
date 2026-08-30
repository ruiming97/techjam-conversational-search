from __future__ import annotations

from starter.src.interfaces import ALLOWED_ATTRIBUTES


class ResponseValidator:

    def __init__(self, asin_set: set[str]) -> None:
        self._asin_set = asin_set

    def validate(self, response: dict, top_k: int = 10) -> dict:
        if not isinstance(response.get("message"), str) or not response["message"]:
            response["message"] = "Let me help you find what you're looking for."

        attr = response.get("ask_attribute")
        if attr is not None and attr not in ALLOWED_ATTRIBUTES:
            response["ask_attribute"] = "other"

        response["recommendations"] = self._validate_recs(
            response.get("recommendations", []), top_k
        )

        usage = response.get("usage") or {}
        response["usage"] = {
            "prompt_tokens": max(0, int(usage.get("prompt_tokens", 0))),
            "completion_tokens": max(0, int(usage.get("completion_tokens", 0))),
        }
        return response

    def _validate_recs(self, recs: list, top_k: int) -> list[dict]:
        seen: set[str] = set()
        valid: list[dict] = []
        for item in recs:
            if not isinstance(item, dict):
                continue
            asin = str(item.get("parent_asin", "")).strip()
            if not asin or asin in seen or asin not in self._asin_set:
                continue
            seen.add(asin)
            valid.append({"parent_asin": asin})
            if len(valid) >= top_k:
                break
        return valid
