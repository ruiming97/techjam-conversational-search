from __future__ import annotations

from pathlib import Path

from starter.src.catalog_index import CatalogIndex
from starter.src.fallback import empty_response, safe_bm25_response
from starter.src.interfaces import SessionState
from starter.src.nlu import NLUModule
from starter.src.retrieval import RetrievalModule
from starter.src.state import StateModule
from starter.src.validation import ResponseValidator


class Agent:

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._catalog = CatalogIndex(catalog_path)
        self._nlu = NLUModule()
        self._state_mod = StateModule()
        self._retrieval = RetrievalModule(self._catalog)
        self._validator = ResponseValidator(self._catalog.asin_set)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            user_profile=user_profile,
        )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        try:
            return self._respond_inner(state, user_message, turn, top_k)
        except Exception:
            try:
                return safe_bm25_response(self._catalog, user_message, top_k)
            except Exception:
                return empty_response()

    def _respond_inner(
        self,
        state: SessionState,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        nlu_result = self._nlu.parse(user_message, turn, state)
        decision = self._state_mod.decide(state, turn, nlu_result, user_message)
        query_text = self._build_query(state, nlu_result.category_text)
        retrieval = self._retrieval.search(query_text, state.constraints, top_k=50)

        response = {
            "message": decision.message_template,
            "ask_attribute": decision.ask_attribute,
            "recommendations": [{"parent_asin": a} for a in retrieval.ranked_asins],
            "usage": {
                "prompt_tokens": state.total_prompt_tokens,
                "completion_tokens": state.total_completion_tokens,
            },
        }
        return self._validator.validate(response, top_k)

    _NOISE = frozenset({
        "those options are not quite right yet",
        "ask me about one specific attribute",
        "i don't have an additional preference",
        "i don't have a preference",
        "please use your judgment",
    })

    def _build_query(self, state: SessionState, category_text: str) -> str:
        parts: list[str] = []
        if state.category_text:
            parts.append(state.category_text)
        elif category_text:
            parts.append(category_text)
        override_turn = None
        if state.override_detected:
            for msg in state.messages:
                if msg["role"] == "user" and "ignore my earlier" in msg["text"].lower():
                    override_turn = msg["turn"]
                    break
        for msg in state.messages:
            if msg["role"] != "user" or not msg.get("text"):
                continue
            if override_turn and msg["turn"] < override_turn and msg["turn"] != 1:
                continue
            text_lower = msg["text"].lower()
            if any(noise in text_lower for noise in self._NOISE):
                continue
            parts.append(msg["text"])
        for c in state.constraints:
            parts.append(c.value)
        tags = state.user_profile.get("preference_tags", [])
        if tags:
            parts.append(" ".join(str(t) for t in tags))
        return " ".join(parts)
