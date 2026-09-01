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
        retrieval = self._retrieval.search(
            query_text,
            state.constraints,
            top_k=50,
            override_category_text=state.category_text if state.override_detected else "",
        )

        displayed_scores = retrieval.scores[:top_k]
        full_match_count = sum(1 for score in displayed_scores if score >= 0.999)

        response = {
            "message": self._nlu.phrase(
                decision,
                state,
                len(retrieval.ranked_asins),
                full_match_count=full_match_count,
                displayed_count=len(displayed_scores),
            ),
            "ask_attribute": decision.ask_attribute,
            "recommendations": [{"parent_asin": a} for a in retrieval.ranked_asins],
            "usage": {
                "prompt_tokens": state.total_prompt_tokens,
                "completion_tokens": state.total_completion_tokens,
            },
        }
        return self._validator.validate(response, top_k)

    def _build_query(self, state: SessionState, category_text: str) -> str:
        """Build retrieval text from the active, user-disclosed shopping state.

        ``RetrievalModule`` already gives structured constraints a phrase boost.
        Adding the full conversation and profile tags to that query introduced a
        large number of generic OR terms (for example, ``comfort`` or ``fit``),
        which could crowd out an exact constraint.  The state is the source of
        truth for active preferences, so use only its category and constraints
        whenever either is available.  Profile tags remain useful to make a
        first-turn recommendation when the user has supplied no searchable
        information, but they are deliberately a fallback rather than a peer
        signal.
        """
        category = state.category_text or category_text
        structured_parts: list[str] = []
        if category:
            structured_parts.append(category)

        # Preserve constraint phrases, while avoiding repeated copies when a
        # user restates the same requirement on a later turn.
        seen = {category.casefold()} if category else set()
        for constraint in state.constraints:
            value = constraint.value.strip()
            normalized = value.casefold()
            if value and normalized not in seen:
                structured_parts.append(value)
                seen.add(normalized)

        if structured_parts:
            return " ".join(structured_parts)

        tags = state.user_profile.get("preference_tags", [])
        return " ".join(str(tag) for tag in tags if str(tag).strip())
