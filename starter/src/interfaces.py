from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


ALLOWED_ATTRIBUTES = frozenset({
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
})


@dataclass
class Constraint:
    raw_text: str
    attribute_type: str
    value: str
    turn_received: int
    confidence: float = 1.0


@dataclass
class SessionState:
    session_id: str
    user_profile: dict
    constraints: list[Constraint] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    category_text: str = ""
    scenario_guess: str = "unknown"
    attributes_asked: list[str] = field(default_factory=list)
    override_detected: bool = False
    boundary_detected: bool = False
    exhausted_attributes: set[str] = field(default_factory=set)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


@dataclass
class RetrievalResult:
    ranked_asins: list[str]
    scores: list[float] = field(default_factory=list)
    method: str = "bm25"


@dataclass
class NLUResult:
    new_constraints: list[Constraint]
    detected_intent: str = "unknown"
    is_override: bool = False
    is_no_preference: bool = False
    is_exhausted: bool = False
    exhausted_attribute: Optional[str] = None
    raw_query_terms: list[str] = field(default_factory=list)
    category_text: str = ""


@dataclass
class StrategyDecision:
    ask_attribute: Optional[str]
    message_template: str
    should_recommend: bool = True
    strategy_note: str = ""
