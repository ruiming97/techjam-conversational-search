# Ten-Turn Shopper

A conversational shopping agent that finds a customer's target product from 50,000 items in an average of 2.4 turns — without an LLM.

## The Problem

Traditional e-commerce search fails at conversations. A customer says *"I'm looking for men's basketball shorts"* and gets 2,000 results. The search engine has no way to ask *"what material?"* or *"does breathability matter?"* — it just dumps results and hopes. The customer scrolls, refines, scrolls again. Most give up.

The challenge: build an agent that holds a natural conversation with a simulated shopper, asks the right follow-up questions, and surfaces the exact target product within 10 turns — across 50,000 Clothing, Shoes & Jewelry items from Amazon Reviews 2023.

## Our Insight

We discovered that the most impactful design decision isn't *how* you search — it's *what you ask*.

The agent can ask about 10 attribute types (material, color, size, etc.) or ask `"other"` — an open-ended probe. We analyzed the constraint distribution across all 200 evaluation sessions and found:

- **50.5%** of customer constraints are product features
- **37.8%** are material-related
- **7.5%** are colors
- Budget, size, style, and use-case together account for **under 5%**

Asking "Do you have a budget range?" almost never yields information. But an open-ended question — *"What matters most to you?"* — always does. The simulated customer reveals their top 2 preferences regardless of type.

**This means the agent can extract all of a customer's preferences in just 2 open-ended questions**, then switch to targeted follow-ups for edge cases. The conversation becomes:

> Turn 1: "I'm looking for basketball shorts" → "What matters most to you?"
> Turn 2: "Polyester, 100% Polyester" → "Any other requirements?"
> Turn 3: "Drawstring closure, mesh for breathability" → **Target found at rank 1.**

Instead of 10 turns of guessing, 3 turns of listening.

## Results

| Metric | Weak Baseline | Our Agent | Improvement |
|---|---|---|---|
| HitRate@10 | 12.5% | **97.5%** | 7.8x |
| MRR | 0.068 | **0.711** | 10.4x |
| Mean Turns to Find | 9.81 | **2.35** | 76% faster |
| TechnicalScore | 0.107 | **0.874** | 8.2x |

| Scenario | Hit Rate | Avg Turn |
|---|---|---|
| Buying (40%) — hard constraint upfront | 96% | 1.7 |
| Browsing (40%) — starts vague | 97.5% | 2.4 |
| Intent Override (15%) — changes mind mid-conversation | 100% | 3.7 |
| Boundary (5%) — has no opinion on asked attribute | 100% | 3.3 |

60% of successful matches land at **rank 1**. 69% of all sessions resolve by **turn 2**.

## How It Works

```
Customer: "I'm looking for basketball shorts, still exploring."
                    │
              ┌─────▼─────┐
              │    NLU     │ Extract: category="basketball shorts"
              │            │ Intent: browsing (no hard constraint)
              └─────┬──────┘
                    │
              ┌─────▼──────┐
              │   State    │ Turn 1, no constraints yet
              │            │ → Ask open-ended question
              └─────┬──────┘
                    │
              ┌─────▼──────┐
              │ Retrieval  │ Search: "basketball shorts" + profile tags
              │            │ → Top 10 candidates (broad)
              └─────┬──────┘
                    │
              ┌─────▼──────┐
              │ Validation │ Dedup ASINs, enforce schema
              └─────┬──────┘
                    │
Agent: "What matters most to you?" + [10 recommendations]
```

### The Pipeline

**NLU** (`starter/src/nlu.py`) — Two-tier constraint parser. Strict regex patterns match the evaluator's exact customer templates with 100% accuracy. A second tier of paraphrase-tolerant patterns (override cues, negation detection, generic need expressions) handles potential variation in the private evaluation set. An optional third tier can delegate parsing to Claude when `LLM_ENABLED=true` and an API key is present — it's off by default, so the graded results come entirely from the two heuristic tiers.

**State** (`starter/src/state.py`) — Conversation state machine. Front-loads open-ended discovery in early turns, tracks which attributes have been asked and exhausted, handles intent overrides by preserving compatible constraints while discarding contradicted ones.

**Retrieval** (`starter/src/retrieval.py`) — Clause-aware BM25. Each customer constraint is treated as an independent search clause weighted by inverse document frequency. A product matching a rare, specific feature like "Triple Moon Pentagram" scores higher than one matching a common term like "cotton". Products are ranked by total clause coverage — those satisfying all constraints outrank partial matches. When several candidates satisfy every disclosed clause equally (common for sibling colorways or sizes of the same listing), a log-compressed catalog review count breaks the tie, gated strictly behind full clause coverage so popularity never outranks relevance.

**Validation** (`starter/src/validation.py`) — Response schema enforcement. Deduplicates ASINs against the catalog, caps recommendations at the scored top 10, and wraps every response in a 3-level fallback chain (`starter/src/fallback.py`). The agent never crashes.

### Key Design Decisions

**1. Listen broadly, then narrow.** Open-ended probing in the first 2-3 turns extracts all customer preferences without guessing. Targeted attribute questions (material, color, feature) follow only when the open-ended well runs dry.

**2. Every constraint is a search clause, not a keyword.** "High quality mesh for maximum breathability" is matched as a phrase against product descriptions, not broken into individual words where "high" and "quality" dilute the signal.

**3. Rare constraints matter more.** IDF weighting ensures "Triple Moon Pentagram" (appears in 1 product) contributes more to ranking than "cotton" (appears in 5,000). Once every disclosed clause is satisfied, a review-count tie-break separates sibling listings that read identically to BM25. Together this pushes the exact target to rank 1 in 60% of hits.

**4. Override doesn't mean start over.** When a customer says "actually, forget leather — I want cotton", the agent keeps their color and size preferences while only replacing the material constraint. Compatible context survives the pivot.

**5. No LLM required, no embeddings, no hard dependencies.** The entire agent runs on Python's standard library by default. SQLite FTS5 handles full-text search. This makes the system reproducible from a clean checkout in under 5 seconds, resilient to network outages, and suitable for offline evaluation. An optional, off-by-default Claude-assisted parsing tier exists for future experimentation, but none of the results above depend on it.

## Setup and Reproduction

Python 3.10+ required. No `pip install` needed.

```bash
# Clone
git clone https://github.com/ruiming97/techjam-conversational-search.git
cd techjam-conversational-search

# Download catalog (from GitHub Releases → participant-kit)
gzip -dk data/catalog.jsonl.gz

# Run evaluation (writes results.json)
python3 -m evaluator.local_evaluator

# Run error analysis (optional diagnostic report)
python3 -m scripts.analyze_results
```

## Limitations

- **BM25 ceiling**: 5/200 sessions (2.5%) miss because keyword matching cannot distinguish the target from near-identical products, even after review-count tie-breaking. Dense retrieval (sentence-transformers) would address this by matching on meaning rather than exact terms.
- **No semantic reranking**: A cross-encoder over the BM25 top-50 could promote borderline candidates, improving MRR on sessions where the target sits at rank 8-10.
- **Profile underutilized**: User preference tags (comfort, fit, durability) serve as a first-turn fallback but don't boost retrieval weights.
- **Evaluator-tuned**: The strict NLU tier is built against the evaluator's templates. The paraphrase fallback adds robustness but hasn't been validated against real human shoppers.

## Tools and Resources

- **Language**: Python 3.10+ (standard library only — sqlite3, json, re, math)
- **Search**: SQLite FTS5 for BM25 full-text indexing with clause-aware ranking
- **Development**: Claude Code (AI-assisted development), VS Code
- **Dataset**: Amazon Reviews 2023, Clothing_Shoes_and_Jewelry (50K products, frozen by organizer)

## Team

| Role | Member | Contribution |
|---|---|---|
| Catalog & Retrieval | verakohh | Clause-aware BM25, IDF weighting, phrase boosting, document frequency scoring |
| State & Routing | verakohh | Boundary broad-reask, compatible constraint override, broad discovery turns |
| NLU & Questions | verakohh | Two-tier parser, paraphrase-tolerant fallback, negation awareness |
| Agent Integration | limjeremy496 | Module interfaces, orchestrator, fallback chain, validation |
| Eval & Report | limjeremy496 | Error analysis tooling, README, results tracking |

---

<details>
<summary>Original Challenge README</summary>

## TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

### What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions unreleased until the Devpost submission deadline. After the deadline, the final evaluation package will be released and teams will run the unmodified official evaluator in their own environments using their frozen submitted commit.

See [`docs/final_evaluation_faq.md`](docs/final_evaluation_faq.md) for the final evaluation, network, credentials, hardware, data, and scoring policy.

### Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

### Technical Metrics

```
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

### Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md`.

</details>
