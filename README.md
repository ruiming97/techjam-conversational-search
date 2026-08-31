# Ten-Turn Shopper

A multi-turn conversational shopping agent that finds a customer's hidden target product from a 50,000-item Clothing, Shoes & Jewelry catalog in at most 10 turns.

## Results

| Metric | Baseline | Our Agent | Improvement |
|---|---|---|---|
| HitRate@10 | 0.125 | **0.930** | 7.4x |
| MRR | 0.068 | **0.675** | 9.9x |
| MTTC | 9.81 | **3.01** | 69.4% faster |
| TechnicalScore | 0.107 | **0.827** | 7.7x |

Per-scenario: Buying **90%**, Browsing **96%**, Boundary **100%**, Intent Override **90%**.

## Architecture

```
customer message
       |
   [NLU] ── parse constraints (strict templates + paraphrase fallback)
       |         detect intent, override, boundary signals
       |
   [State] ── track constraints, route scenario, pick next attribute
       |         open-ended probing first → targeted attributes later
       |         boundary broad-reask, compatible constraint override
       |
   [Retrieval] ── clause-aware BM25 with IDF weighting + phrase boosting
       |              per-clause document frequency scoring
       |              AND-style coverage ranking across all constraints
       |
   [Validation] ── dedup ASINs, enforce schema, 3-level fallback
       |
   response (message + ask_attribute + recommendations)
```

Five modules in `starter/src/`, wired together by `starter/agent.py`:

| Module | File | Purpose |
|---|---|---|
| NLU | `src/nlu.py` | Two-tier parser: strict regex for known evaluator templates, paraphrase-tolerant fallback for private set robustness. Negation-aware (won't treat "no polyester" as a polyester preference). Optional LLM extraction layer. |
| State | `src/state.py` | Conversation state machine with open-ended discovery in early turns, boundary broad-reask recovery, compatible-constraint override logic that preserves useful context across intent pivots. |
| Retrieval | `src/retrieval.py` | Clause-aware BM25: each constraint is scored as an independent clause with IDF weighting, exact-phrase boosting, and AND-style coverage ranking. Products matching all clauses rank above partial matches. |
| Validation | `src/validation.py` | Response schema enforcement against `agent_api_contract.json`, ASIN dedup against catalog membership set. |
| Fallback | `src/fallback.py` | 3-level fallback chain: module-level catch, safe BM25 response, empty safe response. Agent never crashes. |
| Interfaces | `src/interfaces.py` | Shared contracts: Constraint, SessionState, NLUResult, RetrievalResult, StrategyDecision. |
| Config | `src/config.py` | Strategy parameters, attribute messages, boundary reask policy, broad discovery turn count. |
| Catalog Index | `src/catalog_index.py` | SQLite FTS5 index over 50K products with clause search and document frequency estimation. |

### Key Design Decisions

1. **Open-ended probing first**: The agent front-loads broad questions (`ask_attribute="other"`) in the first 3 turns to maximize information gain. The evaluator's simulated customer reveals any 2 undisclosed constraints per "other" ask — regardless of type — extracting all 4 constraints by turn 3. Targeted attributes (material, feature, color) follow for refinement.

2. **Clause-aware retrieval**: Instead of flattening all constraint terms into one OR query (which lets generic matches like "comfortable" compete with exact feature text), each constraint is treated as an independent clause. Products are ranked by how many clauses they satisfy, with IDF weighting so rare, specific constraints count more than common ones.

3. **Paraphrase-tolerant NLU**: Strict regex patterns match the evaluator's exact customer templates (100% accuracy on the public set). A second tier of broader patterns handles potential paraphrasing in the private 800-session set — covering override cues ("actually", "change my mind"), no-preference signals ("doesn't matter", "up to you"), and generic need expressions.

4. **Compatible constraint override**: When the customer changes their mind, the agent doesn't blindly clear all prior constraints. It checks whether existing constraints are compatible with the new intent (e.g., a color preference may still be valid after a material override) and preserves useful context.

5. **Zero external dependencies**: The entire agent runs on Python stdlib (sqlite3, json, re, math, pathlib). No LLM, no embeddings, no pip install required. An optional LLM extraction layer is available but off by default. Safe for offline finals.

## Setup and Reproduction

Python 3.10+ required. No dependencies beyond stdlib.

```bash
# 1. Clone and enter the repo
git clone https://github.com/ruiming97/techjam-conversational-search.git
cd techjam-conversational-search

# 2. Download and decompress the catalog
# (from GitHub Releases → participant-kit)
gzip -dk data/catalog.jsonl.gz

# 3. Run the evaluator
python3 -m evaluator.local_evaluator

# 4. Run the error analysis (optional)
python3 -m scripts.analyze_results
```

Results are written to `results.json`. The analysis script prints a diagnostic report with per-scenario breakdown, hit/miss distribution, and baseline comparison.

## Limitations and Future Work

- **BM25 retrieval ceiling**: 14 sessions (7%) remain as misses where keyword matching cannot distinguish the target from similar products. Dense retrieval via sentence-transformers embeddings would directly address these by matching on semantic similarity rather than exact term overlap.
- **No learned reranking**: A cross-encoder or LLM reranker over BM25 top-50 could push borderline candidates into the top 10, improving MRR on sessions where the target appears at rank 8-10.
- **Profile underutilization**: User preference tags (comfort, fit, durability, style) are used as a first-turn fallback signal but not as retrieval boost weights.
- **Evaluator-tuned parsing**: The strict NLU tier is tuned to the evaluator's exact templates. The paraphrase fallback adds robustness but hasn't been tested against real human shoppers.

## Tools and Libraries

- **Language**: Python 3.10+ (stdlib only)
- **Search**: SQLite FTS5 for BM25 full-text search with clause-aware ranking
- **Development**: Claude Code (AI-assisted development)
- **Dataset**: Amazon Reviews 2023, Clothing_Shoes_and_Jewelry category (50K products, frozen by organizer)

## Team Contributions

| Role | Member | Contribution |
|---|---|---|
| 01 Retrieval | verakohh | Clause-aware BM25 with IDF weighting, phrase boosting, document frequency scoring |
| 02 State & Routing | verakohh | Boundary broad-reask, compatible constraint override, broad discovery turns |
| 03 NLU | verakohh | Two-tier parser with paraphrase-tolerant fallback, negation awareness |
| 04 Integration | limjeremy496 | Module interfaces, agent.py orchestrator, fallback chain, validation |
| 05 Eval & Report | limjeremy496 | Error analysis tooling, README, results tracking |

---

# Original Challenge README

## TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
