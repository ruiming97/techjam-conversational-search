# Ten-Turn Shopper

A multi-turn conversational shopping agent that finds a customer's hidden target product from a 50,000-item Clothing, Shoes & Jewelry catalog in at most 10 turns.

## Results

| Metric | Baseline | Our Agent | Improvement |
|---|---|---|---|
| HitRate@10 | 0.125 | **0.875** | 7.0x |
| MRR | 0.068 | **0.545** | 8.0x |
| MTTC | 9.81 | **3.37** | 65.6% faster |
| TechnicalScore | 0.107 | **0.754** | 7.1x |

Per-scenario: Buying **90%**, Browsing **89%**, Boundary **90%**, Intent Override **77%**.

## Architecture

```
customer message
       |
   [NLU] ── parse constraints, detect intent/override/boundary
       |
   [State] ── track conversation, pick next attribute to ask
       |
   [Retrieval] ── BM25 with phrase boosting over accumulated constraints
       |
   [Validation] ── dedup ASINs, enforce schema, 3-level fallback
       |
   response (message + ask_attribute + recommendations)
```

Five modules in `starter/src/`, wired together by `starter/agent.py`:

| Module | File | Purpose |
|---|---|---|
| Interfaces | `src/interfaces.py` | Shared contracts: Constraint, SessionState, NLUResult, etc. |
| NLU | `src/nlu.py` | Regex parser for evaluator templates, contextual message generation with profile personalization |
| State | `src/state.py` | Conversation state machine, ask strategy (open-ended probing first, then targeted attributes) |
| Retrieval | `src/retrieval.py` | Stateful BM25 with phrase matching over accumulated constraints |
| Validation | `src/validation.py` | Response schema enforcement, ASIN dedup against catalog |
| Fallback | `src/fallback.py` | 3-level fallback chain — module catch, safe BM25, empty response |
| Config | `src/config.py` | Strategy parameters, attribute message templates |
| Catalog | `src/catalog_index.py` | FTS5 index over 50K products, ASIN membership set |

### Key Design Decisions

**Open-ended probing first**: The agent front-loads broad "what matters most?" questions before narrowing to specific attributes. This maximizes information gain per turn — the customer reveals their most important constraints first, regardless of attribute type.

**Stateful constraint accumulation**: Unlike the baseline (which only uses the current turn's text), we accumulate all customer-revealed constraints across the full conversation and search over the combined signal.

**Phrase-boosted BM25**: Multi-word constraints are matched as phrases in FTS5, not just individual terms. This significantly improves precision for specific product features.

**Intent Override handling**: When the customer changes their mind mid-conversation, the agent clears stale constraints and rebuilds the search from the new intent plus the stable category context.

**Zero external dependencies**: Runs on Python stdlib only (sqlite3, json, re, pathlib). No LLM, no embeddings, no pip install. Safe for offline scoring.

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

# 4. Run the error analysis
python3 -m scripts.analyze_results
```

Results are written to `results.json`.

## Limitations and Future Work

- **BM25 ceiling**: ~12.5% of sessions (25/200) are misses where keyword matching can't distinguish the target from similar products. Dense retrieval (embeddings) would directly address this.
- **Intent Override**: Weakest scenario at 77%. After the override, the agent has fewer turns and a generic new constraint (often just a material like "leather"). Semantic reranking would help narrow candidates.
- **Evaluator-specific parsing**: The NLU regex parser is tuned to the evaluator's deterministic customer templates. If the private set uses paraphrased messages, an LLM-based parser would be needed.
- **No personalization beyond tags**: The user profile's preference_tags are used in messages but not in retrieval weighting.

## Tools and Libraries

- **Language**: Python 3.10+
- **Search**: SQLite FTS5 (stdlib) for BM25 full-text search
- **Development**: Claude Code (AI-assisted development)
- **Dataset**: Amazon Reviews 2023, Clothing_Shoes_and_Jewelry category (50K products, frozen)

## Team Contributions

| Role | Member | Contribution |
|---|---|---|
| 04 Agent Integration | limjeremy496 | Module interfaces, agent.py orchestrator, fallback chain, NLU message generation, error analysis, report |
| 01 Catalog & Retrieval | — | BM25 phrase boosting |
| 02 State & Routing | — | Conversation state tracking, override/boundary handling |

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
