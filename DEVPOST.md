# Ten-Turn Shopper — Devpost Description

## Inspiration

Shopping online shouldn't feel like a keyword guessing game. When you walk into a store, a good salesperson doesn't make you fill out a form — they ask *"what are you looking for?"*, listen, then ask one or two follow-ups before pointing you to the right shelf. We wanted to build an agent that works the same way: listen broadly first, narrow quickly, and surface the right product in as few exchanges as possible.

## What it does

Ten-Turn Shopper is a conversational shopping agent that finds a customer's target product from a 50,000-item clothing catalog in an average of **3 turns** — without using an LLM. Given a customer's opening message ("I'm looking for basketball shorts"), the agent asks targeted follow-up questions, accumulates preferences across the conversation, and returns ranked recommendations that improve with each exchange.

It handles four real shopping scenarios:
- **Buying** (customer knows what they want) — 90% success, avg 2.8 turns
- **Browsing** (customer is exploring) — 96% success, avg 2.6 turns
- **Intent Override** (customer changes their mind mid-conversation) — 90% success
- **Boundary** (customer has no preference for a asked attribute) — 100% success

Overall: **93% hit rate**, **TechnicalScore 0.827** (7.7x over the provided baseline).

## How we built it

We started with the weak BM25 starter (12.5% hit rate) and identified two critical gaps: the baseline was *stateless* (threw away all prior context each turn) and never asked clarifying questions (wasting every turn).

Our approach is a 5-module pipeline:

1. **NLU** — Parses customer messages into structured constraints using a two-tier regex system. Strict patterns handle the known evaluator templates; a paraphrase-tolerant fallback layer (with negation detection — "no polyester" won't be misread as a polyester preference) adds robustness for the private evaluation set.

2. **Conversation State** — Tracks constraints across turns, detects buying vs. browsing intent, and decides what to ask next. The key insight: open-ended questions ("what matters most?") extract 2x more information per turn than specific attribute questions ("what material?"), because customers reveal their strongest preferences first regardless of category.

3. **Clause-Aware Retrieval** — Each customer constraint is a separate search clause weighted by inverse document frequency. "Triple Moon Pentagram" (1 matching product) outscores "cotton" (5,000 products). Products satisfying all clauses rank above partial matches — this is what pushes the exact target to rank 1 in 61% of successful sessions.

4. **Validation & Fallback** — Schema enforcement, ASIN deduplication, and a 3-level exception chain. The agent never crashes, even on malformed input.

5. **Integration** — Clean module interfaces let each component be developed and tested independently.

## Challenges we ran into

- **Ask strategy matters more than retrieval quality.** Our biggest score jump (12.5% → 85.5%) came not from improving search, but from changing *what the agent asks*. Asking "budget" first (0% of constraints in the dataset) vs. open-ended "other" (reveals any 2 constraints) was the difference between wasting 2 turns and extracting all preferences by turn 3.

- **Intent override is tricky.** When a customer says "actually, forget leather — I want cotton", naively clearing all prior constraints throws away useful context (their color and size preferences are still valid). We built a compatibility check that preserves constraints unless they directly contradict the new intent.

- **BM25 has a hard ceiling.** 14 sessions (7%) are products that keyword matching simply cannot distinguish from near-identical items. Dense retrieval (embeddings) would help but adds a dependency we wanted to avoid for offline scoring.

## What we learned

- **Information-theoretic question design beats better search.** The single most impactful decision was asking the right questions in the right order — not improving the retrieval algorithm. Open-ended probing first, targeted narrowing second.
- **Constraints are structured data, not free text.** Treating each customer statement as an independent search clause with IDF weighting was 4x more effective than flattening everything into one OR query.
- **Zero-dependency systems are surprisingly competitive.** Python's stdlib (sqlite3 FTS5) gets you to 0.827 TechnicalScore. The marginal gain from adding embeddings or an LLM is real but small compared to the strategy improvements.

## What's next

- **Dense retrieval** via sentence-transformers to address the 7% BM25 miss ceiling
- **Recommendation explanations** — telling the customer *why* each product was suggested
- **Profile-weighted retrieval** — using the customer's preference tags to boost search weights, not just as a fallback signal

## Built with

- Python 3.10+ (standard library only — sqlite3, json, re, math)
- SQLite FTS5 for BM25 full-text search
- Claude Code for AI-assisted development
- Amazon Reviews 2023 dataset (Clothing, Shoes & Jewelry — 50K products)
