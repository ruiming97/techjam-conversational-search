"""Demo replay: pretty-prints a single session step-by-step.

Usage:
    python3 -m scripts.demo_session                          # random session
    python3 -m scripts.demo_session --session public_0006    # specific session
    python3 -m scripts.demo_session --scenario browsing      # random from scenario
    python3 -m scripts.demo_session --slow                   # pause between turns
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"

MAX_TURNS = 10
TOP_K = 10


def load_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def truncate(text: str, length: int = 80) -> str:
    return text[:length] + "..." if len(text) > length else text


def demo(
    sample: dict,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    agent: Agent,
    slow: bool = False,
) -> None:
    target = str(sample["ground_truth"]["parent_asin"])
    product = products[target]
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    cat = coarse_category(categories.get(target, []))

    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  Session: {sample['sample_id']}  |  Scenario: {sample['scenario_type']}  |  Difficulty: {sample.get('difficulty_bucket', '?')}{RESET}")
    print(f"{'=' * 70}")
    print()
    print(f"  {DIM}Target:{RESET} {truncate(str(product.get('title', '?')), 60)}")
    print(f"  {DIM}ASIN:{RESET}   {target}")
    print(f"  {DIM}Price:{RESET}  {product.get('price', 'N/A')}")
    print(f"  {DIM}Categories:{RESET} {' > '.join(product.get('categories', []))}")
    print()
    print(f"  {DIM}Profile:{RESET} {sample['user_profile'].get('summary', 'N/A')}")
    print()
    print(f"  {DIM}Hidden constraints:{RESET}")
    for label, items in [("Hard", card["hard_constraints"]), ("Soft", card["soft_preferences"])]:
        for item in items:
            print(f"    {label}: {truncate(item, 65)}")
    print()

    if sample["scenario_type"] == "intent_override":
        override = behavior.get("override", {})
        print(f"  {YELLOW}Override fires on turn {override.get('turn', '?')}: old='{truncate(str(override.get('old_value', '')), 40)}' -> new='{truncate(str(override.get('new_value', '')), 40)}'{RESET}")
        print()

    print(f"{BOLD}{'─' * 70}{RESET}")

    session_id = f"demo_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective, cat, disclosed)

    for turn in range(1, MAX_TURNS + 1):
        if slow:
            time.sleep(1.5)

        print()
        print(f"  {CYAN}{BOLD}Turn {turn}{RESET}")
        print(f"  {CYAN}Customer:{RESET} {user_message}")

        response = agent.respond(session_id, user_message, turn, TOP_K)

        ask = response.get("ask_attribute")
        msg = response.get("message", "")
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)

        ask_label = f"[ask: {ask}]" if ask else "[no ask]"
        print(f"  {MAGENTA}Agent:{RESET}    {truncate(msg, 55)} {DIM}{ask_label}{RESET}")

        print(f"  {DIM}Top 5:{RESET}  ", end="")
        for i, asin in enumerate(ranked[:5]):
            if asin == target:
                print(f"{GREEN}>>> {asin} (rank {i+1}) <<<{RESET} ", end="")
            else:
                title = truncate(str(products.get(asin, {}).get("title", "?")), 25)
                print(f"{DIM}{asin[:7]}..{RESET} ", end="")
        if not ranked:
            print(f"{DIM}(none){RESET}", end="")
        print()

        if override_applied and target in ranked:
            rank = ranked.index(target) + 1
            print()
            print(f"  {GREEN}{BOLD}>>> HIT at turn {turn}, rank {rank} <<<{RESET}")
            print()
            break

        if turn == MAX_TURNS:
            print()
            print(f"  {RED}{BOLD}MISS — target not found in 10 turns{RESET}")
            print()
            break

        override_info = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override_info.get("turn", 3)):
            override_applied = True
            new_value = str(override_info.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override_info.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    print(f"{'─' * 70}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: replay a single session with pretty output")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--session", default=None, help="Specific session ID (e.g., public_0006)")
    parser.add_argument("--scenario", default=None, help="Pick random session of this scenario type")
    parser.add_argument("--slow", action="store_true", help="Pause between turns for narration")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    if args.session:
        matches = [s for s in samples if s["sample_id"] == args.session]
        if not matches:
            print(f"Session '{args.session}' not found.")
            sys.exit(1)
        sample = matches[0]
    elif args.scenario:
        matches = [s for s in samples if s["scenario_type"] == args.scenario]
        if not matches:
            print(f"No sessions with scenario '{args.scenario}'.")
            sys.exit(1)
        sample = random.choice(matches)
    else:
        sample = random.choice(samples)

    demo(sample, catalog_ids, categories, products, agent, slow=args.slow)


if __name__ == "__main__":
    main()
