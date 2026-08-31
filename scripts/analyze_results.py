"""Role 5: Error analysis tool for the Ten-Turn Shopper agent.

Run after evaluator to diagnose misses and track score progression.

Usage:
    python3 -m scripts.analyze_results [--results results.json] [--dataset data/public_set.jsonl] [--catalog data/catalog.jsonl]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(results_path: str, dataset_path: str, catalog_path: str | None) -> None:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    sessions = results["sessions"]
    samples = {s["sample_id"]: s for s in load_jsonl(dataset_path)}

    catalog = {}
    if catalog_path and Path(catalog_path).exists():
        for line in Path(catalog_path).open(encoding="utf-8"):
            p = json.loads(line)
            catalog[p["parent_asin"]] = p

    print("=" * 60)
    print("TEN-TURN SHOPPER — EVALUATION REPORT")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'Sample count':<20} {results['sample_count']:>10}")
    print(f"{'HitRate@10':<20} {results['hit_rate_at_10']:>10.3f}")
    print(f"{'MRR':<20} {results['mrr']:>10.3f}")
    print(f"{'MTTC':<20} {results['mttc']:>10.2f}")
    print(f"{'Efficiency':<20} {results['efficiency']:>10.3f}")
    print(f"{'TechnicalScore':<20} {results['recommended_technical_score']:>10.3f}")

    usage = results.get("reported_token_usage", {})
    total_tokens = usage.get("total_tokens", 0)
    print(f"{'Total tokens':<20} {total_tokens:>10}")

    print("\n--- Per-Scenario Breakdown ---\n")
    print(f"{'Scenario':<18} {'N':>4} {'HitRate':>8} {'MRR':>8} {'MTTC':>6}")
    print("-" * 48)
    for name, metrics in sorted(results.get("scenario_metrics", {}).items()):
        print(
            f"{name:<18} {metrics['sample_count']:>4} "
            f"{metrics['hit_rate_at_10']:>8.3f} "
            f"{metrics['mrr']:>8.3f} "
            f"{metrics['mttc']:>6.2f}"
        )

    hits = [s for s in sessions if s["hit"]]
    misses = [s for s in sessions if not s["hit"]]

    print(f"\n--- Hit Distribution ---\n")
    print(f"Total hits: {len(hits)} / {len(sessions)} ({len(hits)/len(sessions)*100:.1f}%)")
    if hits:
        turn_dist = Counter(s["first_hit_turn"] for s in hits)
        print(f"\nHits by turn:")
        for turn in sorted(turn_dist):
            bar = "#" * turn_dist[turn]
            print(f"  Turn {turn:>2}: {turn_dist[turn]:>3} {bar}")

        rank_dist = Counter(s["best_rank"] for s in hits)
        print(f"\nHits by rank:")
        for rank in sorted(rank_dist):
            bar = "#" * rank_dist[rank]
            print(f"  Rank {rank:>2}: {rank_dist[rank]:>3} {bar}")

    print(f"\n--- Miss Analysis ({len(misses)} sessions) ---\n")
    if misses:
        miss_by_scenario = Counter(s["scenario_type"] for s in misses)
        for scenario, count in miss_by_scenario.most_common():
            print(f"  {scenario}: {count} misses")

        miss_by_difficulty = Counter(
            samples.get(s["sample_id"], {}).get("difficulty_bucket", "unknown")
            for s in misses
        )
        print(f"\nMisses by difficulty:")
        for diff, count in miss_by_difficulty.most_common():
            print(f"  {diff}: {count}")

        if catalog:
            print(f"\nMissed products:")
            for s in misses:
                sample = samples.get(s["sample_id"], {})
                asin = sample.get("ground_truth", {}).get("parent_asin", "")
                product = catalog.get(asin, {})
                title = str(product.get("title", "unknown"))[:65]
                diff = sample.get("difficulty_bucket", "?")
                print(f"  [{s['scenario_type']}/{diff}] {s['sample_id']}: {title}")
    else:
        print("  No misses — perfect hit rate.")

    baseline = {"hit_rate_at_10": 0.125, "mrr": 0.068034, "mttc": 9.81, "recommended_technical_score": 0.10671}
    print(f"\n--- vs Baseline ---\n")
    print(f"{'Metric':<20} {'Baseline':>10} {'Current':>10} {'Change':>10}")
    print("-" * 52)
    for key, bl_val in baseline.items():
        cur_val = results.get(key, 0)
        label = key.replace("recommended_", "").replace("_", " ").title()
        if key == "mttc":
            change = f"{((bl_val - cur_val) / bl_val * 100):+.1f}%"
        else:
            change = f"{cur_val / bl_val:.1f}x"
        print(f"{label:<20} {bl_val:>10.3f} {cur_val:>10.3f} {change:>10}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()
    analyze(args.results, args.dataset, args.catalog)


if __name__ == "__main__":
    main()
