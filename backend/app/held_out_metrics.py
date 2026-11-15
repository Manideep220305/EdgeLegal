"""Summarize frozen, paired EdgeLegal held-out predictions."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import f1_score

LABELS = ("Contradiction", "Entailment", "NotMentioned")
ARMS = ("A", "B", "C", "candidate")
BUDGET_ARMS = ("budget_c_confidence", "budget_candidate_confidence")


def paired_interval(by_contract):
    rng = random.Random(42)
    contracts = list(by_contract.values())
    samples = []
    while len(samples) < 10_000:
        draw = rng.choices(contracts, k=len(contracts))
        denominator = sum(total for _, total in draw)
        if denominator:
            samples.append(100 * sum(delta for delta, _ in draw) / denominator)
    samples.sort()
    return [samples[250], samples[9750]]


def summarize(rows):
    if len(rows) != 1751 or len({(r["doc_id"], r["hyp_id"]) for r in rows}) != len(rows):
        raise ValueError("Expected 1,751 unique held-out pairs.")
    gold = [r["gold"] for r in rows]
    if any(label not in LABELS for label in gold):
        raise ValueError("Unknown gold label.")
    counts = Counter(r["doc_id"] for r in rows)
    if len(counts) != 103 or set(counts.values()) != {17}:
        raise ValueError("Expected 103 whole contracts with 17 hypotheses each.")
    support = Counter(gold)
    if any(not support[label] for label in LABELS):
        raise ValueError("Expected all three labels in held-out evaluation.")
    metrics = {}
    budget_present = any(arm in row["arms"] for row in rows for arm in BUDGET_ARMS)
    arms = ARMS + BUDGET_ARMS if budget_present else ARMS
    for arm in arms:
        predictions = [r["arms"][arm]["label"] for r in rows]
        if any(p is not None and p not in LABELS for p in predictions):
            raise ValueError(f"Invalid verdict in arm {arm}.")
        predicted = [p or "Unresolved" for p in predictions]
        hits = Counter(g for g, p in zip(gold, predicted) if g == p)
        metrics[arm] = {
            "correct": hits.total(), "accuracy": hits.total() / len(rows),
            "macro_f1": f1_score(gold, predicted, labels=LABELS, average="macro", zero_division=0),
            "recall": {label: {"correct": hits[label], "total": support[label],
                               "rate": hits[label] / support[label]} for label in LABELS},
            "false_mentions": sum(g == LABELS[2] and p in LABELS[:2] for g, p in zip(gold, predicted)),
            "unresolved": predicted.count("Unresolved"),
            "logical_llm_calls": sum(r["arms"][arm]["called_llm"] for r in rows),
        }
    if metrics["A"]["logical_llm_calls"] != len(rows):
        raise ValueError("Config A must escalate every pair.")
    if budget_present and any(metrics[arm]["logical_llm_calls"] != 1400 for arm in BUDGET_ARMS):
        raise ValueError("The matched-budget arms must each escalate exactly 1,400 pairs.")
    paired = {}
    contrasts = [(arm, "A") for arm in ARMS[1:]]
    if budget_present:
        contrasts.append((BUDGET_ARMS[1], BUDGET_ARMS[0]))
    for arm, baseline in contrasts:
        deltas = defaultdict(int)
        contradictions = {doc_id: [0, 0] for doc_id in counts}
        for r in rows:
            delta = int(r["arms"][arm]["label"] == r["gold"]) - int(r["arms"][baseline]["label"] == r["gold"])
            deltas[r["doc_id"]] += delta
            if r["gold"] == LABELS[0]:
                contradictions[r["doc_id"]][0] += delta
                contradictions[r["doc_id"]][1] += 1
        paired[arm] = {
            "accuracy_delta_pp": 100 * sum(deltas.values()) / len(rows),
            "accuracy_delta_pp_95ci": paired_interval({k: (v, counts[k]) for k, v in deltas.items()}),
            "contradiction_delta_pp": 100 * sum(v[0] for v in contradictions.values()) / support[LABELS[0]],
            "contradiction_delta_pp_95ci": paired_interval(contradictions),
        }
    candidate = paired["candidate"]
    budget_comparison = paired.pop(BUDGET_ARMS[1], None)
    return {
        "metrics": metrics, "paired_vs_a": paired, "matched_budget_comparison": budget_comparison,
        "routing_criteria_met": (
            metrics["candidate"]["logical_llm_calls"] <= 1400
            and candidate["accuracy_delta_pp_95ci"][0] > -2
            and candidate["contradiction_delta_pp"] >= -5
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = summarize(payload["rows"])
    print(json.dumps({**result, "provider_usage": payload["provider_usage"]}, indent=2))


if __name__ == "__main__":
    main()
