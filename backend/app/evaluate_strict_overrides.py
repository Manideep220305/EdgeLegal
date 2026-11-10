"""Compare stricter override rules using cached ContractNLI development scores."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/oversized-clause-dev/top-three.json"
OUTPUT = SOURCE.with_name("strict-overrides.json")
THRESHOLD = 0.95
RELATIVE_DISTANCE_BAND = 0.15  # Existing Chroma rank-one anchored gap policy.


def select(row):
    candidates = row["candidates"]
    rescue = row["selected_index"]["rescue_095"]
    if not rescue:
        return {"top_one": 0, "rescue_095": 0, "corroborated": 0, "retrieval_band": 0}
    alternate = candidates[rescue]
    # Filter existing overrides so a conflicting answer cannot become eligible
    # simply because its opposing clause was removed by the stricter rule.
    agreement = len(candidates) == 3 and all(
        c["predicted"] == alternate["predicted"] and c["confidence"] >= THRESHOLD
        for c in candidates[1:]
    )
    nearby = alternate["top_distance"] <= candidates[0]["top_distance"] * (1 + RELATIVE_DISTANCE_BAND)
    return {"top_one": 0, "rescue_095": rescue,
            "corroborated": rescue if agreement else 0,
            "retrieval_band": rescue if nearby else 0}


def summarize(rows, rule):
    correct = overrides = fixed = new_errors = wrong_to_wrong = confident_errors = 0
    confusion = defaultdict(int)
    deltas = defaultdict(int)
    for row in rows:
        idx = row["selected_index"][rule]
        candidate = row["candidates"][idx]
        gold = row["gold"]
        label = candidate["predicted"]
        ok = label == gold
        correct += ok
        overrides += idx != 0
        fixed += bool(idx and ok)
        new_errors += bool(idx and gold == "NotMentioned")
        wrong_to_wrong += bool(idx and not ok and gold != "NotMentioned")
        confident_errors += not ok and candidate["confidence"] >= THRESHOLD
        confusion[gold, label] += 1
        reference = row["candidates"][row["selected_index"]["rescue_095"]]["predicted"]
        deltas[row["document_id"]] += int(ok) - int(reference == gold)

    classes = {}
    for label in ("Entailment", "Contradiction", "NotMentioned"):
        tp = confusion[label, label]
        support = sum(n for (gold, _), n in confusion.items() if gold == label)
        predicted = sum(n for (_, actual), n in confusion.items() if actual == label)
        classes[label] = {"correct": tp, "total": support, "f1": 2 * tp / (support + predicted)}

    rng = random.Random(20261110)
    gains = list(deltas.values())
    boot = sorted(100 * sum(rng.choices(gains, k=len(gains))) / len(rows) for _ in range(10_000))
    return {"correct": correct, "accuracy": correct / len(rows), "overrides": overrides,
            "fixed": fixed, "new_errors": new_errors, "wrong_to_wrong": wrong_to_wrong,
            "confidently_wrong": confident_errors, "classes": classes,
            "macro_f1": sum(x["f1"] for x in classes.values()) / len(classes),
            "accuracy_delta_vs_rescue_pp_95ci": [boot[250], boot[9750]]}


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert source["phase"] == "complete" and source["split"] == "dev"
    assert source["threshold"] == THRESHOLD
    rows = source["rows"]
    assert len(rows) == 1037 and len({(r["document_id"], r["hypothesis_id"]) for r in rows}) == 1037
    assert len({r["document_id"] for r in rows}) == 61
    for row in rows:
        row["selected_index"] = select(row)
    summary = {rule: summarize(rows, rule) for rule in rows[0]["selected_index"]}
    for rule, metrics in summary.items():
        assert metrics["overrides"] == metrics["fixed"] + metrics["new_errors"] + metrics["wrong_to_wrong"]
        assert metrics["correct"] == summary["top_one"]["correct"] + metrics["fixed"] - metrics["new_errors"]
        if rule in ("top_one", "rescue_095"):
            assert metrics["correct"] == source["summary"]["rules"][rule]["correct"]
    payload = {"split": "dev", "source": SOURCE.name, "cases": len(rows), "confidence": THRESHOLD,
               "relative_distance_band": RELATIVE_DISTANCE_BAND, "summary": summary,
               "rows": [{k: r[k] for k in ("document_id", "hypothesis_id", "selected_index")} for r in rows]}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
