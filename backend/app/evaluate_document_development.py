"""Compare a saved document-model development run against EdgeLegal baselines."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from .held_out_metrics import LABELS, paired_interval

ROOT = Path(__file__).resolve().parents[2]


def keyed(rows, predicted):
    result = {}
    for row in rows:
        key = row["document_id"], row["hypothesis_id"]
        if key in result:
            raise ValueError(f"Duplicate prediction: {key}")
        result[key] = row["gold"], predicted(row)
    return result


def evaluate(predictions, dataset, baselines):
    annotations = {
        (doc["id"], hypothesis_id): (doc, annotation)
        for doc in dataset["documents"]
        for hypothesis_id, annotation in doc["annotation_sets"][0]["annotations"].items()
    }
    rows = predictions["rows"]
    result = keyed(rows, lambda row: row["predicted"])
    if len(result) != 1037 or result.keys() != annotations.keys():
        raise ValueError("Unexpected development cohort.")
    hits = Counter()
    support = Counter()
    evidence = Counter()
    for row in rows:
        key = row["document_id"], row["hypothesis_id"]
        doc, annotation = annotations[key]
        gold, prediction = result[key]
        if gold != annotation["choice"] or prediction not in LABELS:
            raise ValueError(f"Invalid development row: {key}")
        support[gold] += 1
        correct = prediction == gold
        hits[gold] += correct
        start, end = row["evidence_window"]
        if not 0 <= start < end <= len(doc["text"]):
            raise ValueError(f"Invalid evidence offsets: {key}")
        if annotation["spans"]:
            matched = any(start <= left and right <= end
                          for left, right in (doc["spans"][index] for index in annotation["spans"]))
            evidence["annotated_cases"] += 1
            evidence["annotated_span_in_top_window"] += matched
            if not correct:
                evidence["wrong_with_span_in_top_window" if matched
                         else "wrong_without_span_in_top_window"] += 1
    comparisons = {}
    for name, baseline in baselines.items():
        if result.keys() != baseline.keys() or any(result[key][0] != row[0] for key, row in baseline.items()):
            raise ValueError(f"Cohort or labels differ for {name}")
        by_contract = {doc_id: [0, 0] for doc_id, _ in result}
        fixed = harmed = 0
        for key, (gold, prediction) in result.items():
            new_correct, old_correct = prediction == gold, baseline[key][1] == gold
            delta = int(new_correct) - int(old_correct)
            fixed += delta == 1
            harmed += delta == -1
            by_contract[key[0]][0] += delta
            by_contract[key[0]][1] += 1
        comparisons[name] = {"fixed": fixed, "regressed": harmed,
                             "accuracy_delta_pp": 100 * (fixed - harmed) / len(result),
                             "accuracy_delta_pp_95ci": paired_interval(by_contract)}
        if name == "document_with_evidence_supervision":
            reference_hits = Counter(gold for gold, prediction in baseline.values() if gold == prediction)
            comparisons[name]["reference_correct_by_class"] = dict(reference_hits)
            comparisons[name]["contradiction_recall_delta_pp"] = (
                100 * (hits["Contradiction"] - reference_hits["Contradiction"])
                / support["Contradiction"]
            )
    total_correct = hits.total()
    return {
        "epoch": predictions["epoch"], "correct": total_correct, "cases": len(result),
        "accuracy": total_correct / len(result),
        "macro_f1": predictions["metrics"]["macro avg"]["f1-score"],
        "correct_by_class": dict(hits), "class_support": dict(support),
        "false_mentions": support["NotMentioned"] - hits["NotMentioned"],
        "evidence_diagnostic": dict(evidence), "paired_comparisons": comparisons,
        "candidate_passes": total_correct >= 632 and hits["Contradiction"] >= 40
                            and hits["NotMentioned"] >= 220,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supervised-reference", type=Path,
                        help="Compare a matched no-evidence-loss run against the supervised document model.")
    args = parser.parse_args()
    dataset = json.loads((ROOT / "backend/data/contractnli/dev.json").read_bytes())
    retrieval = json.loads((ROOT / "results/oversized-clause-dev/retrieval-reranking.json").read_bytes())
    with ZipFile(ROOT / "clause-combiner.zip") as archive:
        combiner = json.loads(archive.read("clause-combiner/dev_predictions.json"))
    baselines = {
        "five_clause": keyed(combiner, lambda row: row["predictions"]["top_five_logistic"]),
        "bm25": keyed(retrieval["rows"], lambda row: row["bm25"][0]["predicted"]),
        "reranked_hybrid": keyed(retrieval["rows"], lambda row: row["reranked"][0]["predicted"]),
    }
    if args.supervised_reference:
        ablated = json.loads(args.predictions.with_name("manifest.json").read_bytes())
        reference = json.loads(args.supervised_reference.with_name("manifest.json").read_bytes())
        if ablated["settings"] != {**reference["settings"], "evidence_weight": 0}:
            parser.error("Ablation must differ only in evidence loss weight.")
        for key in ("backbone", "source_sha256", "checkpoint_sha256", "train_sha256", "dev_sha256",
                    "torch", "transformers", "selection_metric"):
            if ablated[key] != reference[key]:
                parser.error(f"Ablation and supervised reference differ in {key}.")
        reference_rows = json.loads(args.supervised_reference.read_bytes())["rows"]
        baselines["document_with_evidence_supervision"] = keyed(reference_rows, lambda row: row["predicted"])
    report = evaluate(json.loads(args.predictions.read_bytes()), dataset, baselines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
