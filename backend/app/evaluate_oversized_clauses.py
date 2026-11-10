"""Compare the original and size-limited parsers on ContractNLI development data.

Run from the project root with python -m backend.app.evaluate_oversized_clauses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import chromadb
import torch

from backend.app.embeddings.embedder import Embedder
from backend.app.experiment_runner import FROZEN_THRESHOLD
from backend.app.parsing.clause_parser import segment_clauses
from backend.app.retrieval_accuracy_check import gold_span_texts, map_gold_to_clauses
from backend.app.test_pipeline import doc_to_page_block
from backend.app.triage_model.infer import DEFAULT_CHECKPOINT_PATH
from backend.app.triage_model.model import ID2LABEL, TriageModel
from backend.app.vectorstore.chroma_client import add_clauses, init_collection, query

INPUT_LIMIT = 512  # Same limit as TriageModel.forward.
ARMS = ("baseline", "split")
ROOT = Path(__file__).resolve().parents[2]


def retrieve_cases(dataset, embedder):
    db = chromadb.EphemeralClient()
    rows = []
    for i, doc in enumerate(dataset["documents"], 1):
        for arm in ARMS:
            collection = init_collection(f"parser_{arm}_{doc['id']}", client=db)
            clauses = segment_clauses(doc_to_page_block(doc), split_oversized=arm == "split")
            if not clauses:
                raise ValueError(f"No clauses for document {doc['id']} ({arm})")
            add_clauses(collection, clauses, embedder.embed([clause.text for clause in clauses]))
            annotations = doc["annotation_sets"][0]["annotations"]
            for hypothesis_id, annotation in annotations.items():
                hypothesis = dataset["labels"][hypothesis_id]["hypothesis"]
                top, distance = query(collection, hypothesis, k=1, embedder=embedder, with_scores=True)[0]
                spans = gold_span_texts(doc, annotation["spans"])
                matched = map_gold_to_clauses(spans, clauses)
                rows.append({
                    "arm": arm, "document_id": doc["id"], "hypothesis_id": hypothesis_id,
                    "hypothesis": hypothesis, "gold": annotation["choice"],
                    "clause_count": len(clauses), "top_clause_id": top.id,
                    "top_clause": top.text, "top_distance": distance,
                    "evidence_mapped": bool(matched) if spans else None,
                    "evidence_hit": top.id in matched if spans else None,
                })
            db.delete_collection(collection.name)
        if i % 10 == 0 or i == len(dataset["documents"]):
            print(f"Retrieved both arms for {i}/{len(dataset['documents'])} contracts.", flush=True)
    return rows


def score_cases(rows, model):
    tokenizer = model._tokenizer
    by_input = {}
    for row in rows:
        by_input.setdefault((row["top_clause"], row["hypothesis"]), []).append(row)
    inputs = []
    for (clause, hypothesis), matching_rows in by_input.items():
        count = len(tokenizer(clause, hypothesis, truncation=False, verbose=False)["input_ids"])
        encoded = tokenizer(clause, hypothesis, truncation=True, max_length=INPUT_LIMIT)
        inputs.append((count, encoded, matching_rows))
    # Batch similar lengths. Identical inputs across arms share a prediction.
    inputs.sort(key=lambda item: item[0])
    with torch.inference_mode():
        for start in range(0, len(inputs), 8):
            batch = inputs[start:start + 8]
            encoded = tokenizer.pad([item[1] for item in batch], padding=True, return_tensors="pt")
            logits = model._model(**{key: value.to(model.device) for key, value in encoded.items()}).logits
            scores, labels = logits.softmax(dim=-1).max(dim=-1)
            for (count, _, matching_rows), label, score in zip(batch, labels.tolist(), scores.tolist()):
                for row in matching_rows:
                    predicted = ID2LABEL[label]
                    correct = predicted == row["gold"]
                    row.update(input_tokens=count, truncated=count > INPUT_LIMIT,
                               predicted=predicted, confidence=score, correct=correct,
                               confidently_wrong=not correct and score >= FROZEN_THRESHOLD,
                               false_not_mentioned=predicted == "NotMentioned" and not correct)
            if start % 128 == 0 or start + len(batch) == len(inputs):
                print(f"Scored {start + len(batch)}/{len(inputs)} unique classifier inputs.", flush=True)


def summarize(rows):
    summary = {}
    for arm in ARMS:
        cases = [row for row in rows if row["arm"] == arm]
        evidence = [row for row in cases if row["evidence_hit"] is not None]
        summary[arm] = {
            "cases": len(cases),
            "evidence_cases": len(evidence),
            "evidence_hits": sum(row["evidence_hit"] for row in evidence),
            "evidence_unmapped": sum(not row["evidence_mapped"] for row in evidence),
            "truncated": sum(row["truncated"] for row in cases),
            "correct": sum(row["correct"] for row in cases),
            "confidently_wrong": sum(row["confidently_wrong"] for row in cases),
            "false_not_mentioned": sum(row["false_not_mentioned"] for row in cases),
            "confident_false_not_mentioned": sum(
                row["false_not_mentioned"] and row["confidently_wrong"] for row in cases
            ),
            "mean_input_tokens": round(mean(row["input_tokens"] for row in cases), 1),
        }
    return summary


def write_report(out, rows):
    summary = summarize(rows)
    old, new = (summary[arm] for arm in ARMS)
    criteria = {
        "evidence_hits_increase": new["evidence_hits"] > old["evidence_hits"],
        "truncation_decreases": new["truncated"] < old["truncated"],
        "accuracy_does_not_decline": new["correct"] >= old["correct"],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps({
        "split": "dev", "retrieval_k": 1, "input_limit": INPUT_LIMIT,
        "threshold": FROZEN_THRESHOLD, "summary": summary, "criteria": criteria, "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = (
        ("Top-one annotated evidence hits", "evidence_hits", "evidence_cases"),
        ("Unmapped annotated evidence", "evidence_unmapped", "evidence_cases"),
        ("Inputs over 512 tokens", "truncated", "cases"),
        ("Correct predictions", "correct", "cases"),
        ("Confidently wrong predictions", "confidently_wrong", "cases"),
        ("False Not Mentioned", "false_not_mentioned", "cases"),
        ("Confident false Not Mentioned", "confident_false_not_mentioned", "cases"),
    )
    lines = [
        "# Oversized-clause parser comparison", "",
        f"Compared {old['cases']:,} paired ContractNLI development cases using top-one dense retrieval "
        "and the same fine-tuned DeBERTa model. No Groq calls were made.", "",
        "| Measure | Baseline | Split |", "|---|---:|---:|",
    ]
    for title, key, denominator in metrics:
        cells = [f"{x[key]:,}/{x[denominator]:,} ({x[key] / x[denominator]:.1%})" for x in (old, new)]
        lines.append(f"| {title} | {' | '.join(cells)} |")
    lines.append(f"| Mean tokens before truncation | {old['mean_input_tokens']} | {new['mean_input_tokens']} |")
    decision = "Met all three criteria; eligible to carry forward." if all(criteria.values()) else (
        "Did not meet all three criteria; keep the baseline for the next experiment."
    )
    lines += [
        "", f"**Decision.** {decision}", "",
        "Evidence rates use cases with annotated evidence. Unmapped spans count as misses. "
        f"Confidence uses the existing {FROZEN_THRESHOLD:.2f} threshold. These are development "
        "results and the parser option remains off by default.", "",
        "[Per-case data](cases.json)", "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "oversized-clause-dev")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    dataset = json.loads((ROOT / "backend/data/contractnli/dev.json").read_text(encoding="utf-8"))
    embedder = Embedder()
    embedder.load()
    rows = retrieve_cases(dataset, embedder)
    pairs = [{(r["document_id"], r["hypothesis_id"]) for r in rows if r["arm"] == arm} for arm in ARMS]
    if pairs[0] != pairs[1] or len(pairs[0]) != 1037 or len(rows) != 2074:
        raise ValueError("Expected the same 1,037 development pairs in both arms.")
    model = TriageModel()
    model.load(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device="cpu")
    score_cases(rows, model)
    write_report(args.out, rows)


if __name__ == "__main__":
    main()
