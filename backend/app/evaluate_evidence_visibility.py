"""Audit evidence visibility and score annotated inputs for saved retrieval errors.

Run with python -m backend.app.evaluate_evidence_visibility.
"""
from __future__ import annotations

import json
from collections import defaultdict

import torch

from backend.app.evaluate_oversized_clauses import INPUT_LIMIT, ROOT, score_cases
from backend.app.retrieval_accuracy_check import gold_span_texts
from backend.app.triage_model.infer import DEFAULT_CHECKPOINT_PATH
from backend.app.triage_model.model import TriageModel

STATES = ("visible", "partial", "truncated", "elsewhere", "unmapped")


def premise_offsets(tokenizer, clause, hypothesis, *, truncate):
    encoded = tokenizer(
        clause, hypothesis, truncation=truncate, max_length=INPUT_LIMIT,
        return_offsets_mapping=True, verbose=False,
    )
    return [
        (start, end) for (start, end), sequence in
        zip(encoded["offset_mapping"], encoded.sequence_ids())
        if sequence == 0 and end > start
    ]


def evidence_state(tokenizer, row, spans):
    """Check tokens overlapping any complete, verbatim annotated span."""
    clause = row["top_clause"]
    ranges = []
    for span in spans:
        span = span.strip()
        if not span:
            continue
        start = clause.find(span)
        while start >= 0:
            ranges.append((start, start + len(span)))
            start = clause.find(span, start + 1)
    if not ranges:
        return "elsewhere" if row["evidence_mapped"] else "unmapped"

    if not row["truncated"]:
        return "visible"

    full = premise_offsets(tokenizer, clause, row["hypothesis"], truncate=False)
    retained = set(premise_offsets(tokenizer, clause, row["hypothesis"], truncate=True))
    partial = False
    for left, right in ranges:
        tokens = {offset for offset in full if offset[1] > left and offset[0] < right}
        if tokens and tokens <= retained:
            return "visible"
        partial |= bool(tokens & retained)
    return "partial" if partial else "truncated"


def summarize(rows):
    summary = {}
    for arm in ("baseline", "split"):
        cases = [r for r in rows if r["arm"] == arm]
        groups = {}
        for state in STATES:
            selected = [r for r in cases if r["evidence_state"] == state]
            wrong = [r for r in selected if not r["correct"]]
            groups[state] = {
                "cases": len(selected),
                "wrong": len(wrong),
                "confidently_wrong": sum(r["confidently_wrong"] for r in wrong),
                "oracle_correct": sum(r.get("oracle", {}).get("correct", False) for r in wrong),
            }
        summary[arm] = {
            "evidence_groups": groups,
            "false_not_mentioned": sum(r["false_not_mentioned"] for r in cases),
            "not_mentioned_cases": sum(r["gold"] == "NotMentioned" for r in cases),
            "not_mentioned_wrong": sum(r["gold"] == "NotMentioned" and not r["correct"] for r in cases),
        }
    return summary


def main():
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    dataset = json.loads((ROOT / "backend/data/contractnli/dev.json").read_text(encoding="utf-8"))
    saved = ROOT / "results/oversized-clause-dev/cases.json"
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["split"] == "dev" and data["input_limit"] == INPUT_LIMIT
    rows = data["rows"]
    docs = {doc["id"]: doc for doc in dataset["documents"]}
    assert len(rows) == 2074 and len({(r["arm"], r["document_id"], r["hypothesis_id"]) for r in rows}) == 2074
    print("Loading fine-tuned DeBERTa for the evidence diagnostic.", flush=True)
    model = TriageModel().load(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=torch.device("cpu"))
    assert model._tokenizer.is_fast, "Visibility requires tokenizer offset mappings."
    cases = defaultdict(list)
    for row in rows:
        doc = docs[row["document_id"]]
        annotation = doc["annotation_sets"][0]["annotations"][row["hypothesis_id"]]
        assert row["gold"] == annotation["choice"]
        spans = gold_span_texts(doc, annotation["spans"])
        row["evidence_state"] = evidence_state(model._tokenizer, row, spans) if spans else None
        if spans:
            assert (row["evidence_state"] in {"visible", "partial", "truncated"}) == row["evidence_hit"]
            if not row["correct"]:
                cases[row["document_id"], row["hypothesis_id"]].append(row)

    output = ROOT / "results/oversized-clause-dev/evidence-visibility.json"
    def save(phase):
        payload = {"phase": phase, "split": "dev", "input_limit": INPUT_LIMIT,
                   "oracle_cases": len(cases), "source": saved.name,
                   "summary": summarize(rows), "rows": rows}
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(output)
    save("visibility_complete")
    print(f"Visibility complete. Rescoring {len(cases)} distinct wrong cases with annotated evidence.", flush=True)
    print(json.dumps(summarize(rows)), flush=True)

    oracle_rows = []
    for (document_id, hypothesis_id), affected in cases.items():
        doc = docs[document_id]
        annotation = doc["annotation_sets"][0]["annotations"][hypothesis_id]
        premise = " ".join(gold_span_texts(doc, annotation["spans"]))
        oracle_rows.append({"top_clause": premise, "hypothesis": affected[0]["hypothesis"],
                            "gold": annotation["choice"]})
    score_cases(oracle_rows, model)
    for affected, oracle in zip(cases.values(), oracle_rows):
        for row in affected:
            row["oracle"] = {key: oracle[key] for key in
                             ("predicted", "confidence", "correct", "input_tokens", "truncated")}
    save("complete")
    print(json.dumps(summarize(rows), indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
