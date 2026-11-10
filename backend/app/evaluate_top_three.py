"""Evaluate top-three clause scoring on ContractNLI development cases without an LLM.

Run from the repo root with python -m backend.app.evaluate_top_three.
"""
from __future__ import annotations

import json

import chromadb
import torch

from backend.app.embeddings.embedder import Embedder
from backend.app.evaluate_evidence_visibility import evidence_state
from backend.app.evaluate_oversized_clauses import ROOT, FROZEN_THRESHOLD, INPUT_LIMIT, score_cases
from backend.app.parsing.clause_parser import segment_clauses
from backend.app.retrieval_accuracy_check import gold_span_texts
from backend.app.test_pipeline import doc_to_page_block
from backend.app.triage_model.infer import DEFAULT_CHECKPOINT_PATH
from backend.app.triage_model.model import TriageModel
from backend.app.vectorstore.chroma_client import add_clauses, init_collection, query

SOURCE = ROOT / "results/oversized-clause-dev/evidence-visibility.json"
OUTPUT = ROOT / "results/oversized-clause-dev/top-three.json"
LABELS = ("Entailment", "Contradiction", "NotMentioned")
CANDIDATE_FIELDS = ("top_clause_id", "top_clause", "top_distance", "predicted",
                    "confidence", "input_tokens", "truncated", "evidence_state")


def save(phase, rows, summary=None):
    payload = {"phase": phase, "split": "dev", "source": SOURCE.name,
               "threshold": FROZEN_THRESHOLD, "rows": rows, "summary": summary}
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)


def retrieve(dataset, baseline):
    embedder = Embedder()
    embedder.load()
    labels = dataset["labels"]
    vectors = {key: embedder.embed([value["hypothesis"]])[0] for key, value in labels.items()}
    db = chromadb.EphemeralClient()
    rows = []
    for i, doc in enumerate(dataset["documents"], 1):
        doc_id = doc["id"]
        clauses = segment_clauses(doc_to_page_block(doc))
        collection = init_collection(f"top_three_{doc_id}", client=db)
        add_clauses(collection, clauses, embedder.embed([clause.text for clause in clauses]))
        annotations = doc["annotation_sets"][0]["annotations"]
        for hypothesis_id, annotation in annotations.items():
            old = baseline[doc_id, hypothesis_id]
            hypothesis = labels[hypothesis_id]["hypothesis"]
            assert old["gold"] == annotation["choice"] and old["hypothesis"] == hypothesis
            ranked = query(collection, hypothesis, k=min(3, len(clauses)),
                           query_embedding=vectors[hypothesis_id], with_scores=True)
            top, distance = ranked[0]
            assert (top.id, top.text) == (old["top_clause_id"], old["top_clause"]), (
                f"Rank-one mismatch on {doc_id}/{hypothesis_id}"
            )
            candidates = [{key: old[key] for key in CANDIDATE_FIELDS}]
            candidates.extend({"top_clause_id": clause.id, "top_clause": clause.text,
                               "top_distance": distance} for clause, distance in ranked[1:])
            rows.append({"document_id": doc_id, "hypothesis_id": hypothesis_id, "hypothesis": hypothesis,
                         "gold": old["gold"], "candidates": candidates})
        db.delete_collection(collection.name)
        if i % 10 == 0 or i == len(dataset["documents"]):
            print(f"Retrieved up to three clauses for {i}/{len(dataset['documents'])} contracts.", flush=True)
    return rows


def score_extra(rows, dataset):
    groups = {}
    for row in rows:
        for candidate in row["candidates"][1:]:
            key = candidate["top_clause"], row["hypothesis"]
            groups.setdefault(key, []).append((row, candidate))
    pending = [group for group in groups.values() if "predicted" not in group[0][1]]
    if not pending:
        return
    model = TriageModel().load(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=torch.device("cpu"))
    docs = {doc["id"]: doc for doc in dataset["documents"]}
    tokenizer = model._tokenizer
    pending.sort(key=lambda group: len(group[0][1]["top_clause"]))
    for offset in range(0, len(pending), 128):
        batch = pending[offset:offset + 128]
        score_rows = [{"top_clause": g[0][1]["top_clause"], "hypothesis": g[0][0]["hypothesis"],
                       "gold": g[0][0]["gold"]} for g in batch]
        score_cases(score_rows, model)
        for group, score in zip(batch, score_rows):
            for row, candidate in group:
                doc = docs[row["document_id"]]
                annotation = doc["annotation_sets"][0]["annotations"][row["hypothesis_id"]]
                spans = gold_span_texts(doc, annotation["spans"])
                candidate.update({key: score[key] for key in
                                  ("predicted", "confidence", "input_tokens", "truncated")})
                candidate["evidence_state"] = evidence_state(
                    tokenizer, {**candidate, "hypothesis": row["hypothesis"], "evidence_mapped": True}, spans
                ) if spans else None
        save("scoring", rows)
        print(f"Scored {min(offset + len(batch), len(pending))}/{len(pending)} remaining unique extra inputs.", flush=True)


def choose_candidate(candidates, threshold):
    """Only replace Not Mentioned with a confident, non-conflicting alternate."""
    first = candidates[0]
    if first["predicted"] != "NotMentioned":
        return 0
    alternatives = [(i, c) for i, c in enumerate(candidates[1:], 1)
                    if c["predicted"] != "NotMentioned" and c["confidence"] >= threshold]
    if not alternatives or len({c["predicted"] for _, c in alternatives}) > 1:
        return 0
    return max(alternatives, key=lambda item: item[1]["confidence"])[0]


def summarize(rows):
    evidence = [r for r in rows if r["gold"] != "NotMentioned"]
    result = {
        "cases": len(rows),
        "contracts": len({r["document_id"] for r in rows}),
        "evidence_cases": len(evidence),
        "visible_top_one": sum(r["candidates"][0]["evidence_state"] == "visible" for r in evidence),
        "visible_top_three": sum(any(c["evidence_state"] == "visible" for c in r["candidates"])
                                 for r in evidence),
        "rules": {},
    }
    for name, threshold in (("top_one", None), ("rescue_095", FROZEN_THRESHOLD),
                            ("rescue_090", 0.9), ("rescue_098", 0.98)):
        chosen = [choose_candidate(r["candidates"], threshold) if threshold is not None else 0 for r in rows]
        predicted = [r["candidates"][i] for r, i in zip(rows, chosen)]
        pairs = list(zip(rows, predicted))
        per_class = {}
        for label in LABELS:
            support = sum(r["gold"] == label for r in rows)
            true_positive = sum(r["gold"] == c["predicted"] == label for r, c in pairs)
            positive = sum(c["predicted"] == label for c in predicted)
            per_class[label] = {"correct": true_positive, "total": support,
                                "f1": 2 * true_positive / (support + positive)}
        result["rules"][name] = {
            "correct": sum(r["gold"] == c["predicted"] for r, c in pairs),
            "overrides": sum(i > 0 for i in chosen),
            "fixed": sum(i > 0 and r["gold"] == c["predicted"] for i, (r, c) in zip(chosen, pairs)),
            "new_errors": sum(i > 0 and r["gold"] == "NotMentioned" for i, r in zip(chosen, rows)),
            "wrong_to_wrong": sum(i > 0 and r["gold"] not in ("NotMentioned", c["predicted"])
                                  for i, (r, c) in zip(chosen, pairs)),
            "false_not_mentioned": sum(r["gold"] != "NotMentioned" and c["predicted"] == "NotMentioned"
                                       for r, c in pairs),
            "confidently_wrong": sum(r["gold"] != c["predicted"] and c["confidence"] >= FROZEN_THRESHOLD
                                     for r, c in pairs),
            "macro_f1": sum(m["f1"] for m in per_class.values()) / len(LABELS),
            "per_class": per_class,
        }
        for row, index in zip(rows, chosen):
            row.setdefault("selected_index", {})[name] = index
    return result


def main():
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    dataset = json.loads((ROOT / "backend/data/contractnli/dev.json").read_text(encoding="utf-8"))
    if OUTPUT.exists():
        saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
        rows = saved["rows"]
        print(f"Resuming from {saved['phase']}.", flush=True)
    else:
        data = json.loads(SOURCE.read_text(encoding="utf-8"))
        assert data["phase"] == "complete" and data["input_limit"] == INPUT_LIMIT
        baseline = {(r["document_id"], r["hypothesis_id"]): r for r in data["rows"] if r["arm"] == "baseline"}
        rows = retrieve(dataset, baseline)
        save("retrieval_complete", rows)
    assert len(rows) == 1037 and len({(r["document_id"], r["hypothesis_id"]) for r in rows}) == 1037
    score_extra(rows, dataset)
    summary = summarize(rows)
    save("complete", rows, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
