"""Evaluate BM25 alone and reranking the saved Hybrid top five on ContractNLI dev."""
from __future__ import annotations

import json
import math

import torch
from sentence_transformers import CrossEncoder

from backend.app.evaluate_evidence_visibility import evidence_state
from backend.app.evaluate_hybrid_retrieval import (
    BM25, DEPTHS, INPUT_LIMIT, OUTPUT as HYBRID_RESULTS, PREDICTION_FIELDS, RESULTS, ROOT,
    SOURCE as TOP_THREE_RESULTS, paired_comparison, score_cases, summarize, terms,
)
from backend.app.parsing.clause_parser import segment_clauses
from backend.app.retrieval_accuracy_check import gold_span_texts, map_gold_to_clauses
from backend.app.test_pipeline import doc_to_page_block
from backend.app.triage_model.infer import DEFAULT_CHECKPOINT_PATH
from backend.app.triage_model.model import TriageModel

ARMS = ("bge", "bm25", "hybrid", "reranked")
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
OUTPUT = RESULTS / "retrieval-reranking.json"


def prepare(dataset, rows):
    documents = {doc["id"]: doc for doc in dataset["documents"]}
    clauses_by_doc = {key: segment_clauses(doc_to_page_block(doc)) for key, doc in documents.items()}
    indexes = {key: BM25(clauses) for key, clauses in clauses_by_doc.items()}
    positions = {key: {clause.id: i for i, clause in enumerate(clauses)}
                 for key, clauses in clauses_by_doc.items()}
    queries = {key: terms(label["hypothesis"]) for key, label in dataset["labels"].items()}
    for row in rows:
        key, hypothesis_id = row["document_id"], row["hypothesis_id"]
        doc = documents[key]
        annotation = doc["annotation_sets"][0]["annotations"][hypothesis_id]
        assert row["gold"] == annotation["choice"]
        assert row["hypothesis"] == dataset["labels"][hypothesis_id]["hypothesis"]
        spans = gold_span_texts(doc, annotation["spans"])
        clauses = clauses_by_doc[key]
        by_id = {clause.id: clause for clause in clauses}
        assert all(by_id[c["top_clause_id"]].text == c["top_clause"]
                   for arm in ("bge", "hybrid") for c in row[arm])
        ranks = indexes[key].rank(queries[hypothesis_id], positions[key])
        # Keep a deterministic first clause even when no terms match.
        ordered = sorted(clauses, key=lambda clause: ranks.get(clause.id, (math.inf, 0))[0])
        matches = set(map_gold_to_clauses(spans, clauses))
        row["bm25"] = [
            {"top_clause_id": clause.id, "top_clause": clause.text,
             "bm25_score": ranks.get(clause.id, (None, 0.0))[1],
             "evidence_hit": clause.id in matches}
            for clause in ordered[:max(DEPTHS)]
        ]
    return documents


def rerank(rows):
    model = CrossEncoder(RERANKER, revision=RERANKER_REVISION, max_length=INPUT_LIMIT, device="cpu")
    pairs = list(dict.fromkeys((row["hypothesis"], c["top_clause"])
                               for row in rows for c in row["hybrid"]))
    scores = {}
    for start in range(0, len(pairs), 128):
        batch = pairs[start:start + 128]
        values = model.predict(batch, batch_size=16, show_progress_bar=False,
                               activation_fn=torch.nn.Identity())
        scores.update(zip(batch, map(float, values)))
        if start % 512 == 0 or start + len(batch) == len(pairs):
            print(f"Reranked {start + len(batch)}/{len(pairs)} unique statement–clause pairs.",
                  flush=True)
    assert all(math.isfinite(value) for value in scores.values())
    for row in rows:
        ranked = [{**c, "reranker_score": scores[row["hypothesis"], c["top_clause"]]}
                  for c in row["hybrid"]]
        row["reranked"] = sorted(ranked, key=lambda c: -c["reranker_score"])
    return len(pairs)


def evaluate_verdicts(rows, documents, prior):
    pending = []
    cache = {}
    for row in rows:
        key = row["document_id"], row["hypothesis_id"]
        for candidate in prior[key]["candidates"] + row["bge"][:1] + row["hybrid"][:1]:
            fields = {field: candidate[field] for field in PREDICTION_FIELDS}
            cache[candidate["top_clause"], row["hypothesis"]] = fields
    for row in rows:
        for arm in ("bm25", "reranked"):
            top = row[arm][0]
            key = top["top_clause"], row["hypothesis"]
            if key in cache:
                top.update(cache[key])
            else:
                pending.append((top, {"top_clause": key[0], "hypothesis": key[1], "gold": row["gold"]}))
    model = TriageModel().load(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=torch.device("cpu"))
    unique = len({(result["top_clause"], result["hypothesis"]) for _, result in pending})
    print(f"Scoring {unique} uncached DeBERTa inputs.", flush=True)
    score_cases([result for _, result in pending], model)
    for top, result in pending:
        top.update({field: result[field] for field in PREDICTION_FIELDS})
    for row in rows:
        doc = documents[row["document_id"]]
        annotation = doc["annotation_sets"][0]["annotations"][row["hypothesis_id"]]
        spans = gold_span_texts(doc, annotation["spans"])
        for candidate in row["bm25"]:
            candidate["visible"] = False
            if candidate["evidence_hit"]:
                tokenizer = model._tokenizer
                tokens = tokenizer(candidate["top_clause"], row["hypothesis"], truncation=False,
                                   verbose=False)["input_ids"]
                candidate["visible"] = evidence_state(
                    tokenizer, {**candidate, "hypothesis": row["hypothesis"],
                                "truncated": len(tokens) > INPUT_LIMIT, "evidence_mapped": True}, spans,
                ) == "visible"
    return unique


def main():
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    dataset = json.loads((ROOT / "backend/data/contractnli/dev.json").read_text(encoding="utf-8"))
    hybrid = json.loads(HYBRID_RESULTS.read_text(encoding="utf-8"))
    top_three = json.loads(TOP_THREE_RESULTS.read_text(encoding="utf-8"))
    assert hybrid["split"] == top_three["split"] == "dev" and top_three["phase"] == "complete"
    rows = hybrid["rows"]
    prior = {(row["document_id"], row["hypothesis_id"]): row for row in top_three["rows"]}
    keys = {(row["document_id"], row["hypothesis_id"]) for row in rows}
    assert len(rows) == len(prior) == len(keys) == 1037 and keys == prior.keys()
    documents = prepare(dataset, rows)
    rerank_count = rerank(rows)
    verdict_count = evaluate_verdicts(rows, documents, prior)
    metrics = {arm: summarize(rows, arm) for arm in ARMS}
    for arm in ("bge", "hybrid"):
        assert json.loads(json.dumps(metrics[arm])) == hybrid["metrics"][arm]
    for field in ("evidence_hit", "visible"):
        assert metrics["hybrid"]["evidence"][field][5] == metrics["reranked"]["evidence"][field][5]
    comparisons = {
        f"{candidate}_vs_{baseline}": paired_comparison(rows, baseline, candidate)
        for baseline, candidate in (("bge", "bm25"), ("bm25", "hybrid"), ("hybrid", "reranked"))
    }
    settings = {**hybrid["settings"], "bm25_tie_break": "document order",
                "reranker": RERANKER, "reranker_revision": RERANKER_REVISION,
                "reranker_input_limit": INPUT_LIMIT, "rerank_depth": max(DEPTHS),
                "reranker_unique_inputs": rerank_count, "deberta_unique_uncached_inputs": verdict_count,
                "groq_calls": 0}
    summary = {"metrics": metrics, "comparisons": comparisons,
               "bm25_no_term_matches": sum(row["bm25"][0]["bm25_score"] == 0 for row in rows)}
    payload = {"split": "dev", "source": HYBRID_RESULTS.name, "settings": settings, **summary,
               "rows": rows}
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(summary, indent=2))
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
