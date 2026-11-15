"""Compare BGE with BGE + BM25 on ContractNLI development data.

Run from the repo root with python -m backend.app.evaluate_hybrid_retrieval.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict

import chromadb
import torch

from backend.app.embeddings.embedder import Embedder
from backend.app.evaluate_evidence_visibility import evidence_state
from backend.app.evaluate_oversized_clauses import FROZEN_THRESHOLD, INPUT_LIMIT, ROOT, score_cases
from backend.app.evaluate_top_three import LABELS
from backend.app.parsing.clause_parser import segment_clauses
from backend.app.retrieval_accuracy_check import gold_span_texts, map_gold_to_clauses
from backend.app.test_pipeline import doc_to_page_block
from backend.app.triage_model.infer import DEFAULT_CHECKPOINT_PATH
from backend.app.triage_model.model import TriageModel
from backend.app.vectorstore.bm25 import BM25, BM25_B, BM25_K1, terms
from backend.app.vectorstore.chroma_client import add_clauses, init_collection, query

RESULTS = ROOT / "results/oversized-clause-dev"
SOURCE = RESULTS / "top-three.json"
OUTPUT = RESULTS / "hybrid-retrieval.json"
DEPTHS = (1, 3, 5)
ARMS = ("bge", "hybrid")
RRF_K = 60
PREDICTION_FIELDS = ("predicted", "confidence", "input_tokens", "truncated")


def retrieve(dataset, saved, embedder):
    ids = list(dataset["labels"])
    hypotheses = {key: dataset["labels"][key]["hypothesis"] for key in ids}
    vectors = dict(zip(ids, embedder.embed(list(hypotheses.values()))))
    query_terms = {key: terms(text) for key, text in hypotheses.items()}
    client = chromadb.EphemeralClient()
    rows = []
    for count, document in enumerate(dataset["documents"], 1):
        document_id = document["id"]
        clauses = segment_clauses(doc_to_page_block(document))
        index = init_collection(f"hybrid_{document_id}", client=client)
        add_clauses(index, clauses, embedder.embed([clause.text for clause in clauses]))
        lexical = BM25(clauses)
        for hypothesis_id, annotation in document["annotation_sets"][0]["annotations"].items():
            reference = saved[document_id, hypothesis_id]
            hypothesis = hypotheses[hypothesis_id]
            gold = annotation["choice"]
            assert (reference["hypothesis"], reference["gold"]) == (hypothesis, gold)
            dense = query(index, hypothesis, k=len(clauses),
                          query_embedding=vectors[hypothesis_id], with_scores=True)
            expected = [(c["top_clause_id"], c["top_clause"]) for c in reference["candidates"]]
            full_query_prefix = [(clause.id, clause.text) for clause, _ in dense[:3]]
            if full_query_prefix != expected:
                # Preserve the historical order at a floating-point distance tie.
                by_id = {clause.id: (clause, distance) for clause, distance in dense}
                prefix = [by_id[cid] for cid, _ in expected]
                for ((clause, distance), (_, expected_text), old, (_, current_distance)) in zip(
                    prefix, expected, reference["candidates"], dense,
                ):
                    assert clause.text == expected_text
                    assert math.isclose(distance, old["top_distance"], abs_tol=2e-6)
                    assert math.isclose(distance, current_distance, abs_tol=2e-6), (
                        f"BGE rank drift on {document_id}/{hypothesis_id}: {distance} vs {current_distance}"
                    )
                anchored = {clause.id for clause, _ in prefix}
                dense = prefix + [(clause, distance) for clause, distance in dense
                                  if clause.id not in anchored]
            dense_positions = {clause.id: rank for rank, (clause, _) in enumerate(dense, 1)}
            lexical_ranks = lexical.rank(query_terms[hypothesis_id], dense_positions)
            spans = gold_span_texts(document, annotation["spans"])
            mapped = set(map_gold_to_clauses(spans, clauses))
            candidates = []
            for rank, (clause, distance) in enumerate(dense, 1):
                lexical_rank, lexical_score = lexical_ranks.get(clause.id, (None, 0.0))
                fusion = 1 / (RRF_K + rank)
                if lexical_rank is not None:
                    fusion += 1 / (RRF_K + lexical_rank)
                candidates.append({
                    "top_clause_id": clause.id, "top_clause": clause.text,
                    "dense_rank": rank, "top_distance": distance,
                    "bm25_rank": lexical_rank, "bm25_score": lexical_score,
                    "rrf_score": fusion, "evidence_hit": clause.id in mapped,
                })
            ranked = sorted(candidates, key=lambda c: (-c["rrf_score"], c["dense_rank"]))
            rows.append({"document_id": document_id, "hypothesis_id": hypothesis_id,
                         "hypothesis": hypothesis, "gold": gold, "annotated": bool(spans),
                         "full_query_prefix_changed": full_query_prefix != expected,
                         "evidence_mapped": bool(mapped), "bge": candidates[:max(DEPTHS)],
                         "hybrid": ranked[:max(DEPTHS)]})
        client.delete_collection(index.name)
        if count % 10 == 0 or count == len(dataset["documents"]):
            print(f"Ranked {count}/{len(dataset['documents'])} contracts.", flush=True)
    return rows


def score(rows, dataset, saved, model):
    pending = []
    docs = {doc["id"]: doc for doc in dataset["documents"]}
    for row in rows:
        key = row["document_id"], row["hypothesis_id"]
        cached = {c["top_clause_id"]: c for c in saved[key]["candidates"]}
        for arm in ARMS:
            top = row[arm][0]
            if top["top_clause_id"] in cached:
                top.update({field: cached[top["top_clause_id"]][field] for field in PREDICTION_FIELDS})
            else:
                pending.append((top, {"top_clause": top["top_clause"],
                                      "hypothesis": row["hypothesis"], "gold": row["gold"]}))
    print(f"Scoring {len(pending)} top-one cases with uncached DeBERTa predictions.", flush=True)
    score_cases([item[1] for item in pending], model)
    for top, result in pending:
        top.update({field: result[field] for field in PREDICTION_FIELDS})

    for row in rows:
        doc = docs[row["document_id"]]
        annotation = doc["annotation_sets"][0]["annotations"][row["hypothesis_id"]]
        spans = gold_span_texts(doc, annotation["spans"])
        unique = {c["top_clause_id"]: c for arm in ARMS for c in row[arm]}
        for candidate in unique.values():
            candidate["visible"] = False
            if candidate["evidence_hit"]:
                tokens = model._tokenizer(candidate["top_clause"], row["hypothesis"],
                                          truncation=False, verbose=False)["input_ids"]
                candidate["visible"] = evidence_state(
                    model._tokenizer, {**candidate, "hypothesis": row["hypothesis"],
                                       "truncated": len(tokens) > INPUT_LIMIT,
                                       "evidence_mapped": True}, spans,
                ) == "visible"
        reference = saved[row["document_id"], row["hypothesis_id"]]
        assert all(c["visible"] == (old["evidence_state"] == "visible")
                   for c, old in zip(row["bge"], reference["candidates"]))


def summarize(rows, arm):
    annotated = [row for row in rows if row["annotated"]]
    top = [(row["gold"], row[arm][0]) for row in rows]
    confusion = Counter((gold, c["predicted"]) for gold, c in top)
    per_class = {}
    for label in LABELS:
        correct = confusion[label, label]
        support = sum(n for (gold, _), n in confusion.items() if gold == label)
        predicted = sum(n for (_, verdict), n in confusion.items() if verdict == label)
        per_class[label] = {"correct": correct, "total": support, "f1": 2 * correct / (support + predicted)}
    return {
        "evidence_cases": len(annotated),
        "evidence": {field: {k: sum(any(c[field] for c in row[arm][:k]) for row in annotated)
                             for k in DEPTHS} for field in ("evidence_hit", "visible")},
        "correct": sum(gold == c["predicted"] for gold, c in top),
        "macro_f1": sum(result["f1"] for result in per_class.values()) / len(LABELS),
        "per_class": per_class,
        "truncated": sum(c["truncated"] for _, c in top),
        "confidently_wrong": sum(gold != c["predicted"] and c["confidence"] >= FROZEN_THRESHOLD
                                 for gold, c in top),
        "false_not_mentioned": sum(gold != "NotMentioned" and c["predicted"] == "NotMentioned"
                                   for gold, c in top),
    }


def paired_comparison(rows, baseline, candidate):
    by_document = defaultdict(int)
    fixed = harmed = changed = 0
    for row in rows:
        old, new = row[baseline][0], row[candidate][0]
        old_ok, new_ok = (c["predicted"] == row["gold"] for c in (old, new))
        fixed += new_ok and not old_ok
        harmed += old_ok and not new_ok
        changed += old["top_clause_id"] != new["top_clause_id"]
        by_document[row["document_id"]] += int(new_ok) - int(old_ok)
    rng = random.Random(20261113)
    deltas = list(by_document.values())
    samples = sorted(100 * sum(rng.choices(deltas, k=len(deltas))) / len(rows) for _ in range(10_000))
    return {"top_clause_changed": changed, "errors_fixed": fixed, "new_errors": harmed,
            "accuracy_delta_pp": 100 * (fixed - harmed) / len(rows),
            "accuracy_delta_pp_95ci": [samples[250], samples[9750]]}


def main():
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    dataset = json.loads((ROOT / "backend/data/contractnli/dev.json").read_text(encoding="utf-8"))
    reference = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert reference["phase"] == "complete" and reference["split"] == "dev"
    saved = {(r["document_id"], r["hypothesis_id"]): r for r in reference["rows"]}
    assert len(saved) == len(reference["rows"]) == 1037
    embedder = Embedder()
    embedder.load()
    rows = retrieve(dataset, saved, embedder)
    assert len(rows) == len(saved) and len({r["document_id"] for r in rows}) == 61
    model = TriageModel().load(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=torch.device("cpu"))
    score(rows, dataset, saved, model)
    metrics = {arm: summarize(rows, arm) for arm in ARMS}
    comparison = paired_comparison(rows, "bge", "hybrid")
    comparison["dense_full_query_prefix_changes"] = sum(row["full_query_prefix_changed"] for row in rows)
    assert metrics["bge"]["correct"] == reference["summary"]["rules"]["top_one"]["correct"]
    assert metrics["hybrid"]["correct"] - metrics["bge"]["correct"] == comparison["errors_fixed"] - comparison["new_errors"]
    settings = {"embedder": embedder.model_name, "bm25_k1": BM25_K1, "bm25_b": BM25_B,
                "rrf_k": RRF_K, "input_limit": INPUT_LIMIT, "confidence_threshold": FROZEN_THRESHOLD}
    payload = {"split": "dev", "source": SOURCE.name, "settings": settings,
               "metrics": metrics, "comparison": comparison, "rows": rows}
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps({"metrics": metrics, "comparison": comparison}, indent=2))
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
