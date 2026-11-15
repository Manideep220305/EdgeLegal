"""Replay saved retrieval rankings under equal RAG excerpt budgets."""
import json
from collections import Counter
from pathlib import Path

from .evaluate_agentic_rag import evidence_hits, write

ROOT = Path(__file__).resolve().parents[2]
RANKINGS = ROOT / "results/oversized-clause-dev/retrieval-reranking.json"
DATA = ROOT / "backend/data/contractnli/dev.json"
BUDGETS = (2000, 4000, 6000)
METHODS = ("bm25", "bge", "hybrid", "reranked")


def evaluate(dataset, rankings):
    documents = {doc["id"]: doc for doc in dataset["documents"]}
    expected = {(doc["id"], hid) for doc in documents.values()
                for hid in doc["annotation_sets"][0]["annotations"]}
    keys = [(row["document_id"], row["hypothesis_id"]) for row in rankings]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Rankings and development pairs differ.")
    totals = {budget: {method: Counter() for method in METHODS} for budget in BUDGETS}
    pairs = []
    for row in rankings:
        doc = documents[row["document_id"]]
        hid = row["hypothesis_id"]
        gold = doc["annotation_sets"][0]["annotations"][hid]
        if row["gold"] != gold["choice"] or row["hypothesis"] != dataset["labels"][hid]["hypothesis"]:
            raise ValueError(f"Rankings have different labels or statements: {doc['id']}/{hid}")
        spans = [doc["text"][start:end].strip() for start, end in
                 (doc["spans"][idx] for idx in gold["spans"])]
        pair = {"doc_id": doc["id"], "hyp_id": hid, "gold": row["gold"],
                "annotated_spans": len(spans), "budgets": {}}
        for budget, methods in totals.items():
            pair["budgets"][budget] = by_method = {}
            for method, count in methods.items():
                sent = 0
                excerpts = {}
                for candidate in row[method][:5]:
                    cid, text = candidate["top_clause_id"], candidate["top_clause"]
                    size = len(text.encode("utf-8"))
                    if cid not in excerpts and sent + size <= budget:
                        excerpts[cid] = text
                        sent += size
                hits = evidence_hits(excerpts.values(), spans)
                by_method[method] = {"hits": hits, "bytes": sent, "clause_ids": list(excerpts)}
                count.update({
                    "cases": 1,
                    "annotated_cases": bool(spans),
                    "any_evidence": hits > 0,
                    "all_evidence": bool(spans) and hits == len(spans),
                    "contradiction_cases": row["gold"] == "Contradiction",
                    "contradiction_evidence": row["gold"] == "Contradiction" and hits > 0,
                    "empty_context": not excerpts,
                    "total_excerpt_bytes": sent,
                })
            baseline = by_method["bm25"]["hits"]
            for method, count in methods.items():
                hits = by_method[method]["hits"]
                count.update({"gain_over_bm25": hits > 0 and baseline == 0,
                              "loss_against_bm25": hits == 0 and baseline > 0})
        pairs.append(pair)
    return totals, pairs


def main():
    dataset = json.loads(DATA.read_bytes())
    saved = json.loads(RANKINGS.read_bytes())
    if saved["split"] != "dev":
        raise ValueError("Expected development rankings.")
    summary, rows = evaluate(dataset, saved["rows"])
    output = ROOT / "results/reference-rag-dev/context_budget.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, {"source": str(RANKINGS.relative_to(ROOT)), "split": "dev",
                   "max_candidates": 5, "summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
