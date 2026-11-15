"""Audit one-hop contract references with saved ContractNLI development rankings.

Run from the repo root: python -m backend.app.evaluate_clause_references
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.parsing.clause_parser import PageBlock, segment_clauses

from backend.app.parsing.clause_references import index_headings, resolve_references

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/contractnli/dev.json"
RESULTS = ROOT / "results/oversized-clause-dev"
def matching_spans(text, spans):
    return [index for index, span in spans.items() if span and span in text]


def audit(dataset, rankings):
    saved = {(row["document_id"], row["hypothesis_id"]): row for row in rankings}
    assert len(saved) == len(rankings)
    rows = []
    for document in dataset["documents"]:
        document_id = document["id"]
        text = document["text"]
        clauses = segment_clauses([PageBlock(text, 0)])
        by_id = {clause.id: clause for clause in clauses}
        headings = index_headings(clauses)
        lookups = {}
        for hypothesis_id, annotation in document["annotation_sets"][0]["annotations"].items():
            row = saved[document_id, hypothesis_id]
            ranked = row["candidates"]
            assert row["gold"] == annotation["choice"]
            assert row["hypothesis"] == dataset["labels"][hypothesis_id]["hypothesis"]
            assert all(by_id[c["top_clause_id"]].text == c["top_clause"] for c in ranked)
            source = by_id[ranked[0]["top_clause_id"]]
            if source.id not in lookups:
                lookups[source.id] = resolve_references(source, headings)
            references, targets = lookups[source.id]
            spans = {}
            for index in annotation["spans"]:
                start, end = document["spans"][index]
                spans[index] = text[start:end].strip()
            rows.append({
                "document_id": document_id, "hypothesis_id": hypothesis_id, "gold": row["gold"],
                "annotated": bool(spans), "rank_one_clause": source.id,
                "rank_one_verdict": ranked[0]["predicted"],
                "rank_one_spans": matching_spans(source.text, spans),
                "top_three_spans": sorted({index for c in ranked for index in
                                           matching_spans(c["top_clause"], spans)}),
                "references": references,
                "targets": [{"clause_id": clause.id, "heading": clause.section_heading,
                             "spans": matching_spans(clause.text, spans)} for clause in targets],
            })
    assert len(rows) == len(saved)
    return rows


def summarize(rows):
    mentioned = [row for row in rows if row["references"]]
    resolved = [row for row in mentioned if row["targets"]]
    annotated = [row for row in rows if row["annotated"]]
    rank_one_misses = [row for row in resolved if row["annotated"] and not row["rank_one_spans"]]
    first_lookup = [row for row in rank_one_misses if row["targets"][0]["spans"]]
    any_lookup = [row for row in rank_one_misses if any(target["spans"] for target in row["targets"])]
    return {
        "cases": len(rows), "annotated_cases": len(annotated),
        "cases_with_references": len(mentioned),
        "wrong_cases_with_references": sum(row["rank_one_verdict"] != row["gold"] for row in mentioned),
        "cases_with_a_resolved_target": len(resolved),
        "contracts_with_a_resolved_target": len({row["document_id"] for row in resolved}),
        "wrong_cases_with_a_resolved_target": sum(row["rank_one_verdict"] != row["gold"] for row in resolved),
        "annotated_rank_one_misses_with_a_resolved_target": len(rank_one_misses),
        "rank_one_evidence_hits": sum(bool(row["rank_one_spans"]) for row in annotated),
        "first_lookup_recoveries": len(first_lookup),
        "any_lookup_recoveries": len(any_lookup),
        "top_three_evidence_hits": sum(bool(row["top_three_spans"]) for row in annotated),
        "recoveries_beyond_top_three_first_lookup": sum(not row["top_three_spans"] for row in first_lookup),
        "recoveries_beyond_top_three_any_lookup": sum(not row["top_three_spans"] for row in any_lookup),
        "first_lookup_recoveries_by_label": {
            label: sum(row["gold"] == label for row in first_lookup)
            for label in ("Entailment", "Contradiction")
        },
        "wrong_verdicts_among_first_lookup_recoveries": sum(row["rank_one_verdict"] != row["gold"] for row in first_lookup),
    }


def main():
    dataset = json.loads(DATA.read_text(encoding="utf-8"))
    saved = json.loads((RESULTS / "top-three.json").read_text(encoding="utf-8"))
    assert saved["split"] == "dev" and saved["phase"] == "complete"
    rows = audit(dataset, saved["rows"])
    summary = summarize(rows)
    output = RESULTS / "clause-reference-audit.json"
    output.write_text(json.dumps({"split": "dev", "source": "top-three.json",
                                 "summary": summary, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
