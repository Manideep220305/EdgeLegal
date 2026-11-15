"""Offline ContractNLI pilot and development evaluation.

Run from the repository root. Use --help for input and output paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", ANONYMIZED_TELEMETRY="False",
                  DO_NOT_TRACK="1")

import chromadb
from huggingface_hub import snapshot_download
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from backend.app.embeddings.embedder import Embedder
from backend.app.evidence_visibility import INPUT_LIMIT, evidence_state
from backend.app.parsing.clause_parser import segment_clauses
from backend.app.retrieval_accuracy_check import gold_span_texts
from backend.app.test_pipeline import doc_to_page_block
from backend.app.triage_model.infer import DEFAULT_CHECKPOINT_PATH
from backend.app.triage_model.model import HF_CHECKPOINT, ID2LABEL, TriageModel
from backend.app.vectorstore.chroma_client import add_clauses, init_collection, query

ROOT = Path(__file__).resolve().parents[2]
LABELS = tuple(ID2LABEL[i] for i in range(3))
BGE = "BAAI/bge-small-en"
K = 5
BATCH = 16
SPLITS = {"train": 423, "dev": 61}
BASELINE = "results/oversized-clause-dev/cases.json"
TARGETS = {"overall": 574, "Contradiction": 39, "NotMentioned": 220}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, payload):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot_hashes(snapshot):
    return {path.relative_to(snapshot).as_posix(): sha256(path)
            for path in sorted(snapshot.rglob("*")) if path.is_file() and path.name != "README.md"}


def make_manifest(args, bge, deberta):
    source_files = (
        "evaluate_clause_combiner.py", "parsing/clause_parser.py", "embeddings/embedder.py",
        "vectorstore/chroma_client.py", "triage_model/model.py", "triage_model/dataset.py",
        "evidence_visibility.py",
        "retrieval_accuracy_check.py", "test_pipeline.py",
    )
    return {
        "inputs": {str(path.resolve()): sha256(path) for path in
                   [args.data / f"{split}.json" for split in SPLITS] + [args.checkpoint, args.baseline]},
        "source_sha256": {name: sha256(ROOT / "backend/app" / name) for name in source_files},
        "snapshots": {name: {"revision": path.name, "sha256": snapshot_hashes(path)}
                      for name, path in ((BGE, bge), (HF_CHECKPOINT, deberta))},
        "versions": {name: version(name) for name in
                     ("torch", "transformers", "sentence-transformers", "chromadb", "scikit-learn",
                      "numpy", "scipy", "tokenizers", "huggingface-hub", "joblib")},
        "python": sys.version,
        "retrieval": {"k": K, "index": "Chroma HNSW", "metric": "cosine",
                      "normalized_embeddings": True, "hypothesis_embedding_batch": 1,
                      "clause_embedding_batch": 32, "similarity_gap_filter": False,
                      "split_oversized": False},
        "inference": {"device": "cpu", "threads": args.threads, "batch_size": BATCH,
                      "max_pair_tokens": INPUT_LIMIT, "label_order": LABELS},
        "regression": {"scaling": "StandardScaler", "regularization": "L2", "C": 1.0,
                       "solver": "lbfgs", "max_iter": 2000, "tuning": False},
        "dev_targets": TARGETS,
        "missing_slot": {"probabilities": [0.0, 0.0, 0.0], "distance": 0.0,
                         "truncated": False, "present": False},
    }


def verify_dataset(data, split):
    ids = [doc["id"] for doc in data["documents"]]
    hypotheses = set(data["labels"])
    if len(ids) != SPLITS[split] or len(set(ids)) != len(ids) or len(hypotheses) != 17:
        raise ValueError(f"Unexpected {split} contract or hypothesis count")
    for doc in data["documents"]:
        annotations = doc["annotation_sets"][0]["annotations"]
        if set(annotations) != hypotheses or any(a["choice"] not in LABELS for a in annotations.values()):
            raise ValueError(f"Invalid annotations for {split} contract {doc['id']}")
    return set(ids)


def score(rows, model):
    tokenizer = model._tokenizer
    groups = {}
    for row in rows:
        for slot in row["slots"]:
            key = (slot["clause_text"], row["hypothesis"])
            groups.setdefault(key, []).append((row, slot))
    inputs = []
    for (clause, hypothesis), slots in groups.items():
        tokens = len(tokenizer(clause, hypothesis, verbose=False)["input_ids"])
        encoded = tokenizer(clause, hypothesis, truncation=True, max_length=INPUT_LIMIT)
        inputs.append((tokens, encoded, slots))
    inputs.sort(key=lambda item: item[0])
    with torch.inference_mode():
        for start in range(0, len(inputs), BATCH):
            batch = inputs[start:start + BATCH]
            encoded = tokenizer.pad([item[1] for item in batch], padding=True, return_tensors="pt")
            probabilities = model._model(**encoded).logits.softmax(dim=-1).tolist()
            for (tokens, _, matches), probs in zip(batch, probabilities, strict=True):
                if not all(0 <= p <= 1 for p in probs) or not math.isclose(sum(probs), 1, abs_tol=1e-6):
                    raise ValueError(f"Invalid clause probabilities {probs}")
                verdict = LABELS[max(range(3), key=probs.__getitem__)]
                truncated = tokens > INPUT_LIMIT
                for row, slot in matches:
                    slot.update(probabilities=probs, verdict=verdict, truncated=truncated)
                    slot["evidence_state"] = evidence_state(tokenizer, {
                        "top_clause": slot["clause_text"], "hypothesis": row["hypothesis"],
                        "truncated": truncated, "evidence_mapped": row["evidence_mapped"],
                    }, row["gold_spans"]) if row["gold_spans"] else None


def generate_contract(doc, labels, clauses, embedder, vectors, db, model):
    rows = []
    collection = init_collection("clause_combiner_contract", client=db) if clauses else None
    try:
        if clauses:
            add_clauses(collection, clauses, embedder.embed([clause.text for clause in clauses]))
        for hypothesis_id, annotation in doc["annotation_sets"][0]["annotations"].items():
            hypothesis = labels[hypothesis_id]["hypothesis"]
            ranked = query(collection, hypothesis, k=K, query_embedding=vectors[hypothesis],
                           with_scores=True) if clauses else []
            if len(ranked) != min(K, len(clauses)) or any(not math.isfinite(d) for _, d in ranked):
                raise ValueError(f"Unexpected retrieval result for {doc['id']}/{hypothesis_id}")
            spans = gold_span_texts(doc, annotation["spans"])
            rows.append({
                "document_id": doc["id"], "hypothesis_id": hypothesis_id, "hypothesis": hypothesis,
                "gold": annotation["choice"], "gold_spans": spans,
                "evidence_mapped": any(span.strip() and span.strip() in c.text
                                       for span in spans for c in clauses),
                "slots": [{"present": True, "distance": distance, "clause_id": clause.id,
                           "clause_text": clause.text} for clause, distance in ranked],
            })
        score(rows, model)
        for row in rows:
            row["slots"].extend({"present": False, "probabilities": [0.0, 0.0, 0.0],
                                 "distance": 0.0, "truncated": False}
                                for _ in range(K - len(row["slots"])))
    finally:
        if collection is not None:
            db.delete_collection(collection.name)
    return rows


def check_baseline(rows, baseline):
    for row in rows:
        old = baseline[row["document_id"], row["hypothesis_id"]]
        slot = row["slots"][0]
        saved = (old["top_clause_id"], old["top_clause"], old["predicted"], old["gold"], old["hypothesis"])
        current = (slot.get("clause_id"), slot.get("clause_text"), slot.get("verdict"),
                   row["gold"], row["hypothesis"])
        if current != saved:
            raise ValueError(f"Saved rank-one mismatch for {row['document_id']}/{row['hypothesis_id']}: "
                             f"{current[0:1] + current[2:]} vs {saved[0:1] + saved[2:]}")


def generate(split, data, clauses, documents, args, embedder, vectors, db, model, baseline):
    directory = args.out / "contracts" / split
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for index, doc in enumerate(documents, 1):
        path = directory / f"{doc['id']}.json"
        if path.exists():
            result = read_json(path)
            if result["input_manifest_sha256"] != args.manifest_hash:
                raise ValueError(f"Checkpoint provenance mismatch at {path}")
            if result["document_id"] != doc["id"]:
                raise ValueError(f"Contract checkpoint mismatch at {path}")
        else:
            start = perf_counter()
            rows = generate_contract(doc, data["labels"], clauses[doc["id"]], embedder, vectors, db, model)
            result = {"document_id": doc["id"], "clause_count": len(clauses[doc["id"]]),
                      "seconds": perf_counter() - start, "rows": rows,
                      "input_manifest_sha256": args.manifest_hash}
            if split == "dev":
                check_baseline(rows, baseline)
            save_json(path, result)
            print(f"{split} {index}/{len(documents)} contract {doc['id']}: {result['seconds']:.2f}s",
                  flush=True)
        if split == "dev":
            check_baseline(result["rows"], baseline)
        results.append(result)
    return results


def pilot(dev, clauses, args, embedder, vectors, db, model, baseline, startup):
    documents = sorted(dev["documents"], key=lambda doc: (len(clauses[doc["id"]]), doc["id"]))
    # Sample evenly across clause counts, including the shortest and longest contracts.
    sample = [documents[round(i * (len(documents) - 1) / 9)] for i in range(10)]
    start = perf_counter()
    results = generate("dev", dev, clauses, sample, args, embedder, vectors, db, model, baseline)
    rows = [row for result in results for row in result["rows"]]
    present = [slot for row in rows for slot in row["slots"] if slot["present"]]
    expected = sum(min(K, result["clause_count"]) * 17 for result in results)
    if len(present) != expected or any(len(row["slots"]) != K for row in rows):
        raise ValueError("Pilot slot count mismatch")
    seconds = sum(result["seconds"] for result in results)
    report = {
        "manifest_sha256": args.manifest_hash,
        "contracts": [{key: value for key, value in result.items() if key not in
                       ("rows", "input_manifest_sha256")} for result in results],
        "rows": len(rows), "rank_one_matches": len(rows), "present_slots": len(present),
        "padded_slots": len(rows) * K - expected,
        "max_probability_sum_error": max(abs(sum(s["probabilities"]) - 1) for s in present),
        "startup_seconds": startup, "pilot_wall_seconds": perf_counter() - start,
        "contract_seconds": seconds, "estimated_full_seconds": startup + seconds * 484 / 10,
        "estimate_method": "Mean pilot contract time times 484 plus startup",
    }
    save_json(args.out / "pilot.json", report)
    print(json.dumps(report, indent=2), flush=True)


def flatten(slots):
    return [value for slot in slots
            for value in (*slot["probabilities"], slot["distance"], int(slot["truncated"]), int(slot["present"]))]


def export_features(args, split, results):
    rows = [row for result in results for row in result["rows"]]
    pairs = {(row["document_id"], row["hypothesis_id"]) for row in rows}
    if len(rows) != SPLITS[split] * 17 or len(pairs) != len(rows):
        raise ValueError(f"Incorrect {split} row count")
    columns = [f"rank{rank}_{name}" for rank in range(1, K + 1) for name in
               (*(f"p_{label}" for label in LABELS), "distance", "truncated", "present")]
    save_json(args.out / f"{split}_features.json", {
        "manifest_sha256": args.manifest_hash, "feature_columns": columns,
        "rows": [{"document_id": r["document_id"], "hypothesis_id": r["hypothesis_id"],
                  "target": r["gold"], "features": flatten(r["slots"])} for r in rows],
    })
    return rows


def error_category(row, predicted, k):
    if predicted == row["gold"]:
        return "correct"
    if row["gold"] == "NotMentioned":
        return "false_mention"
    visible = [s for s in row["slots"][:k] if s.get("evidence_state") == "visible"]
    if not visible:
        return "annotated_evidence_not_visible" if row["gold_spans"] else "no_annotated_evidence"
    if all(s["verdict"] != row["gold"] for s in visible):
        return "visible_evidence_wrong_deberta_verdict"
    return "visible_evidence_correct_deberta_wrong_combiner"


def evaluate(args, train, dev):
    gold = [r["gold"] for r in dev]
    if (gold.count("Contradiction"), gold.count("NotMentioned")) != (95, 423):
        raise ValueError("Development target denominators do not match")
    predictions = {"rank_one_deberta": [r["slots"][0]["verdict"] for r in dev]}
    for name, k in (("rank_one_logistic", 1), ("top_five_logistic", K)):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))
        model.fit([flatten(r["slots"][:k]) for r in train], [r["gold"] for r in train])
        path = args.out / f"{name}.joblib"
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        joblib.dump(model, temporary)
        temporary.replace(path)
        predictions[name] = model.predict([flatten(r["slots"][:k]) for r in dev]).tolist()
    summaries = {}
    for name, predicted in predictions.items():
        k = K if name == "top_five_logistic" else 1
        hits = Counter(g for g, p in zip(gold, predicted) if g == p)
        summaries[name] = {
            "correct": hits.total(), "accuracy": hits.total() / len(gold),
            "macro_f1": f1_score(gold, predicted, labels=LABELS, average="macro"),
            "recall": {label: {"correct": hits[label], "total": gold.count(label),
                               "rate": hits[label] / gold.count(label)} for label in LABELS},
            "targets_met": {"overall": hits.total() >= TARGETS["overall"], **{
                label: hits[label] >= TARGETS[label] for label in ("Contradiction", "NotMentioned")}},
            "error_counts": dict(Counter(error_category(row, p, k) for row, p in zip(dev, predicted))),
        }
    evidence = [r for r in dev if r["gold_spans"]]
    visibility = {k: sum(any(s.get("evidence_state") == "visible" for s in r["slots"][:k])
                         for r in evidence) for k in (1, 3, 5)}
    for i, row in enumerate(dev):
        row["predictions"] = {name: values[i] for name, values in predictions.items()}
        row["diagnoses"] = {name: error_category(row, values[i], K if name == "top_five_logistic" else 1)
                            for name, values in predictions.items()}
    save_json(args.out / "dev_predictions.json", dev)
    top, first = summaries["top_five_logistic"], summaries["rank_one_logistic"]
    eligible = (all(top["targets_met"].values()) and top["correct"] > first["correct"]
                and top["macro_f1"] > first["macro_f1"]
                and top["correct"] > summaries["rank_one_deberta"]["correct"])
    comparison = {
        "manifest_sha256": args.manifest_hash, "cases": len(dev), "metrics": summaries,
        "evidence_visibility": {"hits": visibility, "annotated_cases": len(evidence)},
        "top_five_vs_rank_one_logistic": {
            "fixed": sum(p == g != q for g, p, q in zip(gold, predictions["top_five_logistic"],
                                                       predictions["rank_one_logistic"])),
            "regressed": sum(p != g == q for g, p, q in zip(gold, predictions["top_five_logistic"],
                                                           predictions["rank_one_logistic"])),
        },
        "proceed_to_held_out": eligible,
    }
    save_json(args.out / "comparison.json", comparison)
    if eligible:
        save_json(args.out / "held_out_protocol.json", {
            "candidate": "top_five_logistic", "manifest_sha256": args.manifest_hash,
            "model_sha256": sha256(args.out / "top_five_logistic.joblib"),
            "evaluation": "One pass on all held-out contracts with frozen retrieval and model; "
                          "no fitting or tuning; report paired accuracy, macro F1, class recall "
                          "and annotated evidence visibility against frozen rank-one DeBERTa.",
        })
    lines = ["# ContractNLI development comparison", "",
             "| Method | Correct | Accuracy | Macro F1 | " + " | ".join(LABELS) + " recall |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metrics in summaries.items():
        recall = " | ".join(f"{m['correct']}/{m['total']}" for m in metrics["recall"].values())
        lines.append(f"| {name} | {metrics['correct']}/{len(dev)} | {metrics['accuracy']:.2%} "
                     f"| {metrics['macro_f1']:.4f} | {recall} |")
    lines += ["", "Annotated evidence visible after tokenization among cases with gold spans.", ""]
    lines += [f"- Rank {k} or earlier: {count}/{len(evidence)}" for k, count in visibility.items()]
    lines += ["", f"Held-out evaluation justified under the recorded criteria: {eligible}.", "",
              "Error categories are diagnostic proxies based on visible annotated spans and clause verdicts."]
    (args.out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "backend/data/contractnli")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--baseline", type=Path, default=ROOT / BASELINE)
    parser.add_argument("--out", type=Path, default=ROOT / "results/clause-combiner")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.INFO)
    start = perf_counter()
    torch.set_num_threads(args.threads)
    data = {split: read_json(args.data / f"{split}.json") for split in SPLITS}
    train_ids, dev_ids = (verify_dataset(data[split], split) for split in SPLITS)
    if train_ids & dev_ids:
        raise ValueError("Training and development document IDs overlap")
    baseline = {(r["document_id"], r["hypothesis_id"]): r for r in read_json(args.baseline)["rows"]
                if r["arm"] == "baseline"}
    expected = {(doc["id"], key) for doc in data["dev"]["documents"] for key in data["dev"]["labels"]}
    if set(baseline) != expected:
        raise ValueError("Saved development baseline pairs differ")
    bge, deberta = (Path(snapshot_download(name, local_files_only=True)) for name in (BGE, HF_CHECKPOINT))
    manifest = make_manifest(args, bge, deberta)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    if path.exists():
        if read_json(path) != json.loads(json.dumps(manifest)):
            raise ValueError(f"Inputs or settings changed. Use a new output directory instead of {args.out}")
    else:
        save_json(path, manifest)
    args.manifest_hash = sha256(path)
    embedder = Embedder()
    embedder.load(str(bge))
    model = TriageModel(hf_checkpoint=str(deberta)).load(args.checkpoint, device=torch.device("cpu"))
    model.eval_mode()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vectors = {}
    for dataset in data.values():
        for label in dataset["labels"].values():
            hypothesis = label["hypothesis"]
            if hypothesis not in vectors:
                vectors[hypothesis] = embedder.embed([hypothesis])[0]
    db = chromadb.EphemeralClient()
    clauses = {doc["id"]: segment_clauses(doc_to_page_block(doc))
               for dataset in data.values() for doc in dataset["documents"]}
    if not (args.out / "pilot.json").exists() or args.pilot_only:
        pilot(data["dev"], clauses, args, embedder, vectors, db, model, baseline, perf_counter() - start)
    if not args.pilot_only:
        rows = {}
        for split in SPLITS:
            results = generate(split, data[split], clauses, data[split]["documents"],
                               args, embedder, vectors, db, model, baseline)
            rows[split] = export_features(args, split, results)
        evaluate(args, rows["train"], rows["dev"])


if __name__ == "__main__":
    main()
