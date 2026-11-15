"""Run the frozen matched held-out comparison using a resumable LLM cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from .held_out_metrics import LABELS, summarize
from .llm.prompts import build_reasoning_prompt

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def choose_threshold(confidences):
    """Maximize development escalations without exceeding the 80% budget."""
    if not confidences or any(not 0 <= value <= 1 for value in confidences):
        raise ValueError("Invalid development confidences.")
    budget = 4 * len(confidences) // 5
    ordered = sorted(confidences)
    return ordered[budget]


def score_document(rows, texts, checkpoint, settings, device):
    import torch

    from .triage_model.model import ID2LABEL, TriageModel
    from .triage_model.train_document import DocumentNLI, encode_document

    wrapper = TriageModel().load(checkpoint_path=None, device=device)
    candidate = DocumentNLI(wrapper._model, settings["window_batch"]).to(device)
    candidate.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    candidate.eval()
    with torch.inference_mode():
        for row in rows:
            windows, bounds = encode_document(wrapper._tokenizer, texts[row["doc_id"]],
                                         row["hypothesis"], settings["max_length"], settings["stride"])
            logits, evidence = candidate(windows, wrapper._tokenizer, device)
            confidence, label = logits.softmax(dim=-1).max(dim=-1)
            yield ID2LABEL[label.item()], confidence.item(), bounds[evidence.argmax().item()]


def development_confidences(args, development, settings):
    import torch

    dataset = read(args.data / "dev.json")
    texts = {doc["id"]: doc["text"] for doc in dataset["documents"]}
    expected = {(doc["id"], hid): annotation["choice"] for doc in dataset["documents"]
                for hid, annotation in doc["annotation_sets"][0]["annotations"].items()}
    actual = {(row["document_id"], row["hypothesis_id"]): row["gold"] for row in development}
    if expected != actual:
        raise ValueError("Development predictions differ from the dataset cohort.")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = [{"doc_id": row["document_id"], "hypothesis": dataset["labels"][row["hypothesis_id"]]["hypothesis"]}
              for row in development]
    confidences = []
    for row, (label, score, _) in zip(development, score_document(inputs, texts, args.candidate, settings, device), strict=True):
        if label != row["predicted"]:
            raise ValueError(f"Development checkpoint disagrees on {row['document_id']}/{row['hypothesis_id']}.")
        confidences.append(score)
    return confidences


def held_out_contracts(args):
    data = read(args.data / "test.json")
    excluded = {r["doc_id"] for r in read(args.diagnostic)["rows"]}
    all_ids = {doc["id"] for doc in data["documents"]}
    docs = [doc for doc in data["documents"] if doc["id"] not in excluded]
    if len(excluded) != 20 or not excluded <= all_ids or len(docs) != 103:
        raise ValueError("Unexpected diagnostic or held-out cohort.")
    return data["labels"], docs


def document_only(args, manifest):
    from collections import Counter

    import torch
    from sklearn.metrics import f1_score

    labels, docs = held_out_contracts(args)
    texts = {doc["id"]: doc["text"] for doc in docs}
    rows = [{"doc_id": doc["id"], "hyp_id": hid, "gold": annotation["choice"],
             "hypothesis": labels[hid]["hypothesis"]}
            for doc in docs for hid, annotation in doc["annotation_sets"][0]["annotations"].items()]
    if len(rows) != 1751 or len({(r["doc_id"], r["hyp_id"]) for r in rows}) != 1751:
        raise ValueError("Expected 1,751 unique held-out examples.")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scored = score_document(rows, texts, args.candidate, manifest["document_settings"], device)
    for i, (row, (label, confidence, bounds)) in enumerate(zip(rows, scored, strict=True), 1):
        row.update(predicted=label, confidence=confidence, evidence_window=bounds)
        if i % 200 == 0:
            print(f"Document-only evaluation: {i}/{len(rows)}.", flush=True)
    support = Counter(row["gold"] for row in rows)
    hits = Counter(row["gold"] for row in rows if row["gold"] == row["predicted"])
    summary = {
        "cases": len(rows), "contracts": len(docs), "correct": hits.total(),
        "accuracy": hits.total() / len(rows),
        "macro_f1": f1_score([r["gold"] for r in rows], [r["predicted"] for r in rows],
                             labels=LABELS, average="macro", zero_division=0),
        "per_class": {label: {"correct": hits[label], "total": support[label]} for label in LABELS},
        "false_mentions": support["NotMentioned"] - hits["NotMentioned"],
        "logical_llm_calls": 0, "device": str(device),
    }
    write(args.out / "document_only.json", {"summary": summary, "rows": rows,
                                           "candidate_sha256": manifest["candidate_sha256"],
                                           "test_sha256": manifest["test_sha256"]})
    print(json.dumps(summary, indent=2), flush=True)


def prepare(args, manifest):
    import chromadb
    import torch

    from .embeddings.embedder import Embedder
    from .parsing.clause_parser import PageBlock, segment_clauses
    from .triage_model.model import TriageModel
    from .vectorstore.chroma_client import add_clauses, gap_filter, init_collection, query

    labels, docs = held_out_contracts(args)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Embedder()
    embedder.load()
    vectors = {key: embedder.embed([value["hypothesis"]])[0]
               for key, value in labels.items()}
    db = chromadb.EphemeralClient()
    rows = []
    for index, doc in enumerate(docs, 1):
        clauses = segment_clauses([PageBlock(doc["text"], 0)])
        collection = init_collection(f"heldout_{doc['id']}", client=db)
        add_clauses(collection, clauses, embedder.embed([c.text for c in clauses]))
        for hid, annotation in doc["annotation_sets"][0]["annotations"].items():
            hypothesis = labels[hid]["hypothesis"]
            ranked = query(collection, hypothesis, k=len(clauses), query_embedding=vectors[hid],
                           with_scores=True)
            chosen = gap_filter(ranked)
            context = [clause.text for clause, _ in chosen]
            rows.append({"doc_id": doc["id"], "hyp_id": hid, "gold": annotation["choice"],
                         "hypothesis": hypothesis, "premise": context[0],
                         "prompt": build_reasoning_prompt(context, hypothesis), "arms": {}})
        db.delete_collection(collection.name)
        if index % 10 == 0 or index == len(docs):
            print(f"Prepared {index}/{len(docs)} contracts.", flush=True)
    assert len(rows) == 1751
    for arm, checkpoint in (("B", None), ("C", args.checkpoint)):
        model = TriageModel().load(checkpoint_path=checkpoint, device=device)
        for row in rows:
            label, confidence = model.predict(row["premise"], row["hypothesis"])
            row["arms"][arm] = {"label": label, "confidence": confidence,
                                "called_llm": confidence < 0.95}
        del model
        print(f"Scored {arm}.", flush=True)
    settings = manifest["document_settings"]
    document_results = args.out / "document_only.json"
    if document_results.exists():
        saved = read(document_results)
        if saved.get("candidate_sha256") != manifest["candidate_sha256"] or saved.get("test_sha256") != manifest["test_sha256"]:
            raise ValueError("Saved document-only predictions have different input provenance.")
        predictions = {(r["doc_id"], r["hyp_id"]): r for r in saved["rows"]}
        if len(predictions) != len(rows):
            raise ValueError("Saved document-only cohort differs.")
        for row in rows:
            previous = predictions[row["doc_id"], row["hyp_id"]]
            if (row["gold"], row["hypothesis"]) != (previous["gold"], previous["hypothesis"]):
                raise ValueError("Saved document-only label or hypothesis differs.")
            confidence = previous["confidence"]
            row["arms"]["candidate"] = {"label": previous["predicted"], "confidence": confidence,
                                        "called_llm": confidence < manifest["candidate_threshold"]}
    else:
        texts = {doc["id"]: doc["text"] for doc in docs}
        for row, (label, confidence, _) in zip(rows, score_document(rows, texts, args.candidate, settings, device), strict=True):
            row["arms"]["candidate"] = {"label": label, "confidence": confidence,
                                        "called_llm": confidence < manifest["candidate_threshold"]}
    for row in rows:
        row["arms"]["A"] = {"label": None, "called_llm": True}
        del row["premise"]
    for arm, score_key in (("budget_c_confidence", "C"), ("budget_candidate_confidence", "candidate")):
        ordered = sorted(rows, key=lambda row: (row["arms"][score_key]["confidence"],
                                               row["doc_id"], row["hyp_id"]))
        for index, row in enumerate(ordered):
            row["arms"][arm] = {"label": row["arms"]["C"]["label"], "called_llm": index < 1400}
    return rows


def fill_llm(rows, args, manifest):
    from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

    from .llm.groq_client import groq_client

    if not groq_client.has_keys:
        raise RuntimeError("A configured Groq key is required for the matched run.")
    client = Groq(api_key=groq_client.get_current_key(), max_retries=0, timeout=90)
    cache_path = args.out / "responses.jsonl"
    cache = {}
    usage_path = args.out / "attempts.jsonl"
    def record_attempt(record):
        with usage_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                cache[entry["key"]] = entry
    with cache_path.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            prompt = row["prompt"]
            key = hashlib.sha256(prompt.encode()).hexdigest()
            if key not in cache:
                entry = {"key": key, "label": None, "attempts": 0, "input_tokens": 0, "output_tokens": 0}
                for attempt in range(3):
                    entry["attempts"] += 1
                    try:
                        response = client.chat.completions.create(
                            model=manifest["provider_model"], temperature=0,
                            response_format={"type": "json_object"},
                            messages=[{"role": "user", "content": prompt}],
                        )
                    except RateLimitError as exc:
                        record_attempt({"key": key, "status": "rate_limited"})
                        wait = groq_client._retry_after_seconds(exc, default=60)
                        if wait > 90 or attempt == 2:
                            raise RuntimeError("Provider quota paused the run. Resume with the same output directory.") from exc
                        time.sleep(wait)
                        continue
                    except (APIConnectionError, APIStatusError) as exc:
                        record_attempt({"key": key, "status": type(exc).__name__})
                        if getattr(exc, "status_code", 500) in {401, 403, 404}:
                            raise
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        entry["error"] = type(exc).__name__
                        break
                    if response.usage:
                        entry.update(input_tokens=response.usage.prompt_tokens,
                                     output_tokens=response.usage.completion_tokens)
                    record_attempt({"key": key, "status": "response"})
                    try:
                        payload = json.loads(response.choices[0].message.content or "")
                        label = payload.get("label") if isinstance(payload, dict) else None
                        if label in LABELS:
                            entry["label"] = label
                        else:
                            entry["error"] = "invalid_label"
                    except (json.JSONDecodeError, IndexError):
                        entry["error"] = "invalid_response"
                    break
                cache[key] = entry
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                time.sleep(args.request_interval)
            for arm in row["arms"].values():
                if arm["called_llm"]:
                    arm["label"] = cache[key]["label"]
            if index % 50 == 0:
                print(f"LLM verdicts {index}/{len(rows)}.", flush=True)
    with usage_path.open(encoding="utf-8") as stream:
        attempts = sum(1 for line in stream if line.strip())
    return {**{field: sum(entry[field] for entry in cache.values())
               for field in ("input_tokens", "output_tokens")},
            "attempts": attempts, "retries": attempts - len(cache),
            "unique_prompts": len(cache),
            "unresolved_prompts": sum(entry["label"] is None for entry in cache.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "backend/data/contractnli")
    parser.add_argument("--diagnostic", type=Path, default=ROOT / "results/C_finetuned_triage_20260927T143308+0000.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--provider-model", required=True)
    parser.add_argument("--request-interval", type=float, default=6)
    parser.add_argument("--calibration-manifest", type=Path,
                        help="Reuse the threshold frozen by the document-only GPU run.")
    parser.add_argument("--out", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--document-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.request_interval) or args.request_interval < 0:
        parser.error("Invalid request interval.")
    development = read(args.development)["rows"]
    if (len(development) != 1037
            or len({(r["document_id"], r["hypothesis_id"]) for r in development}) != 1037):
        parser.error("Expected all 1,037 unique development predictions.")
    correct = [r for r in development if r["predicted"] == r["gold"]]
    if (len(correct) < 632 or sum(r["gold"] == "Contradiction" for r in correct) < 40
            or sum(r["gold"] == "NotMentioned" for r in correct) < 220):
        parser.error("Document model did not meet the development gates.")
    model_manifest = read(args.candidate.with_name("manifest.json"))
    if model_manifest["settings"]["smoke"] or model_manifest["dev_sha256"] != digest(args.data / "dev.json"):
        parser.error("Candidate was not evaluated on the expected development split.")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    frozen = read(path) if path.exists() else None
    manifest = {
        "runner_sha256": digest(Path(__file__)), "test_sha256": digest(args.data / "test.json"),
        "diagnostic_sha256": digest(args.diagnostic), "checkpoint_sha256": digest(args.checkpoint),
        "candidate_sha256": digest(args.candidate), "development_sha256": digest(args.development),
        "prompt_source_sha256": digest(Path(__file__).parent / "llm/prompts.py"),
        "threshold_rule": "strictly below the sorted development confidence at index floor(0.8*n)",
        "document_settings": model_manifest["settings"],
        "baseline_threshold": 0.95, "provider": "groq", "provider_model": args.provider_model,
        "temperature": 0, "retriever": "BGE cosine with default gap filter",
    }
    if frozen is not None:
        manifest["candidate_threshold"] = frozen["candidate_threshold"]
        if frozen != manifest:
            raise ValueError("Frozen evaluation inputs differ from the saved manifest.")
    else:
        if args.calibration_manifest:
            calibration = read(args.calibration_manifest)
            for key in ("test_sha256", "diagnostic_sha256", "checkpoint_sha256", "candidate_sha256",
                        "development_sha256", "prompt_source_sha256", "document_settings", "provider_model"):
                if calibration[key] != manifest[key]:
                    raise ValueError(f"Calibration manifest differs on {key}.")
            threshold = calibration["candidate_threshold"]
            if not 0 <= threshold <= 1:
                raise ValueError("Invalid frozen threshold.")
            manifest["candidate_threshold"] = threshold
        else:
            confidences = development_confidences(args, development, model_manifest["settings"])
            manifest["candidate_threshold"] = choose_threshold(confidences)
        write(path, manifest)
    if args.document_only:
        document_only(args, manifest)
        return
    prepared = args.out / "prepared.json"
    if not prepared.exists():
        write(prepared, prepare(args, manifest))
    if not args.prepare_only:
        rows = read(prepared)
        usage = fill_llm(rows, args, manifest)
        for row in rows:
            del row["prompt"]
        write(args.out / "predictions.json", {"rows": rows, "provider_usage": usage})
        summary = {**summarize(rows), "provider_usage": usage}
        write(args.out / "summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
