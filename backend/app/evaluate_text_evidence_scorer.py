"""Train a clause-text evidence scorer on ContractNLI train and test cached dev overrides."""
from __future__ import annotations

import json
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoTokenizer

from backend.app.evaluate_strict_overrides import ROOT, SOURCE, select, summarize
from backend.app.parsing.clause_parser import PageBlock, segment_clauses

OUTPUT = SOURCE.with_name("text-evidence-scorer.json")
CHECKPOINT = "cross-encoder/nli-deberta-v3-base"
INPUT_LIMIT = 512
CUTOFF = 0.5
MAX_NEGATIVES = 3


def words(text):
    return set(re.findall(r"\b\w+\b", text.lower()))


def visible_text(tokenizer, clause, hypothesis):
    encoded = tokenizer(
        clause, hypothesis, truncation=True, max_length=INPUT_LIMIT,
        return_offsets_mapping=True, verbose=False,
    )
    end = max((end for (_, end), seq in zip(encoded["offset_mapping"], encoded.sequence_ids())
               if seq == 0), default=0)
    return clause[:end]


def train_examples(dataset, tokenizer):
    examples = []
    for i, doc in enumerate(dataset["documents"], 1):
        clauses = [clause.text for clause in segment_clauses([PageBlock(doc["text"], 0)])]
        word_sets = [words(clause) for clause in clauses]
        annotations = doc["annotation_sets"][0]["annotations"]
        for hid, ann in annotations.items():
            hypothesis = dataset["labels"][hid]["hypothesis"]
            query_words = words(hypothesis)
            spans = [doc["text"][a:b].strip() for a, b in (doc["spans"][j] for j in ann["spans"])]
            spans = [span for span in spans if span]
            positive = [j for j, clause in enumerate(clauses) if any(span in clause for span in spans)]
            negative = sorted(set(range(len(clauses))) - set(positive),
                              key=lambda j: (-len(word_sets[j] & query_words), j))[:MAX_NEGATIVES]
            for j in positive + negative:
                text = visible_text(tokenizer, clauses[j], hypothesis)
                if text:
                    examples.append((hid, text, any(span in text for span in spans)))
        if i % 100 == 0 or i == len(dataset["documents"]):
            print(f"Prepared evidence pairs for {i}/{len(dataset['documents'])} training contracts.", flush=True)
    return examples


def design_matrix(vectorizer, examples, hypothesis_index):
    text_features = vectorizer.transform([text for _, text, _ in examples]).tocoo()
    # Preserve shared evidence patterns and allow a term to matter differently
    # for each of ContractNLI's fixed 17 hypotheses.
    width = text_features.shape[1]
    offsets = np.array([hypothesis_index[hid] * width for hid, _, _ in examples])
    interactions = csr_matrix(
        (text_features.data, (text_features.row, text_features.col + offsets[text_features.row])),
        shape=(len(examples), width * len(hypothesis_index)),
    )
    return hstack([text_features, interactions], format="csr")


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert data["phase"] == "complete" and data["split"] == "dev"
    rows = data["rows"]
    train = json.loads((ROOT / "backend/data/contractnli/train.json").read_text(encoding="utf-8"))
    train_ids = {doc["id"] for doc in train["documents"]}
    dev_ids = {row["document_id"] for row in rows}
    assert len(train_ids) == 423 and len(dev_ids) == 61 and not (train_ids & dev_ids)
    assert len(rows) == 1037 and len({(r["document_id"], r["hypothesis_id"]) for r in rows}) == 1037
    hypothesis_index = {hid: index for index, hid in enumerate(sorted(train["labels"]))}
    assert len(hypothesis_index) == 17
    assert all(r["hypothesis"] == train["labels"][r["hypothesis_id"]]["hypothesis"] for r in rows)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    assert tokenizer.is_fast
    examples = train_examples(train, tokenizer)
    y_train = [label for _, _, label in examples]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30_000, sublinear_tf=True)
    vectorizer.fit([text for _, text, _ in examples])
    model = LogisticRegression(max_iter=1000)
    model.fit(design_matrix(vectorizer, examples, hypothesis_index), y_train)
    print(f"Trained on {len(examples):,} pairs with {sum(y_train):,} evidence positives.", flush=True)

    targets = []
    candidates = []
    for row in rows:
        row["selected_index"] = select(row)
        row["text_evidence_score"] = None
        row["selected_index"]["text_scorer"] = 0
        idx = row["selected_index"]["rescue_095"]
        if idx:
            clause = row["candidates"][idx]
            text = visible_text(tokenizer, clause["top_clause"], row["hypothesis"])
            targets.append((row["hypothesis_id"], text, clause["evidence_state"] == "visible"))
            candidates.append(row)

    scores = model.predict_proba(design_matrix(vectorizer, targets, hypothesis_index))[:, 1]
    for row, score in zip(candidates, scores):
        row["text_evidence_score"] = float(score)
        if score >= CUTOFF:
            row["selected_index"]["text_scorer"] = row["selected_index"]["rescue_095"]

    summary = {rule: summarize(rows, rule) for rule in rows[0]["selected_index"]}
    for rule in ("top_one", "rescue_095"):
        assert summary[rule]["correct"] == data["summary"]["rules"][rule]["correct"]
    for stats in summary.values():
        assert stats["correct"] == summary["top_one"]["correct"] + stats["fixed"] - stats["new_errors"]
    ref, fitted = summary["corroborated"], summary["text_scorer"]
    decision = {
        "accuracy_improved": fitted["correct"] > ref["correct"],
        "no_more_new_errors": fitted["new_errors"] <= ref["new_errors"],
        "no_fewer_contradictions": fitted["classes"]["Contradiction"]["correct"] >= ref["classes"]["Contradiction"]["correct"],
    }
    labels = [label for _, _, label in targets]
    result = {
        "train_contracts": len(train_ids), "dev_contracts": len(dev_ids), "dev_cases": len(rows),
        "training_pairs": len(examples), "training_positives": sum(y_train), "threshold": CUTOFF,
        "eligible_dev_candidates": len(labels), "eligible_dev_visible": sum(labels),
        "dev_auroc": roc_auc_score(labels, scores), "dev_average_precision": average_precision_score(labels, scores),
        "decision": decision, "summary": summary,
        "rows": [{key: r[key] for key in ("document_id", "hypothesis_id", "selected_index", "text_evidence_score")}
                 for r in rows],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
