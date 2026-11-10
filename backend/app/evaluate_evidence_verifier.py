"""Test an evidence verifier on cached top-three predictions with contract-grouped folds."""
from __future__ import annotations

import json
import math
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.evaluate_strict_overrides import SOURCE, THRESHOLD, select, summarize

OUTPUT = SOURCE.with_name("evidence-verifier.json")
VERIFIER_THRESHOLD = 0.5


def features(row, index):
    candidates = row["candidates"]
    first, alternate = candidates[0], candidates[index]
    others = [c for i, c in enumerate(candidates[1:], 1) if i != index]
    hypothesis_words = set(re.findall(r"\b\w+\b", row["hypothesis"].lower()))
    clause_words = set(re.findall(r"\b\w+\b", alternate["top_clause"].lower()))
    distance = alternate["top_distance"]
    first_distance = first["top_distance"]
    return [
        index,
        distance,
        distance - first_distance,
        (distance - first_distance) / max(abs(first_distance), 1e-6),
        first["confidence"],
        alternate["confidence"],
        alternate["predicted"] == "Contradiction",
        math.log1p(alternate["input_tokens"]),
        alternate["truncated"],
        len(hypothesis_words & clause_words) / max(len(hypothesis_words), 1),
        any(c["predicted"] == alternate["predicted"] and c["confidence"] >= THRESHOLD for c in others),
    ]


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert data["phase"] == "complete" and data["split"] == "dev" and data["threshold"] == THRESHOLD
    rows = data["rows"]
    assert len(rows) == 1037 and len({r["document_id"] for r in rows}) == 61
    examples = []
    for row in rows:
        row["selected_index"] = select(row)
        row["verifier_score"] = None
        if row["candidates"][0]["predicted"] != "NotMentioned":
            continue
        for index, alternate in enumerate(row["candidates"][1:], 1):
            if alternate["predicted"] != "NotMentioned" and alternate["confidence"] >= THRESHOLD:
                supported = alternate["evidence_state"] == "visible" and alternate["predicted"] == row["gold"]
                examples.append((row, index, supported))

    x = np.array([features(row, index) for row, index, _ in examples])
    y = np.array([supported for _, _, supported in examples], dtype=int)
    documents = np.array([row["document_id"] for row, _, _ in examples])
    scores = np.full(len(examples), np.nan)
    folds = []
    for train, validation in GroupKFold(n_splits=5).split(x, y, documents):
        train_docs, validation_docs = set(documents[train]), set(documents[validation])
        assert not train_docs & validation_docs and set(y[train]) == {0, 1}
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(x[train], y[train])
        scores[validation] = model.predict_proba(x[validation])[:, 1]
        folds.append({"train_documents": sorted(map(int, train_docs)),
                      "validation_documents": sorted(map(int, validation_docs)),
                      "train_examples": len(train), "validation_examples": len(validation),
                      "validation_positives": int(y[validation].sum())})
    assert np.isfinite(scores).all()

    for (row, index, _), score in zip(examples, scores):
        if index == row["selected_index"]["rescue_095"]:
            row["verifier_score"] = float(score)

    for row in rows:
        rescue = row["selected_index"]["rescue_095"]
        score = row["verifier_score"]
        assert bool(rescue) == (score is not None)
        row["selected_index"]["verifier"] = rescue if score is not None and score >= VERIFIER_THRESHOLD else 0

    summary = {rule: summarize(rows, rule) for rule in rows[0]["selected_index"]}
    for result in summary.values():
        assert result["correct"] == summary["top_one"]["correct"] + result["fixed"] - result["new_errors"]
    for rule in ("top_one", "rescue_095"):
        assert summary[rule]["correct"] == data["summary"]["rules"][rule]["correct"]
    ref, verifier = summary["corroborated"], summary["verifier"]
    decision = {
        "accuracy_improved": verifier["correct"] > ref["correct"],
        "no_more_new_errors": verifier["new_errors"] <= ref["new_errors"],
        "no_fewer_contradictions": verifier["classes"]["Contradiction"]["correct"] >= ref["classes"]["Contradiction"]["correct"],
    }
    payload = {
        "split": "dev", "source": SOURCE.name, "cases": len(rows),
        "verifier_threshold": VERIFIER_THRESHOLD, "training_examples": len(examples),
        "positives": int(y.sum()), "auroc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "folds": folds, "summary": summary, "decision": decision,
        "rows": [{key: row[key] for key in ("document_id", "hypothesis_id", "selected_index", "verifier_score")}
                 for row in rows],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("rows", "folds")}, indent=2))


if __name__ == "__main__":
    main()
