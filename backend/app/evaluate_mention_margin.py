"""Evaluate a train-selected Not Mentioned margin on the saved clause scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("Contradiction", "Entailment", "NotMentioned")


def arrays(rows):
    return (np.asarray([row["features"] for row in rows]),
            np.asarray([row["target"] for row in rows]),
            np.asarray([row["document_id"] for row in rows]))


def fit(features, gold):
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=2000)
    ).fit(features, gold)


def verdicts(probabilities, threshold):
    baseline = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    margin = probabilities[:, :2].max(axis=1) - probabilities[:, 2]
    return np.where((baseline != LABELS[2]) & (margin < threshold), LABELS[2], baseline)


def summarize(gold, predicted):
    return {
        "correct": int((gold == predicted).sum()),
        "cases": len(gold),
        "macro_f1": f1_score(gold, predicted, labels=LABELS, average="macro"),
        "correct_by_class": {
            label: int(((gold == label) & (predicted == label)).sum()) for label in LABELS
        },
        "false_mentions": int(((gold == LABELS[2]) & (predicted != LABELS[2])).sum()),
    }


def select_threshold(gold, probabilities):
    control = summarize(gold, verdicts(probabilities, 0.0))
    best_threshold, best = 0.0, control
    # Predefined 0.001 grid. Start with the control and keep the smaller threshold on ties.
    for step in range(1, 1001):
        threshold = step / 1000
        predicted = verdicts(probabilities, threshold)
        correct = gold == predicted
        if correct.sum() < control["correct"]:
            continue
        if (correct & (gold == LABELS[0])).sum() < control["correct_by_class"][LABELS[0]]:
            continue
        if (correct & (gold == LABELS[2])).sum() > best["correct_by_class"][LABELS[2]]:
            best_threshold, best = threshold, summarize(gold, predicted)
    return best_threshold, control, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "clause-combiner.zip")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        def read(name):
            return json.loads(archive.read(f"clause-combiner/{name}.json"))

        train, dev, saved = (read(name) for name in
                             ("train_features", "dev_features", "dev_predictions"))
    if len(train["rows"]) != 7191 or len(dev["rows"]) != 1037:
        raise ValueError("Unexpected feature counts.")
    if train["feature_columns"] != dev["feature_columns"] or len(train["feature_columns"]) != 30:
        raise ValueError("Unexpected feature columns.")
    x_train, y_train, groups = arrays(train["rows"])
    x_dev, y_dev, dev_groups = arrays(dev["rows"])
    if len(set(groups)) != 423 or len(set(dev_groups)) != 61 or set(groups) & set(dev_groups):
        raise ValueError("Unexpected contract split.")
    probabilities = np.empty((len(y_train), len(LABELS)))
    folds = []
    for train_idx, val_idx in GroupKFold(n_splits=5).split(x_train, y_train, groups):
        model = fit(x_train[train_idx], y_train[train_idx])
        assert tuple(model.classes_) == LABELS
        probabilities[val_idx] = model.predict_proba(x_train[val_idx])
        folds.append(sorted(map(int, set(groups[val_idx]))))
    threshold, train_base, train_adjusted = select_threshold(y_train, probabilities)

    model = fit(x_train, y_train)
    assert tuple(model.classes_) == LABELS
    dev_probabilities = model.predict_proba(x_dev)
    base, adjusted = (verdicts(dev_probabilities, cutoff) for cutoff in (0, threshold))
    saved_by_key = {(r["document_id"], r["hypothesis_id"]): r for r in saved}
    rows = []
    for i, row in enumerate(dev["rows"]):
        key = row["document_id"], row["hypothesis_id"]
        expected = saved_by_key[key]
        if expected["gold"] != y_dev[i] or expected["predictions"]["top_five_logistic"] != base[i]:
            raise ValueError(f"Saved baseline prediction differs for {key}.")
        rows.append({"document_id": key[0], "hypothesis_id": key[1], "gold": row["target"],
                     "baseline": base[i], "adjusted": adjusted[i],
                     "probabilities": dict(zip(LABELS, dev_probabilities[i].tolist()))})
    dev_base, dev_adjusted = summarize(y_dev, base), summarize(y_dev, adjusted)
    assert dev_base["correct"] == 631
    classes = dev_adjusted["correct_by_class"]
    with args.archive.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "archive_sha256": archive_hash, "scikit_learn": sklearn.__version__, "threshold": threshold,
        "training_folds": folds, "training_oof": {"baseline": train_base, "adjusted": train_adjusted},
        "development": {"baseline": dev_base, "adjusted": dev_adjusted},
        "carry_forward": bool(threshold and dev_adjusted["correct"] >= 631
                              and classes[LABELS[0]] >= 40 and classes[LABELS[2]] >= 220),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in ("training_folds", "rows")}, indent=2))


if __name__ == "__main__":
    main()
