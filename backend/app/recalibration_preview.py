"""
recalibration_preview.py
─────────────────────────
EXPLORATORY analysis only (not paper numbers): can a cheap post-hoc
recalibration (Platt scaling / isotonic regression on the classifier's
confidence) repair the deployment-time calibration failure, without
retraining?

Uses the triage_label / triage_confidence / gold columns already saved in a
results/*.json run. Fits on some documents and evaluates on held-out
documents (grouped k-fold by doc_id, so no contract is in both sides).

Usage (from EdgeLegal_CodeBase/):
    python -m backend.app.recalibration_preview results/C_finetuned_triage_XXXX.json

Pure Python, no torch needed. LIMITS: the file it's pointed at is a small
TEST-split sample; a proper version must fit on dev-split in-pipeline pairs
and evaluate on test, with the threshold frozen before test. This preview
only tells us whether the idea is worth doing properly.
"""
from __future__ import annotations
import json, math, random, sys
from collections import defaultdict

def ece(conf, ok, bins=10):
    n = len(conf); tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(ok[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            tot += len(idx) / n * abs(acc - avg)
    return tot

def brier(p, ok):
    return sum((pi - yi) ** 2 for pi, yi in zip(p, ok)) / len(p)

def auroc(score, ok):
    pos = [s for s, y in zip(score, ok) if y]; neg = [s for s, y in zip(score, ok) if not y]
    if not pos or not neg: return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))

def logit(c):
    c = min(max(c, 1e-6), 1 - 1e-6); return math.log(c / (1 - c))

def fit_platt(conf, ok, iters=200, lr=0.1):
    a, b = 1.0, 0.0
    xs = [logit(c) for c in conf]
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ok):
            p = 1 / (1 + math.exp(-(a * x + b))); ga += (p - y) * x; gb += (p - y)
        a -= lr * ga / len(xs); b -= lr * gb / len(xs)
    return lambda c: 1 / (1 + math.exp(-(a * logit(c) + b)))

def fit_isotonic(conf, ok):
    pts = sorted(zip(conf, ok)); blocks = [[p[1], 1, p[0], p[0]] for p in pts]  # sum, n, lo, hi
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            s, n, lo, _ = blocks[i]; s2, n2, _, hi2 = blocks[i + 1]
            blocks[i:i + 2] = [[s + s2, n + n2, lo, hi2]]; i = max(i - 1, 0)
        else: i += 1
    def f(c):
        for s, n, lo, hi in blocks:
            if c <= hi: return s / n
        return blocks[-1][0] / blocks[-1][1]
    return f

def main(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = [r for r in d["rows"] if r.get("triage_label") is not None
            and r.get("triage_confidence") is not None and not r.get("error")]
    conf = [r["triage_confidence"] for r in rows]
    ok = [int(r["triage_label"] == r["gold"]) for r in rows]
    docs = sorted({str(r["doc_id"]) for r in rows}); random.Random(0).shuffle(docs)
    fold = {doc: i % 5 for i, doc in enumerate(docs)}
    oof = {"platt": [None] * len(rows), "isotonic": [None] * len(rows)}
    for k in range(5):
        tr = [i for i, r in enumerate(rows) if fold[str(r["doc_id"])] != k]
        te = [i for i, r in enumerate(rows) if fold[str(r["doc_id"])] == k]
        fp = fit_platt([conf[i] for i in tr], [ok[i] for i in tr])
        fi = fit_isotonic([conf[i] for i in tr], [ok[i] for i in tr])
        for i in te: oof["platt"][i] = fp(conf[i]); oof["isotonic"][i] = fi(conf[i])
    print(f"file: {path}\npairs used: {len(rows)}  docs: {len(docs)}  triage accuracy: {sum(ok)/len(ok):.3f}")
    print(f"\nDoes confidence carry ANY signal about correctness?  AUROC = {auroc(conf, ok):.3f}  (0.5 = none, 1.0 = perfect)")
    print(f"\n{'method':<22}{'ECE':>8}{'Brier':>8}")
    print(f"{'raw confidence':<22}{ece(conf, ok):>8.3f}{brier(conf, ok):>8.3f}")
    for name in ("platt", "isotonic"):
        print(f"{name + ' (held-out)':<22}{ece(oof[name], ok):>8.3f}{brier(oof[name], ok):>8.3f}")
    print(f"\nRouting view (accept if confidence >= tau; rest -> LLM):")
    print(f"{'score':<18}{'tau':>6}{'accepted':>10}{'acc when accepted':>19}")
    for name, sc in (("raw", conf), ("platt", oof["platt"]), ("isotonic", oof["isotonic"])):
        for tau in (0.5, 0.6, 0.7, 0.8, 0.95):
            acc = [i for i, s in enumerate(sc) if s >= tau]
            if acc: print(f"{name:<18}{tau:>6}{len(acc)/len(sc):>10.0%}{sum(ok[i] for i in acc)/len(acc):>19.3f}")
            else:   print(f"{name:<18}{tau:>6}{0:>10.0%}{'-':>19}")

if __name__ == "__main__":
    main(sys.argv[1])
