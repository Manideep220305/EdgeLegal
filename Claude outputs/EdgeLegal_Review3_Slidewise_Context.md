# EdgeLegal: Review 3, Slide-wise Context

> Historical snapshot. Use [current research progress](../docs/EdgeLegal_Progress.md) for implementation and experiment status.

Purpose of this file: for each of the 11 slides in the Review 3 structure, what must go on the slide, what to say, where the material comes from in the project, and what is still missing. It follows the format of the Review-II deck (red titles, short bullets, no tables) and uses only facts from the project docs, code and results. No names are included.

Status legend: **READY** = can be used as is. **PENDING** = needs a result or screenshot that does not exist yet. **CAUTION** = a claim that must be worded carefully.

---

## 0. One-paragraph project summary (use this to answer "what is your project?")

EdgeLegal checks whether a contract entails, contradicts, or does not mention a given requirement (the ContractNLI task: 607 NDAs, 17 fixed hypotheses, 3 labels). A small fine-tuned NLI model (DeBERTa-v3-base, about 0.2B parameters) answers first. If its confidence is at least a frozen threshold (0.95) the answer is accepted. If not, the case is escalated to an LLM that reads the retrieved clauses. The research question is whether this confidence-gated triage keeps accuracy while cutting LLM calls, and whether the small model's confidence can be trusted as the gate. Three configurations are compared on the same test split: A (no gate, all cases to the LLM), B (off-the-shelf NLI gate), C (fine-tuned gate).

## 0.1 Honest status in five lines (do not contradict these anywhere)

1. Product: the demo works end to end (upload, analyze, history, analytics, research page). Roughly 85% as a demo, about 45% as something trustworthy.
2. Research: about 40%. The full matched A/B/C comparison on the real pipeline is not finished.
3. Dev-set numbers are strong (accuracy 0.970) but use ideal evidence spans. In the live pipeline accuracy is 0.532 on a 20-document smoke run.
4. The gate is badly calibrated live (ECE 0.440). It accepts about 89% of pairs at 0.95 but only about 55% of those are correct.
5. The LLM used so far is Groq-hosted (gpt-oss-20b). Those numbers are interim, not paper-grade. The local-LLM step is deliberately left for last.

---

## Slide 1: Title

**Status: READY** (names intentionally left out)

Put on the slide, in the Review-II layout (centered, red title):
- Department of Computer Science and Engineering
- Academic Year 2026-2027
- B. Tech IV Year / I Semester
- Project Work Phase-1
- REVIEW-III
- Title (red, bold): **EdgeLegal: Privacy-Preserving Contract Analysis & Contradiction Detection**
- Batch No: C7

Add yourself, only if the department requires it: team members with roll numbers (left) and guide with designation (right), as in the Review-II title slide.

Say: one sentence. "A small AI model answers contract questions first, and only the cases it is unsure about go to a large LLM."

**CAUTION:** the title says "privacy-preserving". That is only true once the LLM runs locally. While the interim LLM is on Groq (a cloud service), contract text leaves the machine for escalated cases. Say this openly: privacy is the design goal, the local-LLM step is the final stage.

---

## Slide 2: Project Overview and Objectives

**Status: READY** (carried forward from Review 2, keep brief)

Problem statement (bullets):
- Existing legal AI relies heavily on LLMs, which raises computational cost and processing time.
- Cloud-based processing raises privacy concerns for sensitive contracts.
- Few systems reliably compare related clauses to detect contradictions.

Objectives (bullets):
- Build a confidence-calibrated two-tier framework: a fine-tuned NLI classifier, with a selective LLM for uncertain cases.
- Evaluate semantic retrieval and confidence-based routing on ContractNLI.
- Measure the effect on accuracy, calibration, and unnecessary LLM calls.

Scope line (final scope, say it explicitly):
- In scope: hypothesis-versus-clause classification on ContractNLI (17 fixed hypotheses, 3 labels).
- Future work: within-document and cross-document contradiction detection, and a privacy-aware escalation layer for terms and conditions.

Say, in plain words:
- Entailment = the contract says it. Contradiction = the contract says the opposite. NotMentioned = the contract is silent.
- "Triage" = sorting by difficulty: easy cases are settled cheaply, hard cases are escalated, like a hospital triage desk.

---

## Slide 3: System Architecture and Design

**Status: READY** (diagram to be drawn from the flow below)

Diagram, left to right:
1. Contract + hypothesis in.
2. Clause extraction.
3. Embeddings and vector store (bge-small-en, ChromaDB).
4. Retrieval with an adaptive gap filter (keep rank-1, drop candidates whose distance jumps more than 15% relative to the previous one).
5. Tier-1 classifier: fine-tuned DeBERTa-v3-base gives a label and a confidence (max softmax probability).
6. Confidence gate / router (LangGraph): confidence at least 0.95, accept the Tier-1 label; below 0.95, escalate.
7. Tier-2: LLM reads the retrieved clauses and decides.
8. Output: label, confidence, evidence clauses, which engine answered.

Methodology bullets:
- Same test split and the same frozen threshold (0.95, chosen on dev, never changed) for all configurations.
- Config A: skip the gate, every case goes to the LLM. Config B: gate is the off-the-shelf nli-deberta-v3-base with no ContractNLI training. Config C: gate is our fine-tuned checkpoint.
- Configs differ only in the gate, which is what makes the comparison fair.
- The "agent" is a fixed LangGraph graph (retrieve, triage, confidence check, accept or call LLM). It is not an autonomous agent.

What changed since Review 2 (be ready to explain, a reviewer will notice):
- Tier-1 model: Review 2 said LegalBERT, we use DeBERTa-v3-base because the starting checkpoint is already an NLI model.
- Tier-2 model: Review 2 said Llama 3 8B / Phi-3 Mini through a local runtime. Currently an interim hosted model (gpt-oss-20b on Groq) is used for fast iteration. The local model is the planned final step.
- Vector store (ChromaDB) and the LangGraph routing idea are unchanged.

---

## Slide 4: Module-wise Breakdown

**Status: READY**

Eight modules, one line each (so each can be swapped independently):
1. Training pipeline: fine-tunes DeBERTa-v3-base on ContractNLI and saves the checkpoint (label order Contradiction=0, Entailment=1, NotMentioned=2).
2. Retrieval: bge-small-en embeddings, ChromaDB, adaptive gap filter.
3. Triage classifier: label plus confidence for each pair.
4. Agent graph: `backend/app/agent/graph.py`, `nodes.py`, `state.py` (retrieve, triage, confidence_check, accept, call_llm).
5. LLM provider layer: Groq with multi-key rotation now (selected by environment settings), local model later.
6. FastAPI backend: `backend/app/main.py`, `live_service.py`, `history_db.py` (saved analyses), `logging_config.py` (logs).
7. Experiment runner: `experiment_runner.py` runs configs A, B, C and reports accuracy, macro-F1, ECE, and McNemar; `recalibration_preview.py` is an exploratory calibration analysis.
8. Frontend: React and TypeScript, pages Landing, App, History, Analytics, Research (`frontend/src/pages`).

Say: modules 1 to 6 are built and working. Module 7 has run on small samples; the full matched run is pending.

---

## Slide 5: Data Ingestion and Preprocessing

**Status: READY** (include the known issue)

Dataset overview:
- ContractNLI: document-level NLI for legal contracts, 607 non-disclosure agreements, 17 fixed hypotheses.
- Each contract-hypothesis pair is labeled Entailment, Contradiction, or NotMentioned. Evidence spans are annotated for Entailment and Contradiction.
- Split: 423 train, 61 dev, 123 test contracts. Test = 123 x 17 = 2,091 pairs.

Preprocessing steps:
- Pair every contract with each of the 17 hypotheses.
- Premise: in training the annotated evidence spans are used. At inference, premises are the clauses returned by retrieval.
- Tokenize (hypothesis, premise) for DeBERTa.
- For live uploads: extract text, split into clauses, embed, store in ChromaDB, retrieve per hypothesis.

Known issue (state it, it is a finding, not a failure):
- NotMentioned examples were trained with the full document as the premise, but live the model sees only retrieved clauses. This train/deploy mismatch is a likely cause of the live accuracy drop. The fix is retraining with deployment-realistic premises (planned, not done yet).

---

## Slide 6: Core Algorithm Execution

**Status: READY**

Pseudocode:
```
for (contract, hypothesis) in pairs:
    clauses = retrieve(contract, hypothesis)
    p = softmax(deberta(hypothesis, clauses))
    conf = max(p)
    if conf >= TAU:            # TAU = 0.95, frozen on dev
        label = argmax(p)      # accept
    else:
        label = llm(hypothesis, clauses)
    record(label, conf, escalated)
```

Formulas and metrics to show as bullets:
- Confidence = highest softmax probability over the 3 labels.
- Gate: accept if confidence is at least 0.95, otherwise escalate.
- ECE (expected calibration error): bin predictions by confidence, then average the gap between confidence and actual accuracy per bin, weighted by bin size. Low ECE means "95% confident" really is right about 95% of the time.
- McNemar test (continuity-corrected): compares two configs on the same pairs using only the pairs where exactly one config is right. Statistic = (|b - c| - 1)^2 / (b + c).
- Calibration vs discrimination: calibration asks whether the probabilities are honest. Discrimination (AUROC of confidence for correctness) asks whether high confidence actually separates right from wrong. A gate needs both.

---

## Slide 7: Interface and UI/UX Design

**Status: READY** (screenshots optional, PENDING if wireframes are required)

API endpoints (FastAPI):
- GET /health: service status
- POST /upload: ingest a contract
- POST /analyze: run triage and LLM, returns the report
- GET /history: past analyses
- GET /history/{id}: one saved report
- GET /history/documents: uploaded documents
- GET /analytics: usage statistics
- GET /research/metrics: config A/B/C results (read from saved result files)
- GET /documents/{id}/file: original file

Frontend pages: Landing (what EdgeLegal does), App (upload and analyze), History (saved reports), Analytics (usage and routing), Research (A/B/C comparison).

Say: the Research page reads saved experiment result files only, so it never shows invented numbers; results are tagged paper-grade or not.

To add: one screenshot each of the App page and the Research page.

---

## Slide 8: Code Snippets and Key Functions

**Status: READY**

Show two snippets (shortened, accurate):
1. `agent/nodes.py`, `confidence_check_node`: reads `triage_confidence` and `threshold` from state, returns `accept` if confidence is at least the threshold, else `escalate`.
2. `agent/graph.py`, `build_graph`: edges retrieve to triage to confidence_check, then conditional edge to accept (end) or call_llm (end). Config A uses a `skip_triage` edge from retrieve straight to call_llm.

Other good candidates if the reviewer wants more: the adaptive gap filter in retrieval, the ECE function in the experiment runner, the Groq key-rotation logic in the LLM client.

Say: the whole routing idea is one comparison against a frozen threshold, which keeps it simple to explain and to test.

---

## Slide 9: Module-1 Demonstration

**Status: PENDING** (needs screenshots or a screen recording)

Module 1 to demonstrate: the backend plus retrieval-and-triage path working end to end.

Capture, in this order:
1. Upload page with a contract loaded.
2. Analysis result showing, per hypothesis, the label, the confidence, and which engine answered (Tier-1 or LLM), with the evidence clause.
3. History page showing a saved report.
4. Research page showing the A/B/C results table.
5. Optional: the log file or terminal showing the pipeline running (retrieve, triage, confidence_check, accept or escalate).

Say: this is the working system, not a mockup; every number on the Research page comes from saved result files.

---

## Slide 10: Preliminary Results and Unit Testing

**Status: READY** (numbers below are real; keep the caveats)

Dev set, ideal evidence spans (classifier alone):
- Accuracy 0.9701, macro-F1 0.9391.
- Per-class F1: Entailment 0.9699, NotMentioned 0.9988, Contradiction 0.8485.
- ECE 0.0285.
- Contradiction is the hardest class and the one that matters most for the project.

Live pipeline, 20 documents, 340 pairs (retrieved clauses as premises):
- Accuracy 0.532, macro-F1 0.480, ECE 0.4396.
- 309 of 340 pairs came out at about 0.994 confidence, but only about 55% of them were correct. About 89% of pairs are accepted at 0.95.

Interpretation (the finding):
- With perfect evidence the model is excellent. With realistic retrieved clauses it is over-confident and often wrong, so the 0.95 gate lets wrong answers through.
- Likely cause: train/deploy premise mismatch (slide 5). Planned fix: retrain on realistic premises.

**CAUTION:** the dev numbers are an upper bound. The live numbers are a 20-document smoke run, not the full 123-contract test. Do not present either as the final result.

Testing mention (if asked for unit tests): the experiment runner prints a comparison table with McNemar significance when A, B, C are run together; type-checking of the frontend passes cleanly; backend logging is in place for debugging.

---

## Slide 11: Performance Evaluation

**Status: PARTLY PENDING**

Measured so far (real):
- LLM prompt size: about 1,100 tokens on average (max about 2,700), about 2.5 clauses per call.
- Reliability: about 98% of LLM calls succeeded on the first attempt (key rotation handles rate limits).
- Interim LLM limits on the free hosted tier: roughly 30 requests per minute, 1,000 per day, 8K tokens per minute, 200K tokens per day per organization. The daily token cap is what interrupted the full A/B/C run.
- Exploratory recalibration (fit and checked on the same 20-document run, so optimistic): ECE improves from 0.44 to about 0.04, but AUROC of confidence for correctness is only about 0.61, and reliable thresholds would accept almost nothing. Conclusion: calibration can be patched after training, usefulness for routing cannot. Retraining is needed.

To fill when the matched run finishes (use this layout, bullets or the red-title style, no tables on the slide):
- Config A: accuracy [pending], LLM calls [pending, all pairs].
- Config B: accuracy [pending], LLM calls [pending], escalation rate [pending].
- Config C: accuracy [pending], LLM calls [pending], escalation rate [pending].
- McNemar C vs A, C vs B on the pairs that are error-free in all three: [pending].

Command that produces them (run from the project root with the backend environment):
`python -m backend.app.experiment_runner --configs A B C --limit 20` (use `--limit 10` if the daily token cap is a problem).

**CAUTION:** state that these LLM numbers are interim (hosted model, small sample). Paper-grade numbers need the local LLM, the full 123-document test split, and repeats.

---

## Closing slide content (if a conclusion slide is requested)

- Built: working end-to-end triage system with a demo app and an experiment framework.
- Found: the small model is excellent with ideal evidence but over-confident in the live pipeline; the likely cause is a train/deploy mismatch.
- Next: retrain with realistic premises, run the matched A/B/C comparison on the full test split with a local LLM, then add within-document contradiction detection and the privacy-aware escalation layer.

---

## Likely questions from the reviewer, with short honest answers

- **"Three classes is too simple."** The three labels are the benchmark's standard task and make results comparable with published work. The harder, novel part is the routing: when can a small model be trusted, and how to measure that. Within-document contradiction is the planned extension.
- **"Why not just use the LLM?"** Cost, latency, and privacy. Config A measures exactly what the LLM-only approach costs and achieves.
- **"How do you know the confidence can be trusted?"** We measured it: ECE and AUROC. On the live pipeline it cannot yet, which is the main finding and the reason for retraining.
- **"Is the LLM answer trustworthy?"** Not automatically. It is compared against the gold labels in the same experiment, and we report its accuracy per config.
- **"Is it really privacy-preserving?"** The design is; the current interim run uses a hosted LLM. The local-LLM step makes it fully local.
- **"What is novel?"** The confidence-gated triage evaluated with a matched A/B/C design and paired significance testing on a legal NLI benchmark, plus the calibration analysis. The closest prior work (multi-agent legal contradiction detection) uses synthetic documents and no confidence gating.

---

## Where each fact comes from (to verify before presenting)

- Project definition, metrics, configs, status: `CLAUDE.md` in the repo root (most complete and current).
- Short current-status summary: the Cowork project doc `claude/edgelegal-status.md`.
- Original scope and research design: `docs/EdgeLegal_Project_Guide.md` (no results).
- Research framing and gap analysis: `researchpaper_claude/paper_draft.md` and `literature_review.md`.
- Raw run results: `results/*.json`. Logs: `logs/edgelegal.log`.
- Do not use `docs/EdgeLegal_Implementation_Summary.md`, `technical_architecture.md` or `architecture123.md` for status; they are outdated.
