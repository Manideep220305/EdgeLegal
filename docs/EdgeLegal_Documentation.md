# EdgeLegal: Full Project Documentation

**Current research status.** See [EdgeLegal Research Progress](EdgeLegal_Progress.md)
for the latest decisions and links to experiment artifacts.

Version: 2026-10-01. Status: living document. This is the single place that explains what EdgeLegal is, what it is meant to prove, what is actually built, what the evidence says so far, and what is still open. It is written to be honest first. Where something is unproven, broken, or only partly done, it says so.

Companion files (each has a different job, see section 15):
`CLAUDE.md` (working notes for AI sessions), `docs/EdgeLegal_Project_Guide.md` (original scope and design), `researchpaper_claude/` (literature review and paper draft).

---

## 1. Overview in plain English

A contract is long. Someone asks a question about it, for example: "Does this agreement stop the receiving party from sharing confidential information with employees?" The answer is one of three:

- **Entailment:** the contract says it.
- **Contradiction:** the contract says the opposite.
- **NotMentioned:** the contract is silent.

A large language model (LLM) can answer this, but it is slow, costly, and, if hosted in the cloud, requires sending private contracts to someone else's server. A small trained classifier is fast and cheap but cannot explain itself and may be wrong without knowing it.

EdgeLegal combines them as a **triage system**. The small classifier answers first and reports how confident it is. If it is confident enough (at least 0.95) its answer is accepted. If not, the case is escalated to an LLM that reads the relevant clauses. The research question is whether this keeps accuracy while cutting LLM calls, and whether the small model's confidence is trustworthy enough to be the gate.

It is a research project (aimed at one conference paper and later a journal paper), and it also has a working demo web app.

---

## 2. Problem and motivation

1. Legal AI built only on LLMs pays for a full LLM call on every clause comparison, including trivial ones.
2. Cloud LLMs mean sensitive contracts leave the user's machine. A private, local deployment has limited compute, so every LLM call matters.
3. Small NLI classifiers are cheap, but their confidence is not automatically trustworthy. Routing on an uncalibrated confidence score is meaningless.
4. Prior work usually evaluates classifiers alone, or LLM pipelines without a cheap gate. Few check whether the gate's confidence can be trusted.

---

## 3. Scope

### In scope (the conference-paper scope, frozen)

- Task: hypothesis-versus-clause classification on **ContractNLI** (607 real NDAs, 17 fixed hypotheses, 3 labels).
- One triage model family (a Transformer NLI classifier), one dataset, one honest three-way comparison (configs A, B, C).
- Metrics: accuracy, precision, recall, F1 per class, calibration (ECE, reliability), LLM-call reduction, latency, McNemar significance.
- A working demo application that runs the same pipeline as the experiments.

### Explicitly out of scope for the first paper

- New model architectures or training algorithms.
- Graph neural networks, reinforcement-learned thresholds, federated or active learning.
- Other languages, other contract types.
- Cross-contract reasoning (kept as future work, section 13).

### Not a claim

The classifier does not "solve legal reasoning". The LLM remains responsible for reasoning and explanation. The tool is decision support, not a substitute for a qualified lawyer.

---

## 4. Research question and contribution

**Research question.** Can a fine-tuned legal NLI classifier act as a confidence-gated triage layer in front of an LLM, accepting its own answers when confident and escalating only uncertain cases, while preserving accuracy, reducing LLM calls, and keeping confidence calibrated enough to trust for routing?

**Contribution we can honestly claim (target):** a matched, standard-benchmark evaluation of confidence-gated triage for legal contradiction detection that explicitly checks whether the routing confidence is calibrated, comparing three configurations on the same test split with paired significance testing.

**What we cannot claim:** that confidence-gated escalation itself is new. It is not (section 11).

**Where the project actually stands against that contribution (important):** the calibration check has been done and the result is negative under deployment conditions (section 10). That is a legitimate, reportable finding, but it means the project currently demonstrates "this gate is not yet safe" rather than "this gate works". The next step is the development evidence diagnostic in section 13. Its results will inform whether to change retrieval or retrain.

---

## 5. System architecture

```
Contract (PDF/text)
   -> Clause extraction (structure-aware parser, keeps page numbers)
   -> Embeddings (BAAI/bge-small-en) -> ChromaDB (vector store)
   -> Retrieval for the hypothesis, with an adaptive gap filter
        (keep rank-1, drop candidates after a >15% relative distance jump)
   -> Tier-1 classifier: fine-tuned DeBERTa-v3-base
        -> label (Entailment / Contradiction / NotMentioned) + confidence
   -> Router (LangGraph): confidence >= 0.95 ?
        YES -> accept the Tier-1 label (no LLM call)
        NO  -> Tier-2: LLM reads retrieved clauses, returns label + reasoning + cited clauses
   -> Final result: label, confidence, which engine answered, evidence clauses, explanation
```

Component responsibilities:

| Component | Does | Does not |
|---|---|---|
| Clause parser | Splits a contract into independently retrievable clauses with page numbers | Predict anything |
| Embedder + ChromaDB | Embeds clauses, retrieves the most similar ones | Classify |
| Triage classifier | Fast label plus confidence | Explain, reason generally |
| LangGraph graph | Applies the frozen threshold, chooses accept or escalate | Predict or explain |
| LLM | Reasons, explains, can override on escalated cases | Handle every case |
| Report layer | Packages label, confidence, engine, evidence | Any prediction |

Important honesty note: the "agent" is a **fixed graph** (retrieve, triage, confidence check, accept or call LLM). It is not an autonomous agent that plans or chooses tools.

### The three configurations

- **A, no triage:** retrieval goes straight to the LLM. Triage never runs and no classifier hint goes into the prompt (an earlier version leaked the classifier's guess into A's prompt, which was fixed with a `skip_triage` edge).
- **B, off-the-shelf gate:** the untouched `cross-encoder/nli-deberta-v3-base` (generic NLI, no ContractNLI fine-tuning) gates the LLM. This deviates from the Project Guide, which said "a pretrained ContractNLI checkpoint (e.g. Span-NLI-BERT)". It is defensible (LegalWiz also uses an off-the-shelf NLI model) but must be stated openly in the write-up.
- **C, our gate:** the same architecture fine-tuned on ContractNLI train. B versus C isolates the effect of domain fine-tuning.

Same test split, same frozen threshold, same LLM across all three.

---

## 6. Features

### Built and working

- **Upload and analyze:** upload a contract PDF, check it against hypotheses, get a verdict per hypothesis.
- **Real triage in the live app:** the demo runs the same graph as the experiments (clause parser, bge-small, gap-filter retrieval, triage classifier, frozen threshold 0.95, LLM on escalation).
- **Decision trail:** each result shows which engine answered, the triage confidence, the threshold, and accept versus escalate.
- **Citations:** each verdict cites the clause it used; clicking jumps to the right PDF page (no highlight box yet, because the parser stores page numbers but not coordinates).
- **History:** every analysis is saved (SQLite) and can be filtered, searched and reopened, with the original file.
- **Analytics dashboard:** documents analyzed, contradiction rate, average classifier confidence, classifier-versus-LLM split, most contradicted hypotheses.
- **Research page:** shows dev-set metrics, reliability diagram and threshold sweep from saved result files, and the A/B/C comparison from `results/*.json`, with banners when runs are not paper-grade or not on the same split. It shows no invented numbers.
- **Experiment runner:** runs A, B, C, records accuracy, macro-F1, ECE, latency and McNemar significance, self-flags non-paper-grade runs, and records failed pairs as explicit errors rather than silently counting them wrong.
- **Logging:** every pipeline stage and every failure is written with context to `logs/edgelegal.log` (rotating), and unhandled server errors are traced.
- **LLM provider switch:** Groq (multi-key rotation) for fast demo and interim runs, local Ollama for paper-grade runs, controlled by `LLM_PROVIDER`. Groq runs are flagged as not paper-grade.

### Designed but not built

- Within-document and cross-document contradiction detection (section 13).
- A privacy-aware escalation layer for terms-and-conditions style documents (idea under discussion only, section 13).
- Bounding-box highlights for citations.
- Structured legal-risk summary and multi-turn Q&A (promised in the Review-II deck, not built).

---

## 7. Implementation status

| Area | Status | Honest comment |
|---|---|---|
| Data setup (ContractNLI splits) | Done | 423 / 61 / 123 contracts |
| Clause parsing, embeddings, ChromaDB retrieval | Built | The oversized-clause fallback was tested and remains opt-in |
| Triage classifier training | Done | Dev accuracy 0.970, but see calibration failure live |
| Calibration on dev | Done | ECE 0.0285 on dev premises (flattering) |
| Calibration in the real pipeline | Done, **failed** | ECE 0.440, accuracy 0.532 on 20 documents |
| LangGraph routing | Done | Fixed graph, working |
| LLM tier | Working on Groq | Local-LLM switch deliberately left as the final step |
| Live app unified with research pipeline | Done | Same graph, same threshold |
| Frontend (5 pages) | Done | Built and visually checked; no checked-in automated test suite |
| Experiment runner with McNemar | Built | Full matched run not completed |
| Matched A/B/C run | Development pilot complete | 51 cases across three development contracts. Full test comparison pending |
| Paper-grade results | **None exist** | Needs local LLM, full 123-document test, repeats |
| Retraining with realistic premises | Not started | Deferred for time |
| Multi-layer contradiction | Not started | Journal-paper scope |

Rough progress estimates (judgement, not measured): product about **85% as a demo** and about **45% as something trustworthy**; research about **40%**. The missing 60% of research is the part that produces the paper's claims.

---

## 8. Data and model

### Dataset: ContractNLI

- 607 non-disclosure agreements, 17 fixed hypotheses, three labels. Evidence spans are annotated for Entailment and Contradiction.
- Splits: 423 train, 61 dev, 123 test contracts. Test = 123 x 17 = 2,091 pairs.
- Class balance: Contradiction is the minority (about 9%) and the class the project is about.

### Triage classifier

- Backbone: `cross-encoder/nli-deberta-v3-base` (about 0.2B parameters), warm-started from its native NLI head. The Project Guide said DeBERTa-v3-small or LegalBERT-base; the base model was chosen instead.
- Training: trained on a Kaggle T4 GPU, 3 epochs, batch 32, learning rate 2e-5, class-weighted cross-entropy, early stopping on macro-F1. The checkpoint is about 738 MB.
- **Label order is Contradiction = 0, Entailment = 1, NotMentioned = 2** (the checkpoint's native order, not alphabetical). An earlier inference file used the wrong order and would have silently swapped Entailment and Contradiction; it was fixed.
- Confidence = maximum softmax probability.
- Threshold: tau = 0.95, selected on the dev split only and frozen (dev coverage 98.4%, macro-F1 0.951 among kept cases). It was never tuned on test.
- Training input: Entailment and Contradiction use the annotated evidence spans as the premise. NotMentioned has no spans, so the full (truncated) document is used instead. This is the suspected source of the live failure (section 10).

### LLM tier

- Interim: `openai/gpt-oss-20b` on Groq, a reasoning model, rotated across multiple API keys. Free-tier limits (about 30 requests per minute, 1,000 per day, 8K tokens per minute, 200K tokens per day, shared per organization) are what repeatedly interrupt long runs. Prompts average about 1,100 tokens with about 2.5 clauses per call.
- Planned for paper numbers: a local model through Ollama (the original plan named Llama 3.1 8B or Phi-3 Mini; the laptops can realistically run 3 to 4B models).
- Rule: Groq numbers are never reported as paper results, because Groq is cloud-fast and cannot show the cost the triage layer is meant to cut, and it breaks the privacy claim.

---

## 9. Experimental design

1. Fine-tune the classifier, freeze the threshold on dev.
2. Run A, B, C on the same test split, same threshold, same LLM, same hardware.
3. Report accuracy and per-class P/R/F1, share of cases resolved without the LLM, latency (mean and spread over repeats), ECE and reliability, and McNemar tests between configs.
4. Report McNemar two ways when pairs error out: on pairs clean in all three configs (primary) and on raw files.
5. Never tune the threshold after seeing test results. Report negative findings. State limitations.

Definition of done (from the Project Guide): classifier trained and calibration checked; A, B, C all run on the same test split; threshold frozen from dev; honest write-up with variance and significance; limitations stated.

---

## 10. Results so far (all with caveats)

### 10.1 Dev set (classifier alone, ideal evidence spans)

- Accuracy 0.9701, macro-F1 0.9391 (macro P/R 0.9305 / 0.9486).
- F1 per class: Entailment 0.9699, NotMentioned 0.9988, **Contradiction 0.8485**.
- ECE 0.0285.
- Caveat: dev premises are gold evidence spans (and, for NotMentioned, a distinctive full-document premise). This is an upper bound, not what the live system sees.

### 10.2 In the real pipeline (20 test documents, 340 pairs, fake LLM so only the classifier runs)

| Metric | Dev (gold spans) | In-pipeline (retrieved clauses) |
|---|---|---|
| Accuracy | 0.9701 | 0.5324 |
| Macro-F1 | 0.9391 | 0.4796 |
| ECE | 0.0285 | 0.4396 |
| Coverage at tau 0.95 | 98.4% | 89.1% |

309 of 340 pairs came out at about 0.994 confidence, and only about 55% of those were correct. So at the exact confidence used to justify skipping the LLM, the classifier is close to a coin flip. All three classes collapsed, not only NotMentioned.

### 10.3 Retrieval inclusion (no LLM, no classifier, full 123-document test split)

- The gold evidence clause is inside the gap-filtered retrieved set only **73.1%** of the time (Entailment 72.7%, Contradiction 74.5%) over 1,188 cases.
- The denominator excludes evidence that the parser could not map. Inclusion anywhere in the filtered set does not measure top-one retrieval, evidence visible after truncation, or a classifier accuracy ceiling.

### 10.4 Exploratory recalibration (20-document sample, grouped 5-fold by document)

- AUROC of confidence for correctness: 0.61 (0.5 means no signal).
- Held-out ECE: raw 0.440, Platt scaling 0.040, isotonic 0.126.
- But after Platt no pair scores above 0.7, so a reliable threshold would accept about 0% of cases. Isotonic accepts about 12% at only 0.67 accuracy.
- Interpretation: calibration can be repaired after the fact (the numbers become honest), but usefulness for routing cannot (the confidence does not separate right from wrong). Calibration and discrimination are separate failures.
- Caveat: small test-split sample used for both fitting (in folds) and checking. A proper version fits on dev in-pipeline pairs and evaluates on test.

### 10.5 Experiment runs on file

The September two-contract runs are matched. The separate 20-contract
classifier diagnostic is exploratory. The later matched 20-contract attempt
stopped at the quota cap. The November matched development pilot and the
paired parser evaluation are linked from [current research progress](EdgeLegal_Progress.md).
The full held-out comparison remains pending.

---

## 11. Research gap and literature

### 11.1 What was searched

On 2026-09-25 a literature search covered arXiv, ACL Anthology, IEEE Xplore, ACM Digital Library, SpringerLink and Semantic Scholar, producing a gap analysis (`researchpaper_claude/literature_review.md`), per-paper notes (`notes/00` to `notes/09`), and a search log. A follow-up on 2026-09-29 revisited an inaccessible paper. A correction was made on 2026-10-01 (see below).

### 11.2 What is NOT new

The mechanism (a small model's confidence decides whether to answer or escalate) exists in general ML (FrugalGPT-style cascades, confidence-based early exit, Adaptive-RAG) and, specifically for legal contradiction detection, in **LegalWiz** (arXiv 2510.03418, Oct 2025). LegalWiz stages semantic similarity, then NLI (an off-the-shelf BART-MNLI model), then an LLM (GPT-4o), and escalates pairs whose NLI confidence is at or below 0.7, with confidence-derived hybrid weights. It must be cited and differentiated explicitly.

Correction logged on 2026-10-01: an earlier note listed "ContraGen" as a separate new paper. That was an error. ContraGen is the v1 title of the same arXiv paper that v2 renamed LegalWiz. It is one paper.

### 11.3 Research question and prior work

The study asks how retrieved evidence affects classifier confidence and the
accuracy versus LLM-call trade-off. The [literature review](../researchpaper_claude/literature_review.md)
records the overlap and open novelty checks.

### 11.4 What we actually did about the gap, in order

1. Searched the literature and wrote the gap analysis and per-paper notes.
2. Identified LegalWiz as the closest competitor and wrote the differentiation points (dataset, purpose, calibration, comparison design).
3. Fixed the experiment design so the baseline is clean (Config A skips triage entirely).
4. Trained and verified the classifier, reproducing the teammate's Kaggle numbers exactly on dev.
5. **Actually ran the calibration check the gap is about**, first on dev (looks excellent, ECE 0.0285) and then in deployment conditions (fails, ECE 0.440). This is the most research-relevant thing done so far.
6. Identified two candidates for diagnosis: the prepared versus retrieved input difference and incomplete evidence retrieval.
7. Ran an exploratory recalibration showing calibration is fixable post hoc but routing usefulness is not.
8. Drafted the paper skeleton (`researchpaper_claude/paper_draft.md`): sections 5.1 and 5.2 filled, 5.3 and 5.4 pending the matched run.
9. Scoped a multi-layer extension for the journal paper (section 13).

### 11.5 Brutally honest read on the gap

- **Novelty is modest and narrow.** The idea is not new; the claim rests on the calibration check plus the standard benchmark plus the three-way design. A reviewer who knows LegalWiz will see this as an incremental, evaluation-focused paper.
- **The gap claim is only as strong as the evidence.** Right now the calibration evidence says our own gate fails. That is publishable only if framed as "we checked, it fails under realistic retrieval, here is why, here is what fixes it", and only if the fix is actually carried out and re-measured.
- **One relevant paper is unread.** "Multi-Agent Orchestration of Local LLMs for Contract Review under Resource Constraints" (ACM ICBAR 2025, DOI 10.1145/3800227.3800266) is the closest title match to EdgeLegal's premise. ACM blocks automated access and its contents are unverified. If it already covers this, the gap shrinks. Get access through the college library before submitting.
- **The field is active.** Several relevant papers are under 12 months old, so the window may close.
- **Within-document and cross-document contradiction is not a gap you can claim yet.** It has not been built or tested here. Prior art exists for each layer (section 13); the unification for contracts is the proposed contribution, not a result.

---

## 12. Known problems and risks (be brutally honest in the viva)

1. **The held-out routing claim is still untested.** The matched development pilot and classifier diagnostics show that the current gate accepts confidently wrong answers.
2. **The classifier collapses in deployment** (accuracy 0.532, ECE 0.440), likely because training used gold evidence spans and, for NotMentioned, a full-document premise, while deployment feeds one short retrieved clause.
3. **Top-one evidence can be missing or truncated.** The next diagnostic will distinguish those cases from errors made with visible evidence.
4. **Dev numbers are inflated** by oracle premises. Do not quote 97% as system accuracy.
5. **No final held-out routing results exist.** The latest local classifier comparison covers the full development split. The matched Groq pilot covers three development contracts.
6. **The privacy claim is not currently true.** The interim LLM is hosted. The title's "privacy-preserving" only holds once escalation runs on a local model, and a 3 to 4B local model may be weaker than the 20B hosted one, which would change the accuracy of every escalated case.
7. **Deviations from the Review-II plan** (LegalBERT, Llama 3 8B or Phi-3, report generation, multi-turn Q&A) must be explained, not hidden.
8. **Single dataset, NDAs only.** Generalization to other contract types is untested. ContractNLI's size limits generalization.
9. **Process risks:** local edits are present; some tracked binary files are noisy; several one-off verification scripts are not in a test suite; real API keys have been shared in chat and should be rotated after experiments; a 101 MB leftover file (`PARTIAL_DELETE_ME.pt`) still needs manual deletion.
10. **Understanding risk.** The code and documentation were produced with heavy AI assistance. Every team member needs to be able to explain triage, confidence, calibration, the three configs, and the live-versus-dev gap without notes. The Review 3 slide-context file and section 1 and section 10 of this document are the study base.

---

## 13. Roadmap and future work

### 13.1 Current research sequence

The oversized-clause experiment is complete and remains opt-in. The
[research handoff](EdgeLegal_Progress.md) records the result and the proposed
top-one versus top-three development diagnostic. It also records the pause on
full A/B/C and the conditions for a final held-out comparison.

### 13.2 Future work: multi-layer contradiction detection (journal-paper scope)

Idea: extend beyond hypothesis-versus-clause to also catch (1) clause-versus-clause contradictions within one document, and (2) clause-versus-clause contradictions across documents (for example a base agreement versus an amendment, or a new contract versus a template), unified by a LangGraph agent that runs whichever checks apply and merges the results, all behind the same calibrated triage gate.

Prior art per layer:
- Hypothesis versus clause: ContractNLI, LegalWiz.
- Within document: **Better Call CLAUSE** (EACL 2026 Findings, arXiv 2511.00340), about 23,955 perturbed examples built from CUAD and ContractNLI, directly reusable (use its in-text subset).
- Across documents: Clash-of-Leges (statute and constitutional-law conflicts, not contracts). **No contract cross-document contradiction dataset exists.** A cheap proxy: ContractNLI applies the same 17 hypotheses to 607 contracts, so two contracts with opposite labels for the same hypothesis form a free "disagreement" pair. Caveat: this captures two unrelated contracts disagreeing, not a contract contradicting its own amendment. A small hand-built set (for example real MSA plus amendment families from SEC EDGAR filings) would strengthen it.

Honest status: nothing of this is built. It reuses the same classifier with different input pairs, so it depends on fixing the calibration problem first. Do not present it as done.

### 13.3 Idea under discussion: privacy-aware escalation for terms and conditions

Motivation: people want to ask questions about terms and conditions or contracts they are signing but cannot paste them into a cloud chatbot for privacy reasons. Directions discussed: keep the whole pipeline local, send only minimal redacted evidence if anything leaves the device, and reuse existing models from Hugging Face where suitable. This is not designed, built, or scoped, and no related-work search was done specifically for it. Treat it as a possible journal-paper angle, not a commitment.

### 13.4 Housekeeping

Commit the work in clean steps, clean tracked binary files, delete the leftover partial checkpoint, rotate exposed API keys, and decide whether the dead files `document_processor.py` and `rag_engine.py` should be removed.

---

## 14. Repository map and how to run

```
EdgeLegal_CodeBase/
  backend/
    app/
      main.py, agent/ (graph, nodes, state, live_service),
      parsing/ (clause parser), embeddings/, vectorstore/ (ChromaDB),
      triage_model/ (model, dataset, train, evaluate, infer, verify_checkpoint),
      llm/ (ollama and groq clients, prompts),
      history_db.py, research.py, logging_config.py,
      experiment_runner.py, recalibration_preview.py,
      retrieval_accuracy_check.py, test_pipeline.py
    data/contractnli/ (train, dev, test)
  frontend/src/pages/ (Landing, App, History, Analytics, Research)
  results/        experiment result files (one per run)
  logs/           edgelegal.log
  docs/           guides and architecture notes
  researchpaper_claude/  literature review, notes, paper draft
  kaggle_training/, pdfs/, scripts/   archival and demo-data utilities
```

Two import roots exist and must not be mixed:
- Research and experiment code is run from the **repository root**: `python -m backend.app.experiment_runner --configs A B C --limit 20`, `python -m backend.app.retrieval_accuracy_check`, `python -m backend.app.recalibration_preview results/<file>.json`.
- The live API is run from `backend/`: `uvicorn app.main:app --reload`. If started without reload, restart after backend edits.
- Frontend: `cd frontend && npm run dev` (set `VITE_API_URL` when deploying separately).
- Settings live in `backend/.env` (never commit or print it): `LLM_PROVIDER`, `GROQ_API_KEYS` (comma-separated), `GROQ_MODEL`, `TRIAGE_CHECKPOINT_PATH`, `EDGELEGAL_LOG_LEVEL`.
- A swapped-in checkpoint goes to `backend/app/triage_model/checkpoints/current/best_model.pt`.

---

## 15. Document map: which file to trust for what

| File | Use it for | Caveat |
|---|---|---|
| `docs/EdgeLegal_Documentation.md` (this file) | Full, honest picture of project, scope, status, research gap | Update after each milestone |
| `CLAUDE.md` | Detailed engineering state, gotchas, decisions, for AI sessions | Long, dense, edited by several sessions, not a readable document |
| `docs/EdgeLegal_Project_Guide.md` | Original scope, design, experiment plan, definition of done | Written before results exist |
| `researchpaper_claude/literature_review.md`, `notes/` | Research gap and prior art | One paper still unread |
| `researchpaper_claude/paper_draft.md` | Paper structure and results written so far | 5.3, 5.4 pending |
| `docs/EdgeLegal_Implementation_Summary.md`, `technical_architecture.md`, `architecture123.md` | History only | Outdated, do not use for status |
| [EdgeLegal Research Progress](EdgeLegal_Progress.md) | Current implementation, experiments, decisions, and next step | Update after each research milestone |

---

## 16. Glossary

- **NLI:** natural language inference, deciding whether a premise entails, contradicts, or is neutral to a hypothesis.
- **Hypothesis:** a fixed statement about the contract (17 in ContractNLI).
- **Triage:** sorting cases by difficulty so cheap handling covers the easy ones and expensive handling covers the hard ones.
- **Confidence:** the classifier's highest softmax probability.
- **Calibration / ECE:** whether stated confidence matches real accuracy. Expected calibration error is the weighted average gap between confidence and accuracy across confidence bins; lower is better.
- **AUROC (of confidence for correctness):** whether higher confidence actually ranks right answers above wrong ones. This is discrimination, separate from calibration.
- **Platt scaling / isotonic regression:** post-hoc ways to rescale confidence so it matches observed accuracy.
- **McNemar test:** paired significance test comparing two configs on the same pairs, using only pairs where exactly one is right; statistic (|b - c| - 1)^2 / (b + c).
- **Gap filter:** retrieval rule that keeps the top clause and drops candidates after a large relative distance jump.
- **Paper-grade:** a result produced with the local LLM, on the full test split, with repeats. Anything else is smoke-scale.
