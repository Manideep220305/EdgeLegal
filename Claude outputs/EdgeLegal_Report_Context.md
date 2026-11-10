# EdgeLegal Major Project Report: Blueprint and Context

> Historical snapshot. Use [current research progress](../docs/EdgeLegal_Progress.md) for implementation and experiment status.

Use this file as the instruction set for generating the EdgeLegal Word report. It describes (1) how the reference report is structured and formatted, (2) the exact EdgeLegal index to follow, section by section, with the content each section must carry and where that content comes from, (3) every diagram, table and chart to produce, and (4) the writing and honesty rules.

Companion file: `EdgeLegal_Report_Manual_Items.md` lists everything that cannot be generated automatically (screenshots, identity pages, measurements not yet run) with the text or description to place instead.

Neither file nor the finished Word document goes into the project folder. They stay in the session outputs only.

---

## 1. What the reference report is

The reference (A2_Documentation.docx) is a completed major-project report from another team in the same college format: "Smart Learning System using Eye Tracking". About 75 pages, 18 tables, about 36 figures, 7 chapters, 19 references, and an appendix of code snippets and screenshots. Its character:

- Everything is written as built and measured. Each claim points to a test, a table, or a measurement.
- Limits are stated openly (for example "a formal user study has not been carried out").
- Requirements are numbered (FR-1, FR-2, ...) and the tests in Chapter 5 are tied to them.
- Test cases carry a small header block: Test Case Number, Type of Testing, Component Being Tested, Date of Testing, Tester, Testing Phase.

EdgeLegal's report must copy this structure and tone, at roughly half the length.

## 2. Target size and look

- Target length: **35 to 42 pages** including front matter and appendices (reference: about 75).
- Page: A4 portrait. Margins 2.54 cm on all sides (reference uses 2.54 cm in the body).
- Font: **Times New Roman**. Body 12 pt, justified, line spacing 1.5, paragraph spacing after about 6 to 10 pt.
- Chapter titles: 16 pt bold, centered, UPPERCASE with number ("1. INTRODUCTION"), each chapter starts on a new page.
- Section titles (2.1): 14 pt bold, left or justified. Sub-section titles (2.2.1): 12 pt bold.
- Table captions: 12 pt bold, centered, **above** the table ("Table 2.2.1 -- Comparative Analysis of Selected Literature"). Figure captions: 12 pt bold, centered, **below** the figure ("Fig 3.4.1 Architecture Diagram"). Line spacing 1.25 for captions.
- Tables: full-width, header row bold with light grey shading, thin borders, 10 or 11 pt text inside tables, line spacing single.
- Code (appendix): Consolas 9 or 10 pt in a bordered single-cell table, grey background.
- Page numbers: roman numerals (i, ii, iii...) for front matter, arabic from Chapter 1, centered footer. No header text.
- Use real Word heading styles (Heading 1 for chapters, Heading 2 for sections, Heading 3 for sub-sections) restyled to the above, so a Table of Contents field works. Use Word caption fields (SEQ) for tables and figures so the List of Figures and List of Tables work.

## 3. EdgeLegal report index (the structure to build)

Chapter and section numbering mirrors the reference exactly. Page budgets are approximate.

### Front matter (roman numerals, about 7 pages)

1. Title page: "A Major Project Report On **EdgeLegal: Privacy-Preserving Contract Analysis & Contradiction Detection**", submitted in partial fulfillment for the Bachelor of Technology in the Department of Computer Science and Engineering, team table (name, roll number), guide, institution block, academic year 2026-2027. Identity details are placeholders (see manual file).
2. Certificate (internal guide, HoD, external examiner lines).
3. Declaration (team table).
4. Acknowledgement (draft text provided in the manual file).
5. Abstract (about 250 to 300 words, in section 4.1 of this file).
6. Table of Contents.
7. List of Figures.
8. List of Tables.

### 1. INTRODUCTION (about 2 pages, prose only, no sub-sections)

### 2. LITERATURE SURVEY (about 5 pages)
- 2.1 Introduction
- 2.2 Related Work: 2.2.1 to 2.2.9 (one short paragraph each, see section 4.2) and **Table 2.2.1 -- Comparative Analysis of Selected Literature**
- 2.3 Identified Gaps: 2.3.1 Confidence Is Routed On but Rarely Verified; 2.3.2 No Evaluation on the Standard ContractNLI Benchmark; 2.3.3 No Matched Three-Way Comparison; 2.3.4 Retrieval Quality Is Assumed, Not Measured; 2.3.5 Privacy and Local Deployment
- 2.4 Problem Statement
- 2.5 Proposed Solution: 2.5.1 Structure-Aware Clause Retrieval; 2.5.2 Fine-Tuned Triage Classifier; 2.5.3 Confidence-Gated Routing; 2.5.4 LLM Escalation with Evidence; 2.5.5 Matched A/B/C Evaluation with Calibration Checks; 2.5.6 Stored History and Analytics
- 2.6 Summary

### 3. SYSTEM REQUIREMENTS SPECIFICATION (about 8 pages)
- 3.1 Introduction: 3.1.1 Purpose; 3.1.2 Scope; 3.1.3 Definitions, Acronyms and Abbreviations (**Table 3.1.3**); 3.1.4 References; 3.1.5 Overview
- 3.2 General Description: 3.2.1 Product Perspective; 3.2.2 Product Functions; 3.2.3 User Characteristics; 3.2.4 General Constraints; 3.2.5 Assumptions and Dependencies
- 3.3 Specific Requirements: 3.3.1 Functional Requirements (**Table 3.3.1**); 3.3.2 Non-Functional Requirements (**Table 3.3.2**); 3.3.3 User Interface Requirements; 3.3.4 Software Requirements (**Table 3.3.4**); 3.3.5 Hardware Requirements (**Table 3.3.5**)
- 3.4 UML Diagrams: 3.4.1 Architecture Diagram; 3.4.2 Use Case Diagram; 3.4.3 Class Diagram; 3.4.4 Sequence Diagrams (a) Upload and Indexing, (b) Analysis with Triage and Escalation, (c) Research Run and Results; 3.4.5 ER Diagram; 3.4.6 Activity Diagram
- 3.5 Summary

### 4. METHODOLOGY (about 8 pages)
- 4.1 Modules: eight modules (module text), **Table 4.1.1 -- Modules and Their Main Source Files**, **Table 4.1.2 -- REST API Operations**
- 4.2 Methodology Proposed for Solution Implementation: a numbered list of the nine implementation steps, **Fig 4.2.1 Overall Workflow of the System**, then 4.2.1 Data and Preprocessing; 4.2.2 Clause Segmentation, Embedding and Retrieval (**Fig 4.2.2**); 4.2.3 Fine-Tuning the Triage Classifier (**Table 4.2.1 -- Main Parameters**); 4.2.4 Confidence, Threshold Selection and Routing (**Fig 4.2.3**); 4.2.5 LLM Escalation and Prompting; 4.2.6 Experimental Configurations and Metrics (**Fig 4.2.4**, **Table 4.2.2**); 4.2.7 Data Storage and Analytics (**Table 4.2.3 -- Database Tables**)
- 4.3 Summary

### 5. TESTING & RESULTS (about 8 pages)
- 5.1 Unit Testing (Test Case 1 header block, **Table 5.1**)
- 5.2 System Testing (Test Case 2 header block, **Table 5.2 -- API System Tests**)
- 5.3 Classifier Accuracy, Calibration and Retrieval Tests: 5.3.1 Dev-Split Evaluation (**Table 5.3**, **Fig 5.3.1** reliability diagram, **Fig 5.3.2** threshold sweep); 5.3.2 In-Pipeline Evaluation (**Table 5.4**, **Fig 5.3.3**); 5.3.3 Retrieval Inclusion Test (**Table 5.5**, **Fig 5.3.4**); 5.3.4 Post-Hoc Recalibration (**Table 5.6**, **Fig 5.3.5**); 5.3.5 Configurations A, B and C (**Table 5.7**, status of the matched run)
- 5.4 Results: screenshots of the application (**Fig 5.4.1 to 5.4.8**), see manual file
- 5.5 Performance Evaluation (**Table 5.8** API response times if measured; LLM prompt and rate-limit figures)

### 6. CONCLUSION AND FUTURE SCOPE (about 2 pages)
- 6.1 Conclusion
- 6.2 Future Scope (bold sub-headings, one paragraph each)

### 7. REFERENCES (about 1 to 2 pages, IEEE style, 15 to 20 entries)

### APPENDIX (about 5 pages)
- I. Code Snippets (A.1 to A.6)
- II. Screenshots (Input, Process and Output), see manual file

---

## 4. Section content: what to write and where the facts come from

Source files: `CLAUDE.md` (engineering state), `docs/EdgeLegal_Documentation.md` (full honest documentation), `docs/EdgeLegal_Project_Guide.md` (scope and design), `researchpaper_claude/literature_review.md` and `notes/00` to `notes/09` (research), `results/*.json`, `backend/app/triage_model/dev_eval_results.json`, and the code itself.

### 4.1 Abstract (draft, adjust numbers only if results change)

Reading a contract to decide whether it supports, contradicts or is silent on a requirement is slow, and sending every check to a large language model is costly and, when the model is hosted, exposes confidential text. EdgeLegal is a web application that treats the problem as triage. A contract is split into clauses, embedded and stored locally, and the clauses relevant to a requirement are retrieved with an adaptive gap filter. A DeBERTa-v3-base classifier fine-tuned on ContractNLI labels the pair as Entailment, Contradiction or NotMentioned and reports a confidence. If the confidence reaches a frozen threshold of 0.95 the label is accepted, otherwise the case is escalated to an LLM that returns a label, reasoning and cited clauses. The system is implemented with a FastAPI back end, a LangGraph routing graph, ChromaDB, an SQLite history store and a React front end, and an experiment runner compares a no-triage baseline, an off-the-shelf NLI gate and the fine-tuned gate. On the ContractNLI dev split the classifier reaches 0.970 accuracy and 0.939 macro-F1 with an expected calibration error of 0.029 when given annotated evidence. In the full pipeline, with retrieved clauses, accuracy on 20 test documents falls to 0.532 and calibration error rises to 0.440, and the gold evidence clause is retrieved only 73.1% of the time. The work therefore reports a verified negative finding about confidence-based routing under realistic retrieval, identifies its causes, and sets out the retraining and retrieval changes needed before the matched three-way comparison can be reported with a local LLM. A matched comparison on the full test split has not yet been completed.

### 4.2 Chapter 1: Introduction (prose, about 7 paragraphs)

1. Contracts are long; checking requirements by hand is slow. Define the three labels in plain words.
2. LLMs can answer but cost time and money per call, and cloud hosting exposes confidential contracts.
3. Small NLI classifiers are cheap but cannot explain and may be wrong without knowing it. Calibration (does 95% confidence mean right 95% of the time) is the key question.
4. The idea: triage. Small model first, LLM only for uncertain cases.
5. Summary of the implemented system (stack and flow).
6. Why it is a research project: the question is not "can we build it" but "can the confidence be trusted to gate the LLM", evaluated on ContractNLI with three configurations. State plainly that the calibration check was done and what it found.
7. Organisation of the report (one sentence per chapter).

### 4.3 Chapter 2: Literature Survey

2.1 Introduction: one paragraph on the field (legal NLP, LLM cascades, confidence-based routing, contradiction detection) and how the survey was done (searched arXiv, ACL Anthology, IEEE Xplore, ACM DL, SpringerLink, Semantic Scholar, September 2026).

2.2 Related Work, one paragraph each (use `researchpaper_claude/notes/`):
- 2.2.1 ContractNLI (benchmark; 607 NDAs, 17 hypotheses, Span-NLI-BERT baseline) (notes/00)
- 2.2.2 LEGAL-BERT and DeBERTa as NLI backbones (notes/01)
- 2.2.3 LLM Cascades and Routing: FrugalGPT, CALM, Adaptive-RAG, routing survey (notes/02)
- 2.2.4 LegalWiz: NLI-confidence-gated escalation for legal contradiction detection, the closest prior work (notes/03). Note it was first posted as "ContraGen" (v1) and renamed LegalWiz (v2); it is one paper.
- 2.2.5 Can Small Models Reason About Legal Documents (notes/04)
- 2.2.6 Deterministic Fuzzy Triage for Legal Compliance (notes/05)
- 2.2.7 ContraVis and Multi-Agent Orchestration of Local LLMs for Contract Review (notes/06; state that the second paper's full text was not accessible)
- 2.2.8 Better Call CLAUSE and related benchmarks (notes/07, notes/09)
- 2.2.9 Calibration of neural networks (Guo et al.), temperature and post-hoc scaling (notes/02)

**Table 2.2.1 -- Comparative Analysis of Selected Literature**: columns S.No., Paper and Year, Method and Metrics, Dataset, Observed Gap. Eight to ten rows from the notes. No invented details; leave a cell short rather than guess.

2.3 Identified Gaps, each 1 paragraph (from `literature_review.md`):
- 2.3.1 Confidence is routed on but rarely verified (routing survey; LegalWiz does no ECE).
- 2.3.2 No confidence-gated system evaluated on ContractNLI (LegalWiz uses 50 synthetic documents).
- 2.3.3 No matched three-way comparison (no triage vs off-the-shelf gate vs fine-tuned gate, McNemar).
- 2.3.4 Retrieval quality is assumed, not measured (our own finding: 73.1% inclusion).
- 2.3.5 Privacy and local deployment constraints (cloud LLMs; limited local compute).

2.4 Problem Statement: 3 short paragraphs, same shape as the reference (the situation, what existing work does, the problem addressed). Core question verbatim from the Project Guide section 1.

2.5 Proposed Solution: 2.5.1 to 2.5.6, one short paragraph each, each pointing back to the gap it answers (Section 2.3.x).

2.6 Summary: one paragraph.

### 4.4 Chapter 3: Requirements and design

3.1.1 Purpose: bullet list of what the system allows (upload, analyze against hypotheses, see label/confidence/engine/evidence, history, analytics, research page, reproducible experiments).

3.1.2 Scope: in scope list (ContractNLI 17 hypotheses; triage; Groq interim and local LLM planned; live app; experiment runner). Out of scope: other contract types, within/cross-document contradiction (future), multilingual, formal user study, paper-grade results.

3.1.3 Table 3.1.3 definitions (about 14 rows): NLI, Entailment, Contradiction, NotMentioned, ContractNLI, Hypothesis, Premise/Clause, Triage, Confidence, Threshold (tau), Calibration, ECE, AUROC, McNemar test, RAG, LLM, DeBERTa, ChromaDB, LangGraph, FastAPI, Groq, Ollama, Platt scaling, isotonic regression. Use the glossary in `docs/EdgeLegal_Documentation.md` section 16.

3.1.4 References: list the standards/papers used inside the SRS (IEEE 830 style notes optional; keep to project references).

3.2.1 Product perspective: stand-alone web app; React SPA client, FastAPI server, SQLite history, local ChromaDB per upload, local classifier checkpoint; external: Groq API (interim), Ollama (planned), Hugging Face model weights. Honest note that the Groq path sends escalated clause text to a hosted service.
3.2.2 Product functions: summarised list.
3.2.3 User characteristics: legal analyst / reviewer; researcher using the Research page; no ML knowledge needed.
3.2.4 Constraints: free-tier LLM rate limits (30 requests per minute, 1,000 per day, 8K tokens per minute, 200K tokens per day, shared per organisation); CPU-only laptops (about 12 to 16 GB RAM, no GPU); only ContractNLI (NDAs); a 738 MB classifier checkpoint.
3.2.5 Assumptions: contracts are English text PDFs; threshold frozen from dev; ContractNLI labels are gold.

3.3.1 **Table 3.3.1 -- Functional Requirements** (ID, Requirement, Description), at least these:
- FR-1 Contract upload (PDF accepted, document id returned, stored; recorded in history)
- FR-2 Clause segmentation (structure-aware segmentation with page numbers)
- FR-3 Embedding and indexing (bge-small-en embeddings into a per-document ChromaDB collection)
- FR-4 Hypothesis input (free text requirement per analysis)
- FR-5 Retrieval with adaptive gap filter (keep rank 1, drop after >15% relative distance jump)
- FR-6 Triage classification (label and confidence from the fine-tuned classifier)
- FR-7 Confidence routing (accept at confidence >= 0.95, otherwise escalate)
- FR-8 LLM escalation (label, reasoning, cited clauses; key rotation on rate limits)
- FR-9 Result report (label, confidence, engine, triage decision trail, cited clauses with page)
- FR-10 History (saved analyses, filters, search, reopen, original file)
- FR-11 Analytics (documents analyzed, contradiction rate, average confidence, classifier vs LLM split, top contradicted hypotheses)
- FR-12 Research metrics (dev metrics, reliability bins, threshold sweep, A/B/C results from saved files with non-paper-grade and split-mismatch warnings)
- FR-13 Experiment runner (configs A, B, C, accuracy, macro-F1, ECE, latency, McNemar)
- FR-14 Logging and error recording (every stage and failure logged with context)

3.3.2 **Table 3.3.2 -- Non-Functional Requirements** (Category, Description): Privacy (design goal; true only with the local LLM, state this), Security (path-traversal protection on the file endpoint, API keys kept in a git-ignored environment file), Performance, Reliability (key rotation, per-pair errors recorded not hidden), Maintainability (modular, shared logging), Reproducibility (frozen threshold, results files flagged paper_grade, same split for all configs), Usability (dark/light theme, reduced-motion support, responsive).

3.3.3 User interface requirements: the five pages and what each shows (Landing, App with step rail and decision trail, History, Analytics, Research).

3.3.4 **Table 3.3.4 -- Software Requirements**: OS (Windows 11 development); Python 3.11; FastAPI, Uvicorn; PyTorch and Hugging Face Transformers (DeBERTa-v3-base checkpoint); sentence embedding model BAAI/bge-small-en; ChromaDB; LangGraph; SQLite; Groq API client; Ollama (planned); Node.js, React, TypeScript, Vite, recharts; Playwright for visual checks. Take exact versions from `backend/requirements.txt` and `frontend/package.json` when generating; do not guess versions.

3.3.5 **Table 3.3.5 -- Hardware Requirements**: development laptops (CPU only, 12 to 16 GB RAM, no GPU); training on Kaggle T4 GPU (3 epochs); about 1 GB disk for the checkpoint and models; internet for the interim Groq LLM.

3.4 UML diagrams: see the diagram specification in section 5 of this file. Each diagram gets one introductory paragraph and a caption.

3.5 Summary.

### 4.5 Chapter 4: Methodology

4.1 Eight modules (one paragraph each, plus the file table):
1. Training Pipeline Module (`triage_model/train.py`, `dataset.py`, `evaluate.py`): fine-tunes `cross-encoder/nli-deberta-v3-base` on ContractNLI train; class-weighted cross-entropy, linear warmup, gradient accumulation, early stopping on macro-F1 (patience 2), batch 32, learning rate 2e-5, up to 3 epochs, Kaggle T4. Label order Contradiction=0, Entailment=1, NotMentioned=2.
2. Document Parsing and Retrieval Module (`parsing/clause_parser.py`, `embeddings/embedder.py`, `vectorstore/chroma_client.py`): segmentation, bge-small-en embeddings, ChromaDB, gap filter.
3. Triage Classifier Module (`triage_model/model.py`, `infer.py`): `TriageModel.predict(premise, hypothesis)` returns label and max-softmax confidence.
4. Agent Graph Module (`agent/graph.py`, `nodes.py`, `state.py`): retrieve, triage, confidence_check, accept, call_llm; Config A skips triage.
5. LLM Provider Module (`llm/groq_client.py`, `ollama_client.py`, `prompts.py`): Groq key rotation and retries, local Ollama client, multi-clause JSON prompt, tolerant parsing.
6. API and Service Module (`main.py`, `agent/live_service.py`, `history_db.py`, `research.py`, `schemas.py`, `logging_config.py`).
7. Experiment and Analysis Module (`experiment_runner.py`, `retrieval_accuracy_check.py`, `recalibration_preview.py`).
8. Front-End Module (`frontend/src/pages/*`, components `UploadPanel`, `HypothesisPanel`, `PDFViewer`, `ReportPanel` with decision trail, `StatTile`, `AppHeader`).

**Table 4.1.2 -- REST API Operations**: GET /health; POST /upload; POST /analyze; GET /history; GET /history/documents; GET /history/{analysis_id}; GET /analytics; GET /research/metrics; GET /documents/{doc_id}/file (columns: Method, Endpoint, Purpose, Main response fields).

4.2 Nine implementation steps (numbered list, as in the reference): requirement analysis and literature survey; design; data setup (ContractNLI); fine-tune and verify classifier; retrieval layer; routing graph; LLM tier; web application and history; experiments, calibration and testing.

4.2.1 to 4.2.7: write each from `CLAUDE.md` and `docs/EdgeLegal_Documentation.md` sections 5, 8, 9. Include **Table 4.2.1** parameters (checkpoint, epochs, batch, learning rate, patience, max sequence length if stated in code, threshold 0.95, gap threshold 15%, embedding model, LLM model, LLM temperature/greedy decoding for local runs). Include the routing pseudocode and the threshold-selection procedure (sweep 0.60 to 0.95 on dev only, freeze before test). **Table 4.2.2** metrics and their meaning. **Table 4.2.3** database tables from `history_db.py`: `documents(id, filename, uploaded_at, page_count)` and `analyses(id, document_id, hypothesis, label, confidence, source, created_at, report_json)` with indexes on `created_at` and `document_id`.

### 4.6 Chapter 5: Testing and results (strict honesty)

Rule: report only what was actually run. For each test group give the header block (Test Case Number, Type, Component, Date, Tester, Testing Phase) then a short paragraph and a table with Expected and Actual results.

5.1 Unit Testing: unit-level checks that can truly be run and logged (for example: history database functions for filters, search escaping, pagination and empty analytics; confidence_check routing at and around 0.95; ECE and McNemar computations on small hand-built inputs; label-order mapping). If a test is not run during report building, it must not appear as passed. If nothing can be run, say in one paragraph that automated unit tests were limited and list what exists.

5.2 System Testing: API-level checks through FastAPI TestClient with the RAG stub where the heavy model cannot load (health, upload validation, analyze for missing document, history list and filters, path traversal on the file endpoint, analytics on an empty database, research metrics reading result files). Record actual outputs.

5.3.1 Dev-split evaluation, **Table 5.3**: accuracy 0.9701; macro precision 0.9305, recall 0.9486, F1 0.9391; F1 per class Entailment 0.9699, NotMentioned 0.9988, Contradiction 0.8485; ECE 0.0285 (10 bins); n = 1,037; frozen threshold 0.95 with coverage 98.4% and macro-F1 0.9511 among kept cases. Charts from `dev_eval_results.json`.
5.3.2 In-pipeline evaluation, **Table 5.4** (dev vs in-pipeline): accuracy 0.9701 vs 0.5324; macro-F1 0.9391 vs 0.4796; ECE 0.0285 vs 0.4396; coverage at 0.95 98.4% vs 89.1%; 309 of 340 pairs at about 0.994 confidence with about 55% accuracy. Explain the premise mismatch (evidence spans and, for NotMentioned, the full document in training vs one retrieved clause live).
5.3.3 Retrieval inclusion test, **Table 5.5**: gold evidence clause in the retrieved set 73.1% overall (Entailment 72.7%, Contradiction 74.5%) over 1,188 cases of the 123-document test split. Treat this as filtered-set evidence inclusion, conditional on a span mapping to a clause.
5.3.4 Post-hoc recalibration, **Table 5.6**: AUROC 0.61; held-out ECE raw 0.440, Platt 0.040, isotonic 0.126; after Platt no pair above 0.7 so a reliable threshold accepts about 0%; isotonic about 12% accepted at 0.67 accuracy. Label as exploratory, 20 documents, grouped 5-fold by document.
5.3.5 Configurations A, B, C, **Table 5.7**: show what exists (smoke runs: A and B on 2 documents, C on 2 and 20 documents, all non-paper-grade, Groq) and mark the matched 20-document comparison as pending. Do not fill McNemar values until a result file exists.

5.4 Results: screenshots (manual file) with one-line descriptions.

5.5 Performance Evaluation: LLM prompt size (about 1,100 tokens average, max about 2,700, about 2.5 clauses per call), first-attempt success about 98% in logs, Groq free-tier limits and their effect, and API response times only if measured (script provided in the manual file). State that latency on the final local-LLM configuration has not been measured.

### 4.7 Chapter 6

6.1 Conclusion (4 paragraphs): what was built; what was found (strong dev numbers, failed live calibration, the two causes, recalibration shows calibration is fixable but usefulness is not); what the verification covered; the limits (single dataset, interim LLM, no paper-grade run, matched comparison pending, privacy claim depends on the local LLM).

6.2 Future Scope (bold headings with one paragraph each): Retraining on Deployment-Realistic Premises; Improving Retrieval (gap threshold, hybrid dense plus lexical retrieval); Local LLM and Paper-Grade Evaluation; Within-Document Contradiction Detection (Better Call CLAUSE data); Cross-Document Contradiction Detection (ContractNLI-derived pairs, amendments); Privacy-Aware Handling of Terms and Conditions; Citation Highlighting (bounding boxes); Other Contract Types and Languages.

### 4.8 References

IEEE style, numbered in order of first citation. Use only entries whose details are confirmed in `researchpaper_claude/notes/` or in this list; mark any unverified entry for the user to check. Candidate list (verify details before final): ContractNLI (Koreeda and Manning, Findings of EMNLP 2021); LEGAL-BERT (Chalkidis et al., Findings of EMNLP 2020); DeBERTa (He et al., ICLR 2021); DeBERTaV3 (He, Gao, Chen, arXiv 2111.09543); FrugalGPT (Chen, Zaharia, Zou, arXiv 2305.05176, 2023); On Calibration of Modern Neural Networks (Guo et al., ICML 2017, pp. 1321-1330); CALM (Schuster et al., NeurIPS 2022); Adaptive-RAG (Jeong et al., NAACL 2024); LegalWiz (Mantravadi et al., arXiv 2510.03418, 2025); Better Call CLAUSE (arXiv 2511.00340, EACL 2026 Findings); ECON (Cheng et al., arXiv 2410.04068, 2024); ReAct (Yao et al., ICLR 2023); Self-RAG (Asai et al., ICLR 2024); BGE embeddings; McNemar (1947); Platt scaling (1999); isotonic regression for calibration (Zadrozny and Elkan, 2002); gpt-oss model card (OpenAI, 2025). Software references (FastAPI, ChromaDB, LangGraph) may be given as footnote-style entries.

### 4.9 Appendix I: code snippets (take from the real files, trimmed, with file path in the caption)

A.1 Adaptive gap filter (`vectorstore/chroma_client.py`, `gap_filter`); A.2 `confidence_check_node` (`agent/nodes.py`); A.3 `build_graph` (`agent/graph.py`); A.4 ECE computation (`triage_model/evaluate.py`, `compute_calibration`); A.5 McNemar test (`experiment_runner.py`, `compute_significance`); A.6 history schema (`history_db.py`, `SCHEMA`); A.7 Platt and isotonic fit (`recalibration_preview.py`). Copy code verbatim from the files; do not retype from memory.

---

## 5. Diagram and chart specification (what Claude generates)

All figures are produced as PNG at 300 dpi, width 15 to 16 cm, white background, black outlines, one accent colour (dark red `#B3121B`) for emphasis, Times-like or Arial 9 to 11 pt labels so they stay legible in print. Generate with Python (graphviz if installed, otherwise matplotlib patches and arrows). Every diagram must reflect the real code, not an idealised design.

**Fig 3.4.1 Architecture Diagram.** Layers left to right: Client (React SPA: Upload, Hypothesis, PDF Viewer, Report with Decision Trail, History, Analytics, Research) -> API (FastAPI: /upload, /analyze, /history, /analytics, /research/metrics) -> Agent (LangGraph: retrieve, triage, confidence_check, accept, call_llm) with side stores: ChromaDB (per-document collections), SQLite history, results/*.json, checkpoint file; external: Groq API (interim) and Ollama (planned). Mark the Groq link with a dashed line and "interim, hosted".

**Fig 3.4.2 Use Case Diagram.** Actors: Reviewer (user), Researcher, LLM Provider (external, Groq/Ollama). Use cases: Upload contract; Enter hypothesis; Run analysis (includes Retrieve clauses, Classify with Tier-1; extends Escalate to LLM when confidence below threshold); View decision trail and citations; Browse history; View analytics; View research metrics; Run experiment (A, B, C).

**Fig 3.4.3 Class Diagram.** Classes with real names and key members: `Clause` (text, page, section heading), `PageBlock`, `Embedder` (load, embed), `TriageModel` (load, predict), `GroqKeyRotator` (get_current_key, rotate_to_next_key, chat), `OllamaClient` (generate, query_llm), `GroqAgentClient` (query_llm), `AgentState` (hypothesis, retrieved_clauses, triage_label, triage_confidence, threshold, routing_decision, final_label, source_engine, reasoning_text, cited_clauses), pydantic schemas `AnalyzeRequest`, `ReportResult`, `TriageDetail`, `ReportCitation`, `HistoryItem`, `ResearchMetrics`, `ConfigResult`. Show associations (graph uses Embedder, TriageModel, LLM client; ReportResult contains TriageDetail and ReportCitation).

**Fig 3.4.4 (a) Sequence: Upload and Indexing.** User -> Frontend -> `POST /upload` -> parse PDF -> segment clauses -> embed (Embedder) -> add to ChromaDB collection -> record document in history DB -> return doc_id.
**Fig 3.4.4 (b) Sequence: Analysis.** User enters hypothesis -> `POST /analyze` -> `run_agent` -> retrieve (ChromaDB + gap filter) -> TriageModel.predict -> confidence_check -> alt accept: build report from classifier; alt escalate: LLM client call (with key rotation on 429) -> build report with citations -> record analysis -> response -> Frontend renders verdict and decision trail.
**Fig 3.4.4 (c) Sequence: Research Run and Results.** Researcher runs `experiment_runner` for A, B, C -> per-pair results written to `results/*.json` -> Research page requests `/research/metrics` -> `research.py` reads files and dev_eval_results.json -> response with guard flags -> page shows tiles, reliability diagram, sweep, A/B/C table.

**Fig 3.4.5 ER Diagram.** `documents (PK id, filename, uploaded_at, page_count)` one-to-many `analyses (PK id, FK document_id, hypothesis, label, confidence, source, created_at, report_json)`; indexes noted; cascade delete noted.

**Fig 3.4.6 Activity Diagram.** Start -> upload contract -> extract text -> segment clauses -> embed and store -> enter hypothesis -> retrieve clauses -> Tier-1 classify -> decision (confidence >= 0.95?) yes: accept; no: escalate to LLM -> aggregate -> report with citations -> store history -> end.

**Fig 4.2.1 Overall Workflow.** Horizontal flow of the nine implementation phases with the artefact produced at each (dataset splits, checkpoint, retrieval layer, graph, LLM tier, web app, experiment files).
**Fig 4.2.2 Retrieval and Gap Filter.** Illustrative bar chart of ranked distances with the kept clause(s) and the cut-off at the first >15% relative jump. Label it "illustration" unless drawn from a real query.
**Fig 4.2.3 Confidence Routing Logic.** Flowchart: retrieved clauses -> classifier -> confidence -> diamond (>= 0.95) -> accept / escalate -> final label and source.
**Fig 4.2.4 Experimental Configurations.** Three parallel lanes: A (retrieve -> LLM), B (retrieve -> off-the-shelf NLI gate -> accept or LLM), C (retrieve -> fine-tuned gate -> accept or LLM), same split and threshold.

**Charts from real data (matplotlib):**
- Fig 5.3.1 Reliability diagram on dev (from `dev_eval_results.json` calibration bins; dots sized by example count, no connecting line, as in the Research page).
- Fig 5.3.2 Threshold sweep, coverage and macro-F1 vs tau (from the same file, y-axis zoomed to 0.90 to 1.00).
- Fig 5.3.3 Dev vs in-pipeline bars: accuracy, macro-F1, ECE (0.9701 / 0.5324, 0.9391 / 0.4796, 0.0285 / 0.4396).
- Fig 5.3.4 Retrieval inclusion bars: overall 73.1%, Entailment 72.7%, Contradiction 74.5%.
- Fig 5.3.5 ECE before and after recalibration: raw 0.440, Platt 0.040, isotonic 0.126.
- Fig 5.5.1 API response times only if the timing run exists.

Do not produce any chart for A/B/C results until real result files with a matched run exist.

---

## 6. Writing rules

- Present tense for the system ("The classifier returns..."), past tense for what was done or measured ("The test was run on...").
- Plain, factual prose. No marketing language. Write numbers exactly as in the sources. Do not round differently from the sources.
- Every table and figure is introduced in the text before it appears ("Table 5.4 compares...") and referred to by number.
- Cross-reference sections as in the reference: "(Section 2.3)".
- Cite with numbered brackets [1], [2] in order of first appearance.
- Never claim paper-grade results, a privacy guarantee, or a passed test that was not actually produced. Where something is pending, write "has not yet been carried out" and say what is needed.
- Do not use first-person singular. Use "the team" or the passive sparingly.
- Do not include personal names in generated body text. Identity pages use the placeholders listed in the manual file.
- Keep the key honest message visible in the Abstract, Chapter 5 and Chapter 6: strong dev results, failed live calibration, two identified causes, matched comparison pending.

## 7. Build procedure (when asked to generate the Word document)

1. Read this file and `EdgeLegal_Report_Manual_Items.md`.
2. Read the source files named in section 4 fresh (do not rely on this summary for numbers); pull exact code for the appendix and exact versions for the software table.
3. Run whatever tests and measurements are actually runnable (unit and API checks, timing script). Record outputs. Anything that cannot run goes to the manual file's pending list instead of the report.
4. Generate all diagrams and charts (section 5), check each visually for legibility and for overlap.
5. Build the .docx with real heading styles, caption fields, a Table of Contents, List of Figures and List of Tables, page-number sections (roman then arabic), and placeholder boxes for every manual item (caption, one-line description, "Insert screenshot here").
6. Render to PDF, check page count (target 35 to 42), fix layout problems, update fields.
7. Deliver the .docx and updated manual file in the session outputs only. Do not write anything into the project folder.

## 8. Mapping: reference report sections to EdgeLegal

| Reference section | EdgeLegal equivalent |
|---|---|
| Title, certificate, declaration, acknowledgement | Same pages, placeholders for identity details |
| Abstract | Section 4.1 above (honest, includes the negative finding) |
| 1 Introduction | Contracts, LLM cost and privacy, triage idea, calibration question |
| 2 Literature survey (14 papers, gaps, solution) | 9 related-work items, Table 2.2.1, five gaps from our own review, solution mapped to gaps |
| 3 SRS (FR/NFR tables, UML) | FR-1 to FR-14, NFR table, same UML set plus activity diagram |
| 4 Methodology (10 modules, gaze pipeline) | 8 modules, retrieval, fine-tuning, routing, LLM tier, experiment design |
| 5 Testing (198 unit tests, 26 API checks, calibration runs, API timing) | Unit and API checks actually run, dev and in-pipeline evaluation, retrieval test, recalibration, A/B/C status, performance |
| 6 Conclusion and future scope | Conclusion with the negative finding; future scope led by retraining and retrieval |
| 7 References (19) | 15 to 20 verified entries |
| Appendix code snippets A.1 to A.9 | A.1 to A.7 from real files |
| Appendix screenshots (input, process, output) | Manual: upload, analysis, result, history, analytics, research |
