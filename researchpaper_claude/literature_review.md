# EdgeLegal — Literature Review & Gap Analysis

Compiled 2026-09-25 via web search (WebSearch/WebFetch) across arXiv, ACL Anthology, IEEE
Xplore, ACM Digital Library, SpringerLink, and Semantic Scholar. See `search_log.md` for
what was searched and a note on coverage limits. Per-paper detail is in `notes/` — read
`notes/03-legalwiz-CLOSEST-COMPETITOR.md` first if you only read one file.

## The honest headline

The core mechanism — a small classifier's confidence gates whether a case is accepted
directly or escalated to an LLM — is **not new in the abstract**. It exists in general ML
(LLM cascades, confidence-based early exit, Adaptive-RAG) and has already been applied
once specifically to legal contradiction detection: **LegalWiz** (Oct 2025) uses NLI
confidence to gate escalation to GPT-4o for contradiction detection in legal documents.
If you didn't know about LegalWiz before writing your related work section, a reviewer
would have found it for you, and that's a much worse way to find out.

## Follow-up (2026-09-29)

- The one paper flagged as inaccessible ("Multi-Agent Orchestration of Local
  LLMs for Contract Review under Resource Constraints", ACM ICBAR 2025,
  DOI 10.1145/3800227.3800266) is **still inaccessible** — ACM returns 403
  to automated fetches on both `/doi/full/` and `/doi/pdf/`, and it isn't
  indexed on Semantic Scholar yet. Confirmed real (not a phantom citation)
  via the DOI resolving and being findable by title, but contents remain
  unverified. Still a priority follow-up if anyone gets institutional ACM
  access.
- **Correction (2026-10-01):** an earlier version of this follow-up listed
  "ContraGen" (arXiv:2510.03418) as a separate, newly found paper. That was
  wrong: ContraGen is the **v1 title** of the same arXiv paper that v2 renamed
  **LegalWiz** (v1 3 Oct 2025, v2 10 Oct 2025). It is one paper, not two, and
  the LegalWiz analysis in `notes/03` already covers it. No separate
  "concurrent work" entry is needed.

## Research question and overlap

LegalWiz uses confidence in a legal contradiction pipeline. Work on
[small legal models](https://arxiv.org/abs/2603.25944) compares keyword and dense
retrieval on ContractNLI. [CoAL-RAG](https://arxiv.org/abs/2608.17536) uses
retrieval consistency to select a search strategy, and
[Verifiable Legal Contract QA](https://ieeexplore.ieee.org/abstract/document/11690878/)
combines clause retrieval, DeBERTa verification, and confidence gating.

EdgeLegal's proposed empirical comparison asks whether retrieved evidence
changes classifier reliability and whether evidence signals help route cases
at a matched LLM-call rate on ContractNLI. The exact overlap with prior work
still needs a targeted review. See [current research progress](../docs/EdgeLegal_Progress.md)
for completed experiments and proposed work.

## Future work — multi-layer contradiction detection (journal-paper scope, not this conference paper)

Session discussion 2026-09-26 identified a genuine extension beyond the single-document,
hypothesis-vs-clause task above. The earlier search found examples of prior work at each layer:

- **Hypothesis-vs-clause, single document** (what this paper does) — ContractNLI, LegalWiz.
- **Clause-vs-clause, within a document** — solved by **Better Call CLAUSE** (EACL 2026
  Findings; arXiv 2511.00340), a benchmark of ~23,955 LLM-generated + human-validated
  perturbations over CUAD + ContractNLI contracts, covering in-text contradictions
  (conflicting sections of the same contract) and legal-standard violations. Usable
  directly as training/eval data for this layer — no new annotation needed.
- **Clause-vs-clause, across documents** — the closest existing work is **Clash-of-Leges**
  (bilingual, statutory/constitutional-law conflict detection between Italian legal
  articles, 632 conflicting pairs), but it's for statutes/court rulings, not contracts.
  No cross-document contradiction dataset exists for contracts. Cheapest path to build
  one: ContractNLI already applies the same 17 fixed hypotheses across 607 independent
  NDAs — pairing any two contracts where the same hypothesis got opposite labels
  (Entailment vs. Contradiction) yields free cross-document contradiction pairs with zero
  new annotation. A smaller hand-labeled gold eval set (e.g. real MSA+amendment families
  pulled from SEC EDGAR filings) would strengthen this further.

**Proposed extension.** Combine the three checks in one workflow and test
whether their confidence scores can support routing. Novelty requires a separate review.

This is scoped as the **journal paper**, which per the publication-target section above
needs substantially new material beyond the conference version — this qualifies. Do not
start this before Configs A/B/C and calibration (the conference paper) are done; the
architecture reuses the same triage classifier, just with different input pairs per layer.

See `notes/09-multilayer-contradiction-future-work.md` for full detail.

## Full reference list (see `notes/` for detail on each)

| # | Paper | Year | Relevance |
|---|---|---|---|
| 1 | ContractNLI (Koreeda & Manning) | 2021 | The benchmark you evaluate on |
| 2 | LEGAL-BERT (Chalkidis et al.) | 2020 | Candidate triage backbone |
| 3 | DeBERTa / DeBERTaV3 (He et al.) | 2020/2021 | Candidate triage backbone |
| 4 | FrugalGPT (Chen et al.) | 2023 | LLM cascade prior art |
| 5 | CALM (Schuster et al.) | 2022 | Confidence-based early exit prior art |
| 6 | Adaptive-RAG (Jeong et al.) | 2024 | Small-classifier-routes-pipeline prior art |
| 7 | Dynamic Model Routing and Cascading: A Survey | 2026 | Confirms calibration gap field-wide |
| 8 | On Calibration of Modern Neural Networks (Guo et al.) | 2017 | ECE/temperature-scaling method citation |
| 9 | **LegalWiz** | 2025 | **Closest competitor — cite and differentiate explicitly** |
| 10 | Can Small Models Reason About Legal Documents? | 2026 | Supports small-model-on-ContractNLI premise |
| 11 | Deterministic Fuzzy Triage for Legal Compliance | 2026 | Adjacent "triage" framing, no LLM, no ContractNLI |
| 12 | ContraVis | 2026 | HCI/visualization, complementary not competing |
| 13 | Multi-Agent Orchestration of Local LLMs for Contract Review (An et al.) | 2025 | **Even the abstract is publisher-elided — priority follow-up, closest title match found** |
| 14 | ContractEval | 2025 | Adjacent LLM legal-risk benchmark |
| 15 | Better Call CLAUSE | 2025/2026 | Motivates why LLM-alone isn't fully trustworthy; also the within-document contradiction dataset for future multi-layer work (see Future Work section) |
| 16 | ACORD | 2025 | Dataset used by #11, contract drafting task |
| 17 | Summary-Augmented Chunking / LegalBench-RAG | 2025 | Legal RAG retrieval-quality background |
| 18 | ReAct (Yao et al.) | 2023 | Tool-use agent framing (for future cross-reference feature) |
| 19 | Self-RAG (Asai et al.) | 2024 | "Model decides it needs more context" framing |
| 20 | NLP for the Legal Domain: A Survey (Ariai & Demartini) | 2024 | Abstract retrieved, full text paywalled — good related-work opener |
| 21 | Clash-of-Leges | 2025/2026 | Cross-document conflict detection precedent (statutory/constitutional law, not contracts) — closest analog for future cross-document layer |

## Immediate action items

1. **Read the LegalWiz paper in full** (`notes/03`) — it's your closest competitor and
   needs to be cited and differentiated explicitly, not discovered by a reviewer.
2. **Try to get full access to "Multi-Agent Orchestration of Local LLMs for Contract
   Review under Resource Constraints"** (`notes/06`) via institutional/library access —
   it's the closest title match to EdgeLegal's exact premise and I could not read it.
3. This is a genuinely active research area — five of the papers above are from the last
   12 months. That cuts both ways: it confirms the topic is publishable right now, but
   also means the open window could close if something even closer to your exact
   combination appears before you submit. Worth moving on the triage classifier soon
   rather than treating the timeline loosely.
