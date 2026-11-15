# EdgeLegal: Confidence-Routed Legal Contradiction Detection — Paper Draft

**Working status (2026-11-13).** This draft reports the development
diagnostics, retrieval comparisons, and small A/B/C pilot. The final held-out routing comparison
remains pending. See the [research handoff](../docs/EdgeLegal_Progress.md)
for experiment decisions and artifact links.

---

## Working title

"Can a Fine-Tuned Legal NLI Classifier Be Trusted to Gate LLM Escalation?
A Calibration-Checked Confidence-Routing Study on ContractNLI"

(Alt, shorter: "Confidence-Gated Triage for Legal Contradiction Detection:
An Honest Calibration Check")

---

## Abstract (development results)

Confidence-based routing can reduce the number of calls to a large language
model (LLM), but it depends on the reliability of the smaller model's
predictions. We study a fine-tuned DeBERTa classifier for document-level
natural language inference on ContractNLI. On 1,037 development cases,
rank-one retrieval yielded 51.5% accuracy. Supplying annotated evidence
corrected 278 of the 301 rank-one errors on Entailment and Contradiction.
Retrieving up to three clauses increased the number of cases with a
complete visible evidence span from 222 to 391 of 614, while a
confidence-based override raised accuracy to 53.7% and reduced recall on
genuine Not Mentioned cases. With the parser and rank-one classifier fixed,
BM25 reached 59.11% accuracy and caught 39 of 95 contradictions. Reranking
Hybrid retrieval with MiniLM reached 58.24% accuracy and caught 49
contradictions. Fine-tuning DeBERTa on overlapping document windows
raised accuracy to 86.11% on development. The frozen checkpoint reached
89.09% on 103 held-out contracts without LLM calls. In a separate
three-contract routing pilot, the
fine-tuned gate would have skipped 42 of 51 LLM calls and made 24 errors
among those 42 accepted predictions. These development findings show how
evidence selection and overconfident predictions can undermine routing.
The full held-out accuracy and call-rate comparison is pending.

---

## 1. Introduction

Legal contract review — checking whether clauses are consistent,
contradictory, or silent relative to a standard obligation — is manual,
slow, and expensive. LLM-based retrieval-augmented generation (RAG)
systems can automate this, but every clause comparison requires a full LLM
inference call, which compounds in cost and latency on long contracts or
when checking many clauses against many standard obligations — especially
in privacy-preserving deployments constrained to local (non-cloud) LLMs.

Small NLI classifiers (fine-tuned Transformers such as DeBERTa or
LegalBERT) can predict Entailment/Contradiction/NotMentioned relationships
between a clause and a hypothesis far faster and cheaper than an LLM, but
cannot generate human-readable legal reasoning, and their predictions are
not inherently trustworthy without validation.

**Research question**: Can a fine-tuned legal NLI classifier act as a
confidence-gated triage layer in front of an LLM-based reasoning pipeline —
accepting its own predictions when confident and escalating only uncertain
cases — while preserving accuracy, reducing LLM calls, and remaining
well-calibrated enough that its confidence can actually be trusted for that
routing decision?

This is not a claim of a new model or algorithm. The contribution is the
system-level question and an honest answer to it, including whether the
classifier's confidence is trustworthy enough to route on — not just
whether it is accurate on average. If the triage layer does not reduce LLM
calls without hurting accuracy, or if its confidence turns out not to be
trustworthy, that is still a valid, honestly reportable finding.

---

## 2. Related Work

Koreeda and Manning define ContractNLI as a document-level inference task
and introduce a baseline that predicts verdicts and identifies evidence
spans. Vaddi evaluates prompted small language models using lexical and
dense retrieval on several legal tasks, including ContractNLI. Buchmann,
Liu, and Gurevych also use ContractNLI in their long-document benchmark
and evaluate whether evidence quality can support selective prediction.
Together, these studies motivate measuring evidence and answer quality
separately. [ContractNLI](https://aclanthology.org/2021.findings-emnlp.164/) ·
[Small models for legal documents](https://arxiv.org/abs/2603.25944) ·
[Attribute or Abstain](https://aclanthology.org/2024.emnlp-main.463/)

FrugalGPT learns cascades across LLMs to balance quality and inference
cost. Guo et al. study whether predicted confidence matches observed
correctness and establish a reference for expected calibration error.
In the legal domain, LegalWiz uses an NLI classifier to mine
contradictions in synthetic documents. Its mining agent escalates
predicted contradictions or low-confidence pairs to GPT-4o. CoAL-RAG
uses retrieval consistency to choose a search strategy.
[FrugalGPT](https://arxiv.org/abs/2305.05176) ·
[Calibration of modern neural networks](https://proceedings.mlr.press/v70/guo17a.html) ·
[LegalWiz, §3.2](https://arxiv.org/html/2510.03418v2#S3.SS2) ·
[CoAL-RAG](https://arxiv.org/abs/2608.17536)

Our study measures how evidence retrieval and token truncation affect
a fixed DeBERTa classifier's document-level verdicts and confidence.
The paired development comparisons track whether annotated spans reach
the classifier and how often supplying annotated evidence changes wrong
verdicts. The planned held-out evaluation measures the accuracy and
call-rate trade-off when an LLM handles escalated cases. The
[literature review](literature_review.md) records the scope of the
comparison and remaining source checks.

---

## 3. System Architecture

Pipeline: PDF -> clause segmentation (structure-aware, not naive
block-chunking) -> embedding (BAAI/bge-small-en) -> semantic retrieval with
an adaptive gap filter (keep rank-1, drop any candidate with a >15%
relative distance jump from it) -> confidence-routed agent:

```
retrieve -> triage -> confidence_check --accept----> accept    -> END
                                        \--escalate--> call_llm -> END
```

The triage classifier is a pairwise (premise, hypothesis) NLI model
(`cross-encoder/nli-deberta-v3-base`, fine-tuned on ContractNLI's train
split), run once against the single highest-ranked retrieved clause. If its
confidence is at or above a frozen threshold (tau = 0.95, selected on the
validation split only, never touched against test), its label is accepted
directly with no LLM call. Below threshold, all gap-filtered candidate
clauses are sent to the LLM for reasoning and a final label. Implemented as
a LangGraph state machine (`agent/graph.py`), used identically by the
research evaluation harness and the live demo application (see
implementation note below).

The app and the standard runner share the routing graph, retrieval, and
triage code. The app extracts text from uploaded PDFs and escalates through
Groq. The benchmark indexes ContractNLI's supplied text. The offline
development diagnostics evaluate the classifier and retrieved evidence
without invoking the LLM.

---

## 4. Experimental Design

**Data and evaluation unit.** ContractNLI contains 607 NDAs and 17 fixed
hypotheses. Its train, development, and test splits contain 423, 61, and
123 contracts. The decision unit is a contract–hypothesis pair with a
document-level label of Entailment, Contradiction, or Not Mentioned. The
development comparisons use all 1,037 pairs. Evidence evaluation uses the
614 pairs with annotated spans. Not Mentioned has no positive span.

**Classifier and evidence comparisons.** The prepared-input evaluator
supplies annotated spans for Entailment and Contradiction and document text
for Not Mentioned. The retrieval baseline supplies the top clause to the
same checkpoint. An evidence span is visible if the tokens overlapping at
least one complete annotated span remain after the 512-token pair limit.
We rerun wrong Entailment and Contradiction cases with the annotated spans
to test how often providing that evidence changes the answer. That rerun
is an error analysis, not a score on all 1,037 cases.

**Parser and scoring variants.** We compare the default parser with an
opt-in split for clauses over 2,000 characters, holding the retriever and
checkpoint fixed. The top-three comparison uses the default parser and
preserves the saved top-one prediction. Its primary rescue rule can replace
a rank-one Not Mentioned verdict if an alternate gives a positive verdict
at confidence ≥0.95 and qualifying alternatives do not disagree. We
compare that rule with agreement between both alternates, the existing
15% retrieval-distance band, a logistic verifier evaluated in five
contract-grouped development folds, and a hypothesis-conditioned TF-IDF
evidence scorer trained on the 423 training contracts. The scorer uses
visible clause text and a fixed 0.50 cutoff to filter eligible overrides.

**Retrieval variants.** We compare the default BGE embeddings with Chroma
cosine ranking, per-contract BM25 with case-folded word tokens
(`k1=1.2`, `b=0.75`), and equal-weight reciprocal rank fusion
(`k=60`). A fourth arm uses the frozen MS MARCO MiniLM-L-6-v2
cross-encoder to reorder Hybrid's five candidates under a 512-token limit.
All four arms score the first clause with the same DeBERTa checkpoint
and use the default parser. We also audit exact numbered references in
the initial BGE result by looking up unique headings within the same
contract. That audit measures raw evidence containment.

**Routing comparison.** A sends all cases to the LLM. B routes through the
off-the-shelf DeBERTa NLI checkpoint, and C through the ContractNLI-tuned
checkpoint. Both gates use the existing 0.95 threshold. The November pilot
uses the same retrieved text and LLM prompt in all three configurations,
with no classifier hint, and reuses identical responses. It reports
logical calls per configuration. The standard runner currently supplies a
preliminary classifier verdict for B/C escalations. For the final matched
comparison, use identical prompts across arms and count failed requests
and retries separately. Report accuracy, per-class recall, LLM call rate,
and measured latency where available. Use paired uncertainty estimates
that keep all hypotheses from each contract together.

The [proposed held-out protocol](../docs/EdgeLegal_Project_Guide.md#64-proposed-held-out-protocol-2026-11-10)
defines a primary cohort of 103 test contracts after excluding the 20
previously explored in the classifier diagnostic. It also specifies a
matched-call-budget comparison and decision criteria to lock before
scoring. The earlier retrieval-only inspection of the official test split
will be disclosed.

## 5. Results

### 5.1 Prepared inputs, retrieval, and evidence visibility

The prepared-input development evaluation reached 97.0% accuracy, 0.939
macro F1, and 0.029 expected calibration error (ECE). Rank-one retrieval
on the same development set yielded 51.5% accuracy. The prepared and
retrieved premises differ, so the 97.0% result does not establish system
accuracy. The 20-contract exploratory test sample is separate from the
paired development comparison. It yielded 53.2% classifier accuracy and
0.440 ECE. At 0.95, the gate accepted 303 of 340 predictions, of which
136 were wrong. Escalation was stubbed in this diagnostic.

| Paired development measure | Default parser | Oversized-clause split |
|---|---:|---:|
| Correct rank-one verdicts | 534/1,037 (51.5%) | 571/1,037 (55.1%) |
| Top clause contains an annotated span | 256/614 | 218/614 |
| Complete annotated span visible after tokenization | 222/614 | 218/614 |
| Inputs exceeding 512 tokens | 195/1,037 | 1/1,037 |
| Wrong Entailment/Contradiction verdicts corrected by annotated inputs | 278/301 | 208/228 |

The default parser produced 34 apparent evidence hits that lost at least
part of the span during tokenization. Giving the classifier annotated text
also corrected 45 of 55 baseline errors where a complete span was already
visible. The error analysis therefore motivates examining both evidence
selection and the rest of the context the classifier receives. The split
parser improved accuracy and reduced truncation, but missed the
predeclared top-one evidence-hit criterion and remains opt-in.
[Parser comparison](../results/oversized-clause-dev/report.md) ·
[Evidence visibility data](../results/oversized-clause-dev/evidence-visibility.json)

### 5.2 Additional clauses and override rules

Top three increased visible evidence coverage from 222 to 391 of 614 cases.
The primary rescue rule fixed 96 errors and introduced 73, all on cases
labeled Not Mentioned. The net gain was 2.22 percentage points, with a
paired 95% contract-bootstrap interval of −0.87 to +5.30 points.

| Default-parser decision rule | Correct / 1,037 | Contradictions caught / 95 | Gold Not Mentioned caught / 423 | New errors vs. rank one |
|---|---:|---:|---:|---:|
| Rank one | 534 | 34 | 221 | 0 |
| Top-three rescue at 0.95 | 557 | 47 | 148 | 73 |
| Both alternates agree | 551 | 39 | 215 | 6 |
| Retrieval-distance filter | 555 | 44 | 173 | 48 |
| Logistic verifier | 553 | 36 | 215 | 6 |
| Train-only text evidence scorer | 573 | 38 | 220 | 1 |

The text scorer used 24,586 training clause pairs, with 4,553 visible
evidence positives. It added 22 correct verdicts over agreement and
introduced five fewer errors. The paired 95% contract-bootstrap interval
for the accuracy difference was +0.58 to +3.86 percentage points. It
caught one fewer contradiction than agreement and missed that development
guardrail. The parser comparison and the top-three variants are separate
experiments. No result in this table combines oversized-clause splitting
with a top-three rule. [Cached top-three cases](../results/oversized-clause-dev/top-three.json)
· [Text scorer decisions](../results/oversized-clause-dev/text-evidence-scorer.json)

### 5.3 Retrieval comparison and cross-reference audit

| Search method | Visible evidence at rank one / 614 | Visible evidence in top five / 614 | Correct / 1,037 | Contradictions caught / 95 | Gold Not Mentioned caught / 423 |
|---|---:|---:|---:|---:|---:|
| BGE | 222 | 484 | 534 (51.49%) | 34 | 221 |
| BM25 | 348 | 521 | 613 (59.11%) | 39 | 237 |
| BGE + BM25 | 312 | 520 | 574 (55.35%) | 42 | 208 |
| BGE + BM25 reranked with MiniLM | 370 | 520 | 604 (58.24%) | 49 | 213 |

Relative to BGE, BM25 gained 7.62 percentage points of accuracy (paired
95% contract-bootstrap interval +3.95 to +11.38 points). Reranking
improved plain Hybrid by 2.89 points (+0.77 to +5.11). Reranked Hybrid's
accuracy was 0.87 points below BM25 (−3.86 to +2.12), while its
contradiction recall was 10 cases higher. BM25 also retrieved 303
rank-one inputs over the DeBERTa token limit, compared with 247 for
reranked Hybrid and 195 for BGE.
[Retrieval comparison](../results/oversized-clause-dev/report.md#bm25-control-and-hybrid-reranking)

The reference audit found a unique target in another clause for 47 of
164 cases with a numbered reference-like phrase in the first BGE result.
Following the first resolvable reference recovered evidence in five cases
missed at rank one, including four missed by the top three. Checking all
direct references recovered nine rank-one misses. All recoveries were
Entailment cases. Verdict changes and token visibility were outside the
scope of this audit.
[Cross-reference audit](../results/oversized-clause-dev/report.md#contract-cross-reference-audit)

### 5.4 Document-level DeBERTa

The document model uses overlapping token windows, an evidence head
supervised by annotated spans, and pooled window representations for a
contract-level verdict. Its epoch-three checkpoint was selected by
development macro F1. The model received only contract text and a
hypothesis at inference.

| Split | Correct | Accuracy | Macro F1 | Contradiction | Entailment | Not Mentioned |
|---|---:|---:|---:|---:|---:|---:|
| Development | 893/1,037 | 86.11% | 0.8113 | 71/95 | 473/519 | 349/423 |
| Primary held-out | 1,560/1,751 | 89.09% | 0.8503 | 138/189 | 759/820 | 663/742 |

On development, the gain over the five-clause logistic combiner was
25.27 percentage points with a paired contract-bootstrap 95% interval of
+21.70 to +28.93 points. On the 103-contract primary held-out cohort, the
accuracy interval was 87.61–90.52% and there were zero LLM calls.
The model missed 51 contradictions and falsely reported a mention on
79 cases. The top-scored evidence window contained at least one complete
annotated span in 938 of 1,009 evidence-bearing cases and in 43 of the 51
missed contradictions. The matched routing comparison is in progress.
[Development comparison](../results/document-nli/comparison.json)
· [Held-out predictions](../results/held-out/document_only.json)

### 5.5 Preliminary A/B/C comparison

The matched November pilot covered 51 cases across three development
contracts using GPT-OSS 20B through Groq.

| Configuration | Correct / 51 | Logical LLM calls |
|---|---:|---:|
| A — LLM only | 33 | 51 |
| B — pretrained NLI gate | 29 | 12 |
| C — fine-tuned NLI gate | 23 | 9 |

Config C would skip 42 calls, while 24 of those accepted verdicts were
wrong. The sample is too small to rank the configurations reliably. The
full held-out comparison, latency and resource measurements, and
confirmatory uncertainty estimates are pending. [Pilot report](../results/groq-pilot-20261110/report.md)

### 5.6 Exploratory reference lookup and agentic RAG

On all 1,037 development cases, BM25 top-five retrieval contained at
least one complete annotated passage in 548 of the 614 evidence-bearing
cases. Prioritizing the top clause's resolved reference targets under the
same 6,000-byte excerpt cap yielded 546. Both covered every annotated
span in 492 cases. The agent prototype can request a reference lookup,
neighboring clauses, or a new search. A synthetic smoke case confirmed
that it can request a second search and cite an exception found by
that search. The development pilot measures actual excerpt disclosure
and verdict quality. Offline replay of the matched fixed lookup policy
raised evidence coverage from 333 to 451 of 614 annotated development
pairs. At a separate 2,000-byte excerpt budget, reranked hybrid retrieved
contradiction evidence in 43/95 cases versus 32/95 for BM25. At 6,000
bytes both reached 70/95. These runs used different parser and packing
settings and report evidence coverage. The live pilot completed 14
development pairs. Document-level DeBERTa, top-five RAG, deterministic
lookup, and agentic lookup got 12, 9, 7, and 6 verdicts correct,
respectively. Contradiction recall was 4/6, 4/6, 3/6, and 2/6. The agent
requested no follow-up on the eight cases selected for missing evidence
and got all eight wrong. It sent 15,523 excerpt bytes versus 70,349 for
deterministic lookup. Neither recovered a DeBERTa error. These observations
are limited to one targeted development pilot.
[Live pilot](../results/reference-rag-dev/hard_live.json)
[Hard-case audit](../results/reference-rag-dev/hard_audit.json)
· [Excerpt budget results](../results/reference-rag-dev/context_budget.json)
[Retrieval audit](../results/reference-rag-dev/retrieval.json)
· [Live tool trace](../results/reference-rag-dev/live_tool_smoke.json)

## 6. Discussion

A confident prediction from one retrieved clause can become an erroneous
verdict about the whole contract. This matters especially for Not Mentioned:
an irrelevant clause may contain no evidence while another part of the
contract settles the question. Additional retrieval made more annotated
evidence visible, but the confidence-only rescue rule also replaced
correct Not Mentioned verdicts. Better evidence selection and a reliable
way to combine evidence across clauses are both needed.

With the classifier fixed, BM25 improved overall accuracy and recovered
more gold Not Mentioned cases than the other tested rankings. Reranked
Hybrid led on visible rank-one evidence and contradiction recall.
Retrieval quality should therefore be assessed alongside the verdicts
for each class. The multi-clause comparison will test whether these
rankings also improve decisions when the model uses several clauses.

The prepared-input result and the exploratory calibration check suggest
that confidence should be measured on the inputs the gate actually sees.
The 51-case pilot demonstrates the cost of accepting high-confidence
errors in that sample. Whether a revised gate preserves accuracy while
saving calls remains a held-out question. A final comparison should track
contradiction recall and false Not Mentioned verdicts alongside overall
accuracy and call rate.

## 7. Limitations

- ContractNLI uses 17 fixed hypotheses on NDAs. Other contract types and novel hypotheses remain untested.
- Evidence visibility measures complete annotated spans. A visible span may be insufficient by itself, and a retrieved clause without an exact annotated match may still contain relevant text.
- The annotated-input rerun selects errors and uses information unavailable during deployment. It estimates how often those errors change under prepared evidence.
- Several variants were compared on development data. The reported intervals describe those paired development comparisons. The 20 explored test contracts and prior test-set retrieval diagnostic must be disclosed in the final protocol.
- Benchmark input is supplied contract text, whereas the app receives PDFs. The small A/B/C pilot uses hosted Groq. It provides no evidence for fully local deployment or production latency.

## 8. Conclusion (development findings)

The current rank-one gate accepts many incorrect verdicts with high
confidence. Annotated evidence resolves many of those errors. On
development data, BM25 gave the highest rank-one accuracy and reranked
Hybrid caught the most contradictions. Multi-clause verdict evaluation
and the held-out accuracy and LLM-call comparison remain pending.

---

**Draft checklist before submission.** Verify the original training input
construction and related-work positioning. Freeze the evaluation protocol,
including prompts and contract grouping, before the final held-out run.
Replace provisional claims only when the matched results are available.
