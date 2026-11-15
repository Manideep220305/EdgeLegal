# EdgeLegal research handoff

Updated **14 November 2026**. This is the current research status and decision log. The [Project Guide](EdgeLegal_Project_Guide.md) records the original scope and intended A/B/C comparison. The [paper draft](../researchpaper_claude/paper_draft.md) is a working manuscript. Use code and saved run artifacts to verify claims.

## Current results at a glance

The retrieval comparisons use the same **1,037 ContractNLI development
pairs from 61 contracts**. The default parser and fine-tuned DeBERTa
checkpoint are fixed. Only the first clause from each ranking is used
for the verdict.

| Retrieval method | Correct / 1,037 | Contradictions caught / 95 | Correct Not Mentioned / 423 |
|---|---:|---:|---:|
| BGE baseline | 534 (51.49%) | 34 | 221 |
| BM25 | **613 (59.11%)** | 39 | **237** |
| BGE + BM25 (Hybrid) | 574 (55.35%) | 42 | 208 |
| Hybrid + MiniLM reranker | 604 (58.24%) | **49** | 213 |

The earlier **55.26% (573/1,037)** used a learned text filter to decide
when to override a BGE rank-one verdict. The saved **97.01%** used prepared
premises that include annotated evidence for Entailment and Contradiction.
The [experiment report](../results/oversized-clause-dev/report.md) gives the
input conditions, metrics, and reproduction steps for each run.

**Decision.** The verified BGE five-clause classifier reached 60.85% and
missed the Not Mentioned gate. BM25 is available as an opt-in live mode.
The default remains BGE with one-clause DeBERTa and the 0.95 gate.
Document-level DeBERTa reached **893/1,037 (86.11%)** on development
and **1,560/1,751 (89.09%)** on the primary held-out cohort without
LLM calls. BM25 with reranking and the matched held-out routing
comparison remain pending. The cross-reference
audit is complete. A model-controlled lookup agent is available as an
experimental opt-in. Its development comparison is below.

## Project brief for the incoming engineer

EdgeLegal is a major-project prototype and research study for reviewing
non-disclosure agreements. It checks a statement, such as whether a party
can share confidential information with a third party, against the contract.
The expected verdict is **Entailment**, **Contradiction**, or **Not Mentioned**.
The paper studies how retrieved evidence affects a small classifier's
reliability and whether an LLM can be reserved for cases that need it.
The task is a statement against one contract. Within-contract clause
conflicts and conflicts across documents are separate future work.

The app accepts a PDF, extracts text, splits it into clauses, embeds the
clauses with BGE, and indexes them in ChromaDB. LangGraph runs retrieval,
triage, and escalation. The fine-tuned DeBERTa scores the first retrieved
clause against the hypothesis. The live gate accepts its prediction at
confidence ≥ 0.95 and calls Groq below that threshold. The live LLM can
receive several clauses. The benchmark runner uses ContractNLI's supplied
text and shares the core workflow. The offline parser and top-three
evaluators bypass the LLM to isolate classifier behavior.

The dataset has **607 NDAs and 17 fixed hypotheses**. Labels describe
the whole contract, even though DeBERTa currently sees only one clause.
Entailment and Contradiction have annotated spans. A single unrelated
clause cannot establish that the whole contract is silent. That distinction
explains why retrieving extra clauses improved contradiction recall in our
experiments but sometimes overrode correct Not Mentioned verdicts.

**Repository and file map.** The local work is in `EdgeLegal_CodeBase`.
`backend/app/agent/` contains the live graph and routing nodes.
`backend/app/triage_model/` contains model loading, scoring, and the
local checkpoint directory. `backend/app/parsing/` and
`backend/app/vectorstore/` contain segmentation and retrieval.
`frontend/` is the demo. `backend/app/evaluate_*.py` contains
offline experiments and `results/` contains their artifacts.
The personal GitHub repo is [Manideep220305/EdgeLegal](https://github.com/Manideep220305/EdgeLegal).
It began as a snapshot. The shared folders contain later local changes and
local-only data and weights. Use those folders for this assignment and
check the working tree before editing.

**Assignment boundary.** The immediate job is the offline classifier
experiment in the [implementation handoff](#implementation-handoff-multi-clause-contract-classifier).
Complete the training and development comparison before changing the
live app or running the final A/B/C comparison. The original parser stays
the baseline. The oversized-clause parser was tested as an opt-in variant.
The top-three and text-filter results also came from offline variants.

**Research status.** The paper already has development evidence that the
classifier often changes a wrong answer when given annotated text. The
current research contribution is the paired analysis of evidence retrieval,
visibility after truncation, and classifier errors. The small Groq pilot
checks feasibility and exposes failure modes. A final claim about saving
LLM calls while preserving accuracy still requires the matched held-out
evaluation. Keep the paper, saved results, and app behavior distinct when
reporting progress.

## Current system

- The live app and research runner share the LangGraph workflow. BGE and ChromaDB rank clauses. DeBERTa-v3-base judges the highest-ranked clause against the hypothesis. The current gate accepts at confidence **≥ 0.95** and escalates lower-confidence cases. The live escalation provider is Groq.
- Config A sends every case to the LLM. Config B uses the off-the-shelf `cross-encoder/nli-deberta-v3-base` as the gate. Config C uses that backbone fine-tuned on ContractNLI. The original guide proposed a ContractNLI-specific existing checkpoint for B. The implemented B uses general NLI weights.
- The original structural parser is the default in the app and standard experiment runner. The new `split_oversized=True` option splits clauses above 2,000 characters and is exercised by the [paired evaluator](../backend/app/evaluate_oversized_clauses.py). Existing indexes have not been migrated. Top-three classification, an evidence-aware gate, and retrieval-style retraining remain proposals.

## What was run and decided

| Run | What it establishes |
|---|---|
| [Saved prepared-input development evaluation](../backend/app/triage_model/dev_eval_results.json) | 97.0% classifier accuracy on 1,037 cases. Entailment and Contradiction use annotated evidence. The current evaluator uses document text for Not Mentioned. This is a separate input condition from retrieval. Exact original Kaggle preprocessing still needs provenance verification. |
| September pilots and diagnostics | The matched two-contract test smoke run scored 26/34 for [A](../results/A_no_triage_20260927T140411+0000.json), 18/34 for [B](../results/B_pretrained_triage_20260927T140547+0000.json), and 26/34 for [C](../results/C_finetuned_triage_20260927T140710+0000.json). The [20-contract C diagnostic](../results/C_finetuned_triage_20260927T143308+0000.json) scored 181/340 with stubbed escalation according to the run notes. Grouped recalibration reduced ECE from 0.440 to 0.040 with Platt scaling, with no scores above 0.7 after calibration. The later matched 20-contract Groq attempt stopped at the quota cap. |
| [November Groq A/B/C pilot](../results/groq-pilot-20261110/report.md) | Matched on 51 cases from three **development** contracts. A got 33 correct, B 29, and C 23. C accepted 42 cases automatically and got 24 of those wrong. A single Groq response per case was reused for escalations under an identical prompt without the classifier hint. This pilot used a separate harness. The standard runner still passes the classifier hint in B/C. |
| [Paired parser experiment](../results/oversized-clause-dev/report.md) | Same 1,037 development cases in both arms, top-one retrieval, same fine-tuned model, zero Groq calls. Splitting reduced inputs over 512 tokens from 195 to 1 and improved classifier accuracy from 51.5% to 55.1%. Top-one annotated evidence hits fell from 256 to 218 of 614. The change failed the agreed requirement to improve evidence hits, so the original parser stays the default. The split result is classifier-only. |

The older **73.1%** diagnostic was inclusion anywhere in a gap-filtered set, conditional on the evidence being mappable. It is a different metric from top-one evidence hits and cannot be used as an accuracy ceiling. Also, a long top-ranked clause can contain an annotated span outside DeBERTa's 512-token input. The parser experiment measured containment and truncation separately. It did not measure whether each evidence span survived truncation.

## Why the sequence changed

The original plan moved from training to a full matched A/B/C comparison. The pilots showed that the 0.95 gate can accept many wrong verdicts, and a full Groq run is constrained by token quota. We paused the full comparison to find where the classifier loses accuracy. The parser experiment was a controlled check of one observed problem, an entire 14,464-character contract being indexed as one clause. Its mixed result does not establish an improved production pipeline.

**Evidence diagnostic complete (2026-11-10).** The [saved results](../results/oversized-clause-dev/evidence-visibility.json) reuse all 1,037 pairs per arm. Of the 614 cases with annotations, at least one full annotated span survived tokenization in 222 baseline cases and 218 split-parser cases. The baseline had 33 evidence hits fully cut by truncation and one partially cut. Annotated inputs corrected 278 of 301 baseline Entailment/Contradiction errors and 208 of 228 split-parser errors. In the baseline, they corrected 202 of 215 errors whose evidence mapped to a clause outside rank one, and 45 of 55 errors where at least one span was already visible. Supplying the annotated spans can change the verdict even when a retrieved clause contains one full span. This annotation-assisted diagnostic does not estimate live accuracy or show that top-three retrieval will recover the evidence.

The annotated-input rerun covered 321 distinct errors across both arms with the same checkpoint. No new retrieval run or Groq calls were needed. Not Mentioned has no positive annotated span and was excluded from that rerun. On the 423 gold Not Mentioned cases, the baseline made 202 errors and the split parser made 238. That is separate from the reduction in false Not Mentioned verdicts on actual Entailment or Contradiction cases.

**Top-three development experiment complete (2026-11-10).** The [offline evaluator](../backend/app/evaluate_top_three.py) ran on all 1,037 cases with the default parser, the same checkpoint, and zero Groq calls. Rank one matched the saved baseline. Visible annotated evidence rose from 222 to 391 of 614 cases. At the primary 0.95 override threshold, accuracy rose from 534/1,037 (51.5%) to 557/1,037 (53.7%). The rule fixed 96 errors, introduced 73, and changed 16 wrong verdicts to a different wrong verdict. Contradiction recall rose from 34/95 to 47/95, while gold Not Mentioned recall fell from 221/423 to 148/423. The paired contract-bootstrap interval for the accuracy change was −0.87 to +5.30 percentage points. See the [comparison report](../results/oversized-clause-dev/report.md#top-three-development-comparison) and [per-case data](../results/oversized-clause-dev/top-three.json).

**Stricter override experiment complete (2026-11-10).** On the same 1,037 cases, requiring both extra clauses to agree at confidence ≥ 0.95 cut new errors from 73 to 6. It scored 551/1,037 and found 39/95 contradictions, versus 557/1,037 and 47/95 for the unrestricted rescue rule. Keeping the selected alternate within the existing 15% retrieval-distance band cut new errors to 48 and scored 555/1,037 with 44/95 contradictions. Both improved gold Not Mentioned recall relative to unrestricted rescue, but neither improved its overall accuracy. See the [report](../results/oversized-clause-dev/report.md#stricter-override-comparison) and [per-case decisions](../results/oversized-clause-dev/strict-overrides.json). The experiment replayed cached scores with no model reruns or Groq calls.

**Evidence verifier complete (2026-11-10).** A logistic regression model used retrieval distance, clause rank and agreement, DeBERTa confidence, token length, and lexical overlap. Training labels required a visible annotated span and a correct candidate verdict. Neither annotation visibility nor the gold verdict was an input. Five folds held out whole contracts within the development split. At a fixed 0.50 cutoff, the verifier got 553/1,037 correct versus 551/1,037 for agreement. Both introduced six errors, but the verifier caught 36/95 contradictions versus 39/95 for agreement. The verifier failed the contradiction guardrail. See the [report](../results/oversized-clause-dev/report.md#evidence-verifier-comparison) and [folds and per-case decisions](../results/oversized-clause-dev/evidence-verifier.json).

**Decision.** Keep top-one scoring in production and agreement as the conservative research baseline. The verifier reused cached DeBERTa scores and required no Groq calls. Full A/B/C remains paused. A stronger follow-up would train a text-based evidence scorer on the ContractNLI training split and compare it on development contracts, with contradiction recall and false overrides as guardrails.

**Text-based scorer complete (2026-11-10).** Fit a hypothesis-conditioned TF-IDF logistic scorer on 24,586 clause pairs from the 423 ContractNLI training contracts. The training and development sets shared no IDs or exact normalized contract text. At a fixed 0.50 cutoff on the saved development cases, the scorer got 573/1,037 correct versus 551/1,037 for agreement, fixed 40 errors versus 23, and introduced one new error versus six. The paired 95% contract-bootstrap interval for the accuracy gain was +0.58 to +3.86 percentage points. It caught 38/95 contradictions versus 39/95 for agreement, so it missed the contradiction guardrail. See the [comparison](../results/oversized-clause-dev/report.md#text-based-evidence-scorer) and [per-case decisions](../results/oversized-clause-dev/text-evidence-scorer.json). No Groq calls or DeBERTa prediction reruns were made.

**Cross-reference audit complete (2026-11-13).** On the same development cases, 47 of 164 rank-one cases with reference-like text pointed to at least one unique numbered heading in another clause. One lookup recovered five missed evidence cases, four missed by the top three. Looking up all direct matches recovered nine rank-one misses. All recoveries were Entailment. The audit measured raw evidence containment and used zero model or Groq calls. See the [audit report](../results/oversized-clause-dev/report.md#contract-cross-reference-audit).

**Hybrid retrieval comparison complete (2026-11-13).** BGE plus per-contract BM25 with reciprocal rank fusion was tested on all 1,037 development cases, with the same default parser and rank-one DeBERTa scoring and zero Groq calls. Evidence visible to DeBERTa rose from 222/614 to 312/614 at rank one and from 484/614 to 520/614 in the top five. Correct verdicts rose from 534/1,037 (51.49%) to 574/1,037 (55.35%), a gain of 3.86 percentage points with a paired 95% contract-bootstrap interval of +1.06 to +6.65 points. Contradictions caught rose from 34/95 to 42/95, while correct gold Not Mentioned verdicts fell from 221/423 to 208/423 and truncation increased from 195 to 250 cases. Carry hybrid retrieval forward as an offline candidate. Use the original BGE retrieval for the first multi-clause handoff run, then compare both retrievers with the same multi-clause scoring method. The live app continues to use BGE and the fixed confidence gate. See the [hybrid comparison](../results/oversized-clause-dev/report.md#hybrid-bge-and-bm25-retrieval) and [per-case results](../results/oversized-clause-dev/hybrid-retrieval.json).

**BM25 control and reranking complete (2026-11-13).** With rank-one DeBERTa fixed on all 1,037 development cases, BM25 alone got 613 correct (59.11%), plain Hybrid got 574 (55.35%), and reranking the Hybrid top five with frozen MS MARCO MiniLM got 604 (58.24%). BM25 got 237/423 gold Not Mentioned cases and 39/95 contradictions correct. Reranked Hybrid caught 49/95 contradictions and had visible evidence in rank one for 370/614 annotated cases versus 348/614 for BM25. Reranked Hybrid used 247 over-limit DeBERTa inputs versus 303 for BM25. Zero Groq calls were made. Carry BM25 and reranked Hybrid forward as candidates for the multi-clause comparison, retaining the BGE handoff run as the original control. See the [experiment report](../results/oversized-clause-dev/report.md#bm25-control-and-hybrid-reranking) and [per-case results](../results/oversized-clause-dev/retrieval-reranking.json).

## Project completion priorities

**Completion work started (2026-11-13).** The teammate's five-clause
result was verified against the supplied archive. It achieved 631/1,037
correct, 40/95 contradictions, and 203/423 gold Not Mentioned. The
[cached-score margin experiment](../results/clause-combiner/mention-margin.json)
reproduced every saved baseline verdict. Five contract-grouped training
folds selected a probability margin of 0.003. On development it reached
632 correct, 40 contradictions, and 205 Not Mentioned. The margin missed
the 220-case requirement and was rejected. The experiment used no
DeBERTa inference or Groq calls.

The app has an opt-in BM25 mode via `EDGELEGAL_RETRIEVAL_MODE=bm25`
in the backend process environment. The mode uses the saved experiment's
BM25 formula and document-order tie break. An escalation failure now
returns a retry message, and an unmatched excerpt is omitted from the
clickable citations. The BGE workflow remains the default pending
held-out evaluation.

The frontend production build passed with a bundled PDF worker. A
Chromium check on the bundled demo NDA verified upload, a BM25 and
DeBERTa verdict, clicking the source-page citation, and reopening the
saved report. An API smoke test separately verified a page-two citation
on a generated two-page NDA, rejection of unreadable files and missing
indexes, and a simulated provider failure that left history unchanged.
The first BM25 analysis in that test took 6.97 seconds including model
load. It was a single cold measurement. A fresh local clone with the
current app changes applied also passed `npm ci`, the production
build, and a clean backend dependency install. Its demo NDA produced an
Entailment verdict with a page-one citation and a saved report that
could be reopened. The API took 6.53 seconds for the first analysis,
including checkpoint load. The clone used the documented checkpoint
path and cached model assets.

Exact passage matching against the source PDF now supplies a highlight
when there is an unambiguous rectangle. A separate two-page PDF check
verified that a passage found only on page two corrects the page number
and returns the highlight coordinates.

The [GPU smoke run](https://www.kaggle.com/code/saimanideepkatta/edgelegal-document-nli-smoke)
verified the checkpoint and split hashes, updated both the DeBERTa
encoder and evidence head, and loaded a saved resume checkpoint on
a Tesla T4. The [full GPU run](https://www.kaggle.com/code/saimanideepkatta/edgelegal-document-nli-train)
started at about 8:25 pm IST and completed all three epochs on Kaggle.
Development accuracy was 83.32%, 85.92%, and 86.11% across the three
epochs. Epoch three had the highest macro F1 (0.8113) and got 71/95
Contradiction, 473/519 Entailment, and 349/423 Not Mentioned cases
correct. It passed all three development gates. Compared with the
five-clause logistic baseline, it fixed 308 errors and introduced 46,
for a gain of 25.27 percentage points (paired 95% contract-bootstrap
interval +21.70 to +28.93 points). The top evidence window contained
a complete annotated span in 568/614 cases. See the [training history](../results/document-nli/history.json)
and [paired comparison](../results/document-nli/comparison.json).

The document-model opt-in path passed an end-to-end browser check with
the trained checkpoint. The bundled NDA returned Entailment and cited
the confidentiality clause on page one. The citation opened the correct
PDF page from the review and from saved history. The first analysis took
12.78 seconds including checkpoint load. The app selects a clause by BM25
inside the model's highest-scoring evidence window for display only. A
follow-up API check confirmed that the wrapped clause now returns a PDF
highlight box. The same check covered corrupt PDFs, image-only uploads,
and missing indexes. A simulated missing provider returned 503 with retry
guidance and created no history entry. Repeating the review with local
routing produced one saved verdict. The Delaware governing-law hypothesis
returned Entailment with a highlighted citation on page two of the
three-page demo. A separate checkout with the current app changes and
locally supplied model weights passed the frontend production build,
Python dependency check, the demo API checks, and a Chromium walkthrough.
That walkthrough simulated a 503 in the browser, repeated the statement
check, navigated to the highlighted passage on page two, and verified
one saved result.

Two original ContractNLI development PDFs exposed a local tokenizer issue.
The older Windows tokenizer kept only two windows for contracts that
required five and eleven, respectively. The same two wrong verdicts
occurred on the original benchmark text with that tokenizer. The pinned
`transformers==5.20.2` and `tokenizers==0.23.3` restored full coverage
and the expected verdicts on both PDFs. Live document inference now
checks for omitted text before returning a verdict. With the pins, three
statement checks across the two PDFs produced the expected verdicts and
highlighted supporting text on the original pages. Those checks used
local routing to exercise document inference and citation display.

The [held-out runner](../backend/app/evaluate_held_out.py)
checks the development gates before scoring the test cohort. It freezes
the candidate's confidence threshold on development scores, reuses one
LLM response per prompt, and reports the fixed-budget routing contrast.
The metrics and promotion gates were checked on synthetic predictions.

**Document-only held-out evaluation (2026-11-14).** The frozen epoch
three checkpoint got 1,560/1,751 cases correct across 103 contracts
(89.09%, 95% contract-bootstrap interval 87.61–90.52%). Macro F1
was 0.8503, with 138/189 contradictions, 759/820 entailments, and
663/742 gold Not Mentioned cases correct. The run made zero Groq calls.
The highest-scoring evidence window contained at least one full annotated
span in 938 of 1,009 evidence-bearing cases. Among the 51 missed
contradictions, 43 had a full annotated span in that window. This measures
the evidence window used to select a citation. It does not measure
whether the shorter citation itself matches the annotation.
The separately frozen confidence threshold would route 1,394 of the
same cases to the LLM. That is a routing count before the matched
provider comparison. See [per-case predictions](../results/held-out/document_only.json)
and the [frozen manifest](../results/held-out/manifest.json).

**Matched provider comparison (paused at Groq quota).** All 1,751 cases and 1,698
unique LLM prompts are prepared under the frozen inputs. The live Groq run
uses `openai/gpt-oss-20b` and caches responses for resumption. The candidate
routes 1,394 cases, B routes 293, C routes 244, and A routes all 1,751.
The candidate got 354 of the 357 cases it would handle locally correct.
These counts measure decisions to escalate per case. The shared cache
reduces A from 1,751 cases to 1,698 unique requests and the candidate
from 1,394 cases to 1,354 unique requests. See the
[routing manifest](../results/held-out-routing/manifest.json) and
[response cache](../results/held-out-routing/responses.jsonl). The final
matched accuracy, confidence intervals, and promotion decision are pending.
On 14 November the provider returned 132 valid unique verdicts covering
147 cases before rejecting the next request at the daily quota. The cache records 199,650
reported tokens across successful responses. The attempt log also records
two transient bad requests and the quota rejection.
Groq publishes an 8,000-token per minute and 200,000-token per day free-plan
limit for this model. Tokenizing all frozen prompts and adding the observed
template overhead gives about 2.09 million input tokens. At the average
completion length from the first 41 responses, the full run would use
about 2.89 million tokens before provider-side caching, or roughly 15 daily
quota windows. Twenty-one estimated inputs exceed 8,000 tokens, with a
maximum of 8,183, so some requests may also hit the minute limit on their
own. The same output directory can be reused after quota resets with the
frozen inputs.

Resume from the repository root with the same provider and output directory.

```powershell
python -m backend.app.evaluate_held_out `
  --checkpoint backend/app/triage_model/checkpoints/current/best_model.pt `
  --candidate results/document-nli/best_document.pt `
  --development results/document-nli/dev_predictions.json `
  --provider-model openai/gpt-oss-20b `
  --request-interval 16 `
  --out results/held-out-routing
```

1. **Make PDF review reliable.** The live app rejects image-only, empty,
   and unreadable PDFs with a specific upload error. Analysis returns an
   error for older uploads with no indexed text. Next, verify upload,
   verdict, PDF citation navigation, and saved history on representative
   NDAs, including provider failures.
2. **Bring better retrieval into the app.** Evaluate reranking the BM25
   top five. Choose a candidate using overall accuracy, contradiction
   recall, and gold Not Mentioned recall. Integrate the candidate and
   check source pages and response time on real uploads.
3. **Improve decisions and escalation.** The
   [multi-clause classifier handoff](#implementation-handoff-multi-clause-contract-classifier)
   starts with BGE to isolate the effect of combining clause scores.
   That work can run alongside app work. Compare routing signals at the
   same LLM call budget and inspect confidently wrong verdicts before
   changing the live gate.
4. **Verify the finished system.** Check that a fresh clone can run the
   demo with the documented dependencies, checkpoint, and provider setup.
   Freeze the research candidate before the held-out comparison.

The [held-out protocol](EdgeLegal_Project_Guide.md#64-proposed-held-out-protocol-2026-11-10) defines the 103-contract primary cohort, identical LLM prompts, a matched call budget, and criteria for preserving accuracy and contradiction recall. The 20 explored test contracts and the earlier full-test retrieval inspection are documented in the protocol.

Update this page whenever an experiment completes or a decision changes. Link the artifacts and distinguish a planned experiment from a shipped change. The older implementation summary, architecture sketches, and Review 3 notes are historical snapshots.

## Reference lookup and agentic RAG

The [shared review code](../backend/app/agent/reference_review.py) starts
from BM25 results. The model can follow a numbered reference, read the
neighboring clauses, search again with its own terms, or finish. The
experiment caps each case at 6,000 UTF-8 bytes of excerpts, counting
repeated sends, and up to three calls. The fixed controls use one call
with top-five retrieval or top-five retrieval with a direct reference
lookup prioritized. Quoted citations must occur verbatim in a supplied
clause. Citation overlap with annotations is a proxy for support.

The [retrieval audit](../results/reference-rag-dev/retrieval.json) ran on
all 1,037 development cases with no LLM calls. Among 614 cases with
evidence annotations, top five retrieved at least one complete span in
548 cases, and fixed reference lookup did so in 546. Both found every
span in 492 cases. Only 41 cases had a resolvable reference in the
top result. The fixed rule displaced the only matching evidence in two
cases and recovered none under these settings.

The [pilot runner](../backend/app/evaluate_agentic_rag.py) compares the
three RAG variants on a small, label-stratified sample with a distinct
contract per case. It also reports the frozen document model on the same
pairs. All Groq requests in this pilot use public ContractNLI examples.
The pilot checks behavior and feasibility. A larger frozen comparison
and human citation review are needed for a paper claim.

Groq's daily quota stopped both the initial run and a cached retry.
On the nine completed pairs out of eleven planned, the [pilot results](../results/reference-rag-dev/live.json)
showed 4/9 correct for top five, 5/9 for fixed reference lookup, 5/9 for
the agent, and 9/9 for document-level DeBERTa. The agent matched 1/3
contradictions versus 3/3 for fixed lookup. It sent 8,502 excerpt bytes
versus 37,191 for fixed lookup, with zero requests for additional evidence
on those nine real cases. The lower transmission in this sample reflects
starting from a single passage.

A [live tool smoke test](../results/reference-rag-dev/live_tool_smoke.json)
used a synthetic NDA exception. Groq requested a new search, retrieved
the exception, and cited both the main rule and the exception in an
Entailment verdict. A scripted API check verified a page-two citation
and a saved report. The frontend production build passed.

To reproduce from the repository root, run
`python -m backend.app.evaluate_agentic_rag` for the offline audit or
`python -m backend.app.evaluate_agentic_rag --live` for the Groq pilot.
The pilot caches successful responses and resumes with the same directory.
Each arm has the same maximum excerpt and call budgets. The fixed arms
send several passages in one call. The agent starts with one passage
and spends additional calls and text only when it requests a lookup.

The app experiment is enabled by setting `EDGELEGAL_AGENTIC_RAG=1` on
the backend. In this mode review uses Groq directly for the agent's
tool choices and verdict. The excerpt budget measures contract text
shared with the provider. The hypothesis and prompt instructions also
go to the provider. Use public contracts for the demo. This setting
selects an experimental route for all reviews in that backend process.
The existing production route remains the default while the comparison
is exploratory.

**Hard-case audit.** The [per-case audit](../results/reference-rag-dev/hard_audit.json)
checks complete annotated spans missing from the starting passage. Under
the same excerpt budget, 91 cases could recover a span through a numbered
reference or neighboring clause and another 122 through lower-ranked BM25
results. The groups include 36 contradictions in total. Direct numbered
references recovered a missing span in just six cases, all Entailment.
Whole-contract DeBERTa missed 30 of the 213 lookup candidates. These
counts measure recoverable evidence. They do not establish that every
missing annotation is required to decide the case.

The audit selects 14 distinct contracts, eight lookup cases and six
controls. The structural group includes two direct-reference examples.
In the matched comparison, the fixed policy and agent both start with
the same clause and can use up to three calls and 6,000 excerpt bytes.
The fixed policy tries references, neighboring clauses, and the original
BM25 query in that order as space allows. The agent can choose the tool
and write its own search query. Ordinary top-five RAG is the one-call
control. Scripted checks verified the starting clauses and budgets on all
14 cases. Run `python -m backend.app.evaluate_agentic_rag --hard` for the
offline audit and add `--live` to resume the public-data Groq comparison.
Groq's daily quota interrupted the initial run before the first matched
case completed. A resumed run completed [all 14 cases](../results/reference-rag-dev/hard_live.json).

| Method | Correct / 14 | Contradictions / 6 | Unresolved | Logical calls | Excerpt bytes |
|---|---:|---:|---:|---:|---:|
| Document-level DeBERTa | 12 | 4 | 0 | 0 | 0 |
| Top-five RAG | 9 | 4 | 3 | 14 | 61,390 |
| Matched fixed lookup | 7 | 3 | 2 | 34 | 70,349 |
| Agentic lookup | 6 | 2 | 6 | 14 | 15,523 |

None of the RAG methods corrected either DeBERTa error. The agent made no
follow-up requests and scored 0/8 on cases selected for missing evidence
and 6/6 on controls. Its lower disclosure reflects the smaller starting
context. Top-five and fixed lookup returned quotes overlapping annotated
evidence in eight and seven cases, respectively. The agent did so in four.
Spot checks confirmed supporting quotes on the nontechnical-information
contradictions and on the compelled-disclosure notice. An exact quote can
still be irrelevant to the verdict. These are exploratory development
results. Keep the agent opt-in and use DeBERTa as the accuracy reference.
The resumed runner handles Groq JSON-validation failures as unresolved
responses and preserves the prompt.
The pilot now reads Groq's remaining-token and remaining-request headers,
waits when it estimates the next call exceeds the available budget, and
saves Groq's Retry-After deadline for subsequent launches. Daily token
headroom is not provided by those response headers. The default runner
uses one key. For verified organization assignments, set
`EDGELEGAL_GROQ_ORGANIZATIONS` to a comma-separated list of labels in
distinct key order. For example, `alpha,alpha,beta` groups the first two
keys together and uses the third key for a separate quota. The runner
uses one key per group and persists cooldowns per group. Keep a group's
label stable across restarts. The same model, prompts and cache are used
across groups. The model-list API authenticated all 14 configured keys
and did not report their organizations. Keys 9 and 13 accepted a small
request during a later check. That does not establish separate quotas.
The exact grouping remains unverified, and the pilot has used one key at
a time.

**Offline fixed-policy replay.** The audit now replays the fixed policy
on all 1,037 cases with zero Groq calls. It counts passages that would
actually be sent across the simulated calls, including repeated text
in the byte budget. Across 614 annotated cases, complete evidence-span
coverage rose from 333 to 451 cases with at least one span. The policy
added evidence in 141 cases, including 22 where DeBERTa gave the wrong
verdict. Among 36 contradiction cases selected for having recoverable
evidence, coverage rose from 2 to 32 cases. This is retrieval coverage.
Verdict accuracy and whether the agent improves on this control require
the live comparison. Across all cases the replay used 2,463 simulated
calls and 4,968,117 excerpt bytes. All cases stayed within the caps.

**RAG excerpt-budget comparison.** The [offline replay](../results/reference-rag-dev/context_budget.json)
compares the saved BM25, BGE, hybrid, and MiniLM-reranked hybrid top-five
rankings on all 1,037 development cases. Each method packs whole clauses
in rank order under the same UTF-8 byte cap, skipping clauses that do
not fit. At 2,000 bytes, reranked hybrid retrieved a complete evidence
span in 305/614 annotated cases versus 288/614 for BM25. On contradictions,
the counts were 43/95 versus 32/95. Both used about 1.59 million excerpt
bytes in total. At 6,000 bytes, BM25 retrieved evidence in 474/614
versus 467/614 for reranked hybrid, and both reached 70/95 contradictions.
These are evidence-coverage results from saved rankings using the original
parser. The replay made zero Groq calls and does not establish verdict
accuracy. The 2,000-byte result makes reranked hybrid a candidate for
the smaller-context RAG comparison.

Reproduce with `python -m backend.app.evaluate_rag_context_budget`.

## Document model training and verification

**Evidence-loss ablation (14 November, stopped).** Submitted a separate private
Kaggle GPU run
using the archived trainer and the same initial checkpoint, split hashes,
seed, windows, and three-epoch budget. The direct evidence loss weight is
zero. The evidence head still weights the windows and can learn indirectly
from the verdict loss. This tests the value of annotated evidence supervision.
The [paired evaluator](../backend/app/evaluate_document_development.py)
checks the run settings and compares the two arms on all 1,037 development
pairs. The published baseline and required-lookup RAG test follow in order.
Kaggle reported the job queued when the user requested a stop. The
separate notebook was removed from Kaggle while queued. The
[archived launcher](../kaggle_training/document_nli_no_evidence_kaggle.py)
remains in the repository. The comparison harness passed a
self-comparison check. No ablation results were produced. The two
following experiments are paused.

**Next two experiments.** For the published baseline use Stanford's
[Span NLI BERT implementation](https://github.com/stanfordnlp/contract-nli-bert)
at commit `058c56fd62d56897bb4fcfbf1be71b17aee3a79c`. Its documented
workflow trains from the upstream config. If no official weights are
available, train the baseline on a GPU and record it as a reproduction.
Compare against the same 103-contract evaluation cohort. Then run the
required-lookup RAG control against ordinary RAG at an equal excerpt
budget, counting repeated transmission, verdicts, and contradiction
recall. Neither stage has been run.

**Model and data.** The full Kaggle run fine-tuned one document-level
DeBERTa-v3-base model from EdgeLegal's existing fine-tuned `best_model.pt`.
The backbone is `cross-encoder/nli-deberta-v3-base`. It used the public
ContractNLI training split (423 NDAs and 7,191 contract–statement pairs).
The development split (61 NDAs and 1,037 pairs) selected the best epoch
by macro F1, with contradiction recall as the tie-breaker. A separate GPU
job evaluated the selected weights on 103 contracts and 1,751 pairs from
the official test split. The other 20 contracts had been used in an
earlier diagnostic. No test pairs were used to update weights in this run.

**Training method.** Each example paired the statement with overlapping
sections of the contract. The 512-token input budget included the statement
and special tokens, with 96 contract tokens of overlap. The model scored
every section with a new evidence head, gave higher-scored sections more
weight, and combined their representations for one of three verdicts.
Training updated the encoder, the verdict head, and the evidence head
together. It used AdamW at a learning rate of `2e-5`, a seed of 42, one
contract–statement pair per optimizer step, and up to two windows per
encoder batch. The objective combined verdict cross-entropy with an
evidence-window loss weighted at `0.5`. Annotated evidence provided
training targets. Evaluation predicted from the text and statement alone.

**Runs.** A small smoke run exercised two training examples and three
development examples, checked that both the encoder and evidence head
updated, and loaded a resume checkpoint. One full run completed three
epochs on a Tesla T4 (14.6 GiB) in about 8 hours 26 minutes of total epoch
time. Development accuracy was 83.32%, 85.92%, and 86.11%. Epoch three was
selected. The log and per-case files reproduce 893/1,037 development and
1,560/1,751 held-out correct verdicts. BGE, BM25, and the Groq model were
used as existing components or comparison methods during this work.

| Evidence | Saved files or Kaggle run |
|---|---|
| GPU smoke test and resume | [Kaggle run](https://www.kaggle.com/code/saimanideepkatta/edgelegal-document-nli-smoke), [raw log](../results/edgelegal-document-nli-smoke.log) |
| Three-epoch GPU training | [Kaggle run](https://www.kaggle.com/code/saimanideepkatta/edgelegal-document-nli-train), [raw log](../results/edgelegal-document-nli-train.log), [epoch history](../results/document-nli/history.json) |
| Exact trainer and launch configuration | [Trainer](../backend/app/triage_model/train_document.py), [archived Kaggle launcher](../kaggle_training/document_nli_train_kaggle.py), [training manifest](../results/document-nli/manifest.json) |
| Selected model and development comparison | [Weights](../results/document-nli/best_document.pt), [development predictions](../results/document-nli/dev_predictions.json), [paired baseline comparison](../results/document-nli/comparison.json) |
| Held-out evaluation | [Kaggle run](https://www.kaggle.com/code/saimanideepkatta/edgelegal-document-nli-held-out), [raw log](../results/edgelegal-document-nli-held-out.log), [archived launcher](../kaggle_training/document_nli_held_out_kaggle.py), [predictions](../results/held-out/document_only.json), [manifest](../results/held-out/manifest.json) |
| Cached five-clause experiment and partial Groq comparison | [Mention-margin results](../results/clause-combiner/mention-margin.json), [Groq verdict cache](../results/held-out-routing/responses.jsonl), [request attempt log](../results/held-out-routing/attempts.jsonl) |
| PDF app demo | [Saved verdict and API events](../results/document-nli/app_demo_verification.json) |

The local weights file is 737,791,763 bytes and its SHA-256 is
`1fa9dc54a46cb58a1475a604e9975201eb50680abcbb8f63ca13d155ad4c5b0c`,
which matches the held-out manifest. The original input checkpoint,
trainer source, and train, development, and test split hashes were also
verified. The three Kaggle jobs reported complete through the account's
Kaggle CLI. Kaggle notebooks are private. The weights and dataset files
are excluded from Git, so a collaborator needs the corresponding Kaggle
access or a copy of those inputs. The local verification recounted
predictions and matched their labels to the dataset. A new GPU run would
independently reproduce inference from the checkpoint.

## Implementation handoff: multi-clause contract classifier

This section records the original assignment. The completed five-clause
result and mention-margin follow-up are under project completion priorities.

**Assignment.** Build and evaluate an offline classifier that turns up to
five DeBERTa clause scores into one verdict for the whole contract. The
experiment should tell us whether a learned decision rule improves accuracy
and how much of that gain comes from additional clauses. This experiment has
not been implemented. Use the existing DeBERTa checkpoint as a frozen feature
extractor. The inputs, training, and evaluation are all local.

**Why this is the next task.** The BGE rank-one classifier scores
534/1,037 (51.49%). The earlier text filter reached 573/1,037 (55.26%)
and changed 46 verdicts. The current rank-one retrieval references are
BM25 at 613/1,037 and reranked Hybrid at 604/1,037. The learned classifier
will test how much additional clauses contribute when retrieval is fixed.

### 1. Set up and verify inputs

- Work from the `EdgeLegal_CodeBase` root. Use the existing Python 3.11
  environment if it works, or create one and install
  `backend/requirements.txt`. Record Python, PyTorch, Transformers,
  Chroma, sentence-transformers, and scikit-learn versions and the device used.
- Verify `backend/data/contractnli/train.json` (423 contracts) and
  `dev.json` (61 contracts). If absent, run
  `python backend/app/data_setup.py` from the repo root. Each contract has
  the same 17 hypotheses. Verify disjoint contract IDs and check for exact
  normalized text duplicates across train and dev. Keep test data out of
  this experiment.
- Verify `backend/app/triage_model/checkpoints/current/best_model.pt` is
  the team's trained checkpoint, or supply its equivalent path. Load it
  explicitly and record its SHA-256 hash. GitHub's initial snapshot does
  not include the local dataset or checkpoint. Ask the project owner for
  the checkpoint if the shared folders lack it. A pretrained or stub
  model does not reproduce the saved rank-one result.
- Start from the [top-three evaluator](../backend/app/evaluate_top_three.py)
  for retrieval and batching, [DeBERTa model code](../backend/app/triage_model/model.py)
  for input formatting and output label IDs, and the [visibility evaluator](../backend/app/evaluate_evidence_visibility.py)
  for measuring evidence after truncation. The [saved comparison report](../results/oversized-clause-dev/report.md)
  and the linked JSON files are the reference results.

### 2. Generate a resumable clause-score cache

1. Use the default parser, the same BGE embeddings, and the same Chroma
   cosine-distance ranking as the existing offline top-three experiment.
   Change only candidate depth to five. Use up to five raw ranked clauses
   before the live 15% distance filter, as the offline top-three evaluator
   does. Rank one must match the saved baseline on development contracts.
2. Score each retrieved clause paired with its hypothesis. Use the same
   512-token pair limit, evaluation mode, and disabled gradients. Cache all
   three softmax scores in the checkpoint's label order. The existing
   `predict()` and batch scorer save the winning score only, so expose or
   save the full score vector from the logits. Verify that the vector's
   argmax reproduces the existing label mapping and prediction.
3. Save one record per split, contract ID, and hypothesis ID. Include the
   gold contract label, then each clause's rank, ID, cosine distance, text
   or source pointer, three scores, untruncated token count, and truncation
   flag. Save annotated-span visibility separately for evaluation. Include
   a run manifest with the checkpoint hash, code revision, data hashes,
   retriever settings, and package versions. Write a completed contract
   atomically and resume only if the manifest matches.
4. First run a smoke test on several development contracts. Check that
   rank-one clause IDs, text, and predicted labels match the saved
   baseline. Exercise a contract with fewer than five chunks, verify score
   vectors sum to one, and time a representative batch. Investigate any
   rank-one mismatch before the full run. Expect 7,191 training and 1,037
   development records, with at most 41,140 clause scores before reuse.
   Once cached, later training runs can read the cache directly.

### 3. Train the two comparison models

- Compare the frozen rank-one DeBERTa prediction with a learned one-clause
  model and a learned up-to-five-clause model. For each learned version,
  concatenate ranked slots with three scores, one retrieval distance, and
  a presence indicator per slot. Zero-fill missing slots and set their
  presence indicator to zero.
- Fit the same scikit-learn pipeline for both versions on training rows
  only. Start with `StandardScaler` followed by `LogisticRegression`
  with `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, and no class
  weights. The target is the contract-level label. Gold evidence, document
  IDs, and split names must stay out of the features. Both models use the
  same feature definition per rank.
- The frozen checkpoint was fine-tuned on ContractNLI training contracts.
  The second-stage model therefore learns from first-stage predictions on
  contracts the checkpoint has seen. Disclose this as a limitation, and
  compare only the separate development predictions when judging gains.

### 4. Evaluate and hand back the experiment

- On all 1,037 development pairs, report correct counts, accuracy, macro
  F1, and recall for each class. Compare the five-clause model with the
  one-clause model to measure what additional clauses contribute. Include
  the 534/1,037 rank-one result and the separate 573/1,037 text-scorer
  reference. Also list the newer rank-one retrieval references, BM25 at
  613/1,037 and reranked Hybrid at 604/1,037, as context for the later
  retrieval comparison.
- Report the number of cases with at least one complete annotated span
  visible at depths 1, 3, and 5 out of the 614 annotated development pairs.
  Report the 423 gold Not Mentioned pairs through verdict metrics because
  they lack a positive annotated span. Use paired contract-level bootstrap
  intervals for accuracy differences. For the first development decision,
  look for accuracy above 573/1,037 with at least 39/95 contradictions and
  220/423 gold Not Mentioned cases correct. These are the original BGE
  handoff guardrails and precede the later retrieval results. Record
  trade-offs if the model misses one.
- Hand back the runnable evaluator, the manifest, per-case verdicts, the
  metric table, and a short breakdown of where the extra clauses helped
  or hurt. Update this file with results and the next decision. Diagnose
  whether remaining errors come from missed evidence, visible evidence
  with poor clause scores, or a poor final combination. The existing
  [held-out protocol](EdgeLegal_Project_Guide.md#64-proposed-held-out-protocol-2026-11-10)
  applies after a candidate is chosen on development data.
