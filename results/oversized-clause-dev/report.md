# EdgeLegal development experiment report

The November offline comparisons use the 1,037 contract–hypothesis
pairs from 61 ContractNLI development contracts. Retrieval metrics use
the 614 pairs with annotated evidence. The 423 gold Not Mentioned pairs
are included in verdict metrics. A raw evidence hit requires a complete
annotated span in a retrieved clause. A visible hit additionally requires
that the span survive DeBERTa's 512-token pair limit. At depth three or
five, a hit means at least one candidate qualifies. All experiments in
this report use zero Groq calls. See the [separate A/B/C pilot](../groq-pilot-20261110/report.md)
for hosted LLM results.

## Oversized-clause parser comparison

Compared 1,037 paired ContractNLI development cases using top-one dense retrieval and the same fine-tuned DeBERTa model. No Groq calls were made.

| Measure | Baseline | Split |
|---|---:|---:|
| Top-one annotated evidence hits | 256/614 (41.7%) | 218/614 (35.5%) |
| Unmapped annotated evidence | 0/614 (0.0%) | 3/614 (0.5%) |
| Inputs over 512 tokens | 195/1,037 (18.8%) | 1/1,037 (0.1%) |
| Correct predictions | 534/1,037 (51.5%) | 571/1,037 (55.1%) |
| Confidently wrong predictions | 405/1,037 (39.1%) | 346/1,037 (33.4%) |
| False Not Mentioned | 244/1,037 (23.5%) | 156/1,037 (15.0%) |
| Confident false Not Mentioned | 212/1,037 (20.4%) | 123/1,037 (11.9%) |
| Mean tokens before truncation | 473.5 | 188.2 |

Contract 405 has 14,464 source characters. The default parser produced one
clause. The opt-in parser produced nine chunks, the largest 1,992
characters, preserving the sequence of all non-whitespace characters.
The opt-in setting applies after normal segmentation and short-fragment
merging. It preserves the heading and assigns each chunk its own ID and
the page of its starting character. Local checks covered ordinary clauses,
an oversized whitespace-free span, and a synthetic page boundary.

**Decision.** Keep the original parser as the default. The split improved
accuracy and reduced truncation, while raw evidence hits fell.

Evidence rates use cases with annotated evidence. Unmapped spans count as misses. Confidence uses the existing 0.95 threshold. These are development results and the parser option remains off by default.

[Per-case data](cases.json)

## Evidence visibility and annotated-input diagnostic

The [visibility evaluator](../../backend/app/evaluate_evidence_visibility.py)
measured span survival using tokenizer offsets. It also rescored 321
distinct wrong Entailment or Contradiction cases across the two parser
arms using their annotated evidence. A case wrong under both parsers
reused one annotated-input prediction.

| Measure | Default parser | Oversized-clause split |
|---|---:|---:|
| Visible rank-one evidence | 222/614 | 218/614 |
| Rank-one hits fully removed by truncation | 33 | 0 |
| Rank-one hits partially removed | 1 | 0 |
| Wrong annotated cases corrected by supplied evidence | 278/301 | 208/228 |

With the default parser, supplied evidence corrected 202/215 errors
where a span mapped elsewhere and 45/55 errors where a span was already
visible. The rerun uses annotations and selects errors, so those counts
describe the diagnostic intervention. Gold Not Mentioned has no positive
annotated span and was outside this rerun.

[Visibility states and annotated-input verdicts](evidence-visibility.json)

## Top-three development comparison

The same 1,037 development cases were evaluated with the original parser and
fine-tuned checkpoint. Rank-one predictions match the saved baseline. The
alternate rule can replace a rank-one Not Mentioned verdict with a confident
Entailment or Contradiction from rank two or three when qualifying alternatives
agree. The main threshold is 0.95. No LLM calls were made.

| Metric | Rank one | Top-three rescue at 0.95 |
|---|---:|---:|
| Visible annotated evidence | 222/614 (36.2%) | 391/614 (63.7%) |
| Correct verdicts | 534/1,037 (51.5%) | 557/1,037 (53.7%) |
| Macro F1 | 0.464 | 0.482 |
| Contradictions correctly identified | 34/95 | 47/95 |
| Gold Not Mentioned correctly identified | 221/423 | 148/423 |
| Confidently wrong verdicts | 405 | 401 |

Of 185 overrides, 96 fixed errors, 73 introduced errors, and 16 changed
one wrong verdict to another. The net gain was 2.22 percentage points.
A paired bootstrap with 10,000 contract-level resamples gave a 95% interval
of −0.87 to +5.30 percentage points. The 0.90 and 0.98 sensitivity checks
each scored 558/1,037 and still lost substantial Not Mentioned recall.

Keep rank-one scoring as the live default. Test whether corroboration or
retrieval scores can improve the decision to override before a held-out run.

[Top-three per-case data](top-three.json)

## Stricter override comparison

Replayed the same 1,037 cached development cases. The stricter rules filter
the existing 0.95 rescue decisions. Agreement requires both rank-two and
rank-three clauses to give the same positive label at confidence at least
0.95. The distance band keeps the selected alternate only if its cosine
distance is at most 1.15 times the rank-one distance, matching the existing
retrieval gap policy. No model reruns or LLM calls were made.

| Rule | Correct verdicts | Contradiction recall | Gold Not Mentioned recall | Errors fixed | New errors |
|---|---:|---:|---:|---:|---:|
| Top one | 534/1,037 (51.5%) | 34/95 | 221/423 | — | — |
| Original 0.95 rescue | 557/1,037 (53.7%) | 47/95 | 148/423 | 96 | 73 |
| Both extra clauses agree | 551/1,037 (53.1%) | 39/95 | 215/423 | 23 | 6 |
| Alternate within 15% distance band | 555/1,037 (53.5%) | 44/95 | 173/423 | 69 | 48 |

Both filters reduced new errors but gave up some correct overrides. Paired
95% contract-bootstrap intervals for the accuracy change versus the original
rescue rule were −3.38 to +2.22 percentage points for agreement and −1.64 to
+1.25 points for the distance band. Neither filter earned a change to the
live default. [Per-case decisions and metrics](strict-overrides.json)

## Evidence verifier comparison

The [offline evaluator](../../backend/app/evaluate_evidence_verifier.py) fit a
logistic regression model to recognize candidates that both contained a
visible annotated span and gave the correct verdict. Its inputs were
inference-time signals, including retrieval scores, clause agreement, model
confidence, lexical overlap, and input length. Annotations were used only for
training targets and scoring. There were 56 positive examples among 237
eligible candidates. Five development folds held out whole contracts. The
verifier could accept only a previously eligible 0.95 rescue override, at a
fixed verifier cutoff of 0.50.

| Rule | Correct verdicts | Contradictions caught | New errors | Errors fixed |
|---|---:|---:|---:|---:|
| Both extra clauses agree | 551/1,037 (53.1%) | 39/95 | 6 | 23 |
| Evidence verifier | 553/1,037 (53.3%) | 36/95 | 6 | 25 |

The verifier's out-of-fold AUROC was 0.804 and average precision was 0.489
for the candidate training target. The routing comparison gained two correct
verdicts overall but caught three fewer contradictions. It did not meet the
contradiction guardrail. This is a development comparison from cached model
outputs. No DeBERTa reruns or Groq calls were made.

[Fold membership, per-case decisions, and metrics](evidence-verifier.json)

## Text-based evidence scorer

The [offline evaluator](../../backend/app/evaluate_text_evidence_scorer.py)
trained a hypothesis-conditioned TF-IDF logistic scorer on 24,586 clause pairs
from 423 training contracts, with 4,553 visible-evidence positives. Negative
examples included clauses with high lexical overlap but no complete annotated
span. Annotation misses can make some of these weak negatives. The classifier
and vocabulary were fit on training data only. Both training and development
inputs use the text visible within the 512-token DeBERTa input window.

| Rule | Correct verdicts | Contradictions caught | New errors | Errors fixed |
|---|---:|---:|---:|---:|
| Both extra clauses agree | 551/1,037 (53.1%) | 39/95 | 6 | 23 |
| Text evidence scorer | 573/1,037 (55.3%) | 38/95 | 1 | 40 |

The fixed 0.50 cutoff filtered the existing 185 rescue candidates. The scorer
had AUROC 0.961 and average precision 0.896 for visible evidence within those
candidates. The paired 95% contract-bootstrap interval for the accuracy
gain versus agreement was +0.58 to +3.86 percentage points. The scorer
missed the contradiction guardrail by one case. Agreement uniquely got
three contradictions right, while the text scorer uniquely got two right.
The default stays unchanged. No DeBERTa prediction reruns or Groq calls were made.

[Per-case decisions and metrics](text-evidence-scorer.json)

## Contract cross-reference audit

On all 1,037 development cases, the [offline audit](../../backend/app/evaluate_clause_references.py)
reused saved BGE rankings and verified clause IDs and text against the default
parser. It scanned rank-one clauses for explicit numbered section, article,
paragraph, and clause references. A lookup required a unique exact numeric
heading in a different clause of the same contract. The audit skipped
self-references, ambiguous headings, and qualified references other than
references to the agreement. It did not expand subsection references to
parent headings or follow more than one hop.

Of 164 cases with a reference-like phrase, 47 across 11 contracts had at
least one resolvable target. The rank-one clause missed annotated evidence
in 21 of those 47. Adding the first resolvable target in text order
recovered a complete annotated span in five cases, four of which the
baseline classified incorrectly. This raised rank-one evidence containment
from 256/614 to 261/614. Four of the five recoveries were also missed by
the saved top three, so adding the first target to those results raised
containment from 443/614 to 447/614. Following all resolvable direct
references recovered nine rank-one misses across four contracts, including
seven misses from the top three. All recoveries were Entailment cases.

This is evidence containment in raw clause text. Visibility within
DeBERTa's 512-token window and verdict accuracy were not measured. The
audit used no model inference or Groq calls. The small gain under exact
heading matching supports prioritizing broader retrieval experiments
before building an agent for reference lookup.

[Per-case reference matches and evidence spans](clause-reference-audit.json)

## Hybrid BGE and BM25 retrieval

The [evaluator](../../backend/app/evaluate_hybrid_retrieval.py) compared the
same 1,037 development cases from 61 contracts. Both arms used the default
parser, BAAI/bge-small-en with Chroma cosine distance, and the same fine-tuned
DeBERTa checkpoint on the first clause at a 512-token limit. The hybrid arm
added per-contract BM25 with case-folded word tokens, k1=1.2 and b=0.75. It
combined the two rankings with equal-weight reciprocal rank fusion at k=60.
These settings were fixed for this run. One BGE rank-three boundary had a
floating-point distance tie. Both arms used the saved BGE order at that tie.
The BGE top-three prefix and rank-one predictions matched the saved baseline
on all cases. There were zero Groq calls.

| Metric | BGE | BGE + BM25 |
|---|---:|---:|
| Raw evidence in top one | 256/614 | 360/614 |
| Raw evidence in top three | 443/614 | 535/614 |
| Raw evidence in top five | 539/614 | 577/614 |
| Evidence visible in top one | 222/614 | 312/614 |
| Evidence visible in top three | 391/614 | 478/614 |
| Evidence visible in top five | 484/614 | 520/614 |
| Correct rank-one verdicts | 534/1,037 (51.49%) | 574/1,037 (55.35%) |
| Macro F1 | 0.464 | 0.508 |
| Entailments caught | 279/519 | 324/519 |
| Contradictions caught | 34/95 | 42/95 |
| Correct gold Not Mentioned | 221/423 | 208/423 |
| False Not Mentioned | 244 | 201 |
| Confidently wrong at 0.95 | 405 | 366 |
| Rank-one inputs over 512 tokens | 195 | 250 |

For visibility at each depth, at least one complete annotated span had to
survive within one candidate's 512-token clause–hypothesis input. Each
candidate was tokenized individually. The verdict model used rank one only.

The hybrid arm changed 370 first-choice clauses, fixed 104 verdicts and
introduced 64 errors. The accuracy gain was 3.86 percentage points, with
a paired 95% contract-bootstrap interval of +1.06 to +6.65 points. The
earlier text scorer got 573 correct using a separate override experiment.
The hybrid gain comes with more truncation and lower gold Not Mentioned
recall. Carry it forward as a retrieval candidate. Compare BGE and hybrid
with the same multi-clause scoring method on development data before
choosing a candidate for held-out evaluation.

[Per-case rankings, verdicts, and summary](hybrid-retrieval.json)

## BM25 control and Hybrid reranking

The [evaluator](../../backend/app/evaluate_retrieval_reranking.py) used the
same 1,037 development cases, parser, DeBERTa checkpoint, and 512-token
verdict input limit. BM25 used the same per-contract scoring and parameters
as the hybrid experiment, with document order breaking ties independently
of BGE. Every query matched at least one term. A frozen
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranked only the saved Hybrid top
five, with a 512-token input limit and original Hybrid order breaking ties.
There were 4,787 unique reranker inputs, 390 uncached DeBERTa inputs, and
zero Groq calls.

| Metric | BGE | BM25 | Hybrid | Hybrid + MiniLM |
|---|---:|---:|---:|---:|
| Rank-one accuracy | 534/1,037 (51.49%) | 613/1,037 (59.11%) | 574/1,037 (55.35%) | 604/1,037 (58.24%) |
| Raw evidence in top one | 256/614 | 405/614 | 360/614 | 412/614 |
| Raw evidence in top three | 443/614 | 528/614 | 535/614 | 544/614 |
| Raw evidence in top five | 539/614 | 578/614 | 577/614 | 577/614 |
| Visible evidence in top one | 222/614 | 348/614 | 312/614 | 370/614 |
| Visible evidence in top three | 391/614 | 472/614 | 478/614 | 487/614 |
| Visible evidence in top five | 484/614 | 521/614 | 520/614 | 520/614 |
| Macro F1 | 0.464 | 0.539 | 0.508 | 0.541 |
| Entailments caught | 279/519 | 337/519 | 324/519 | 342/519 |
| Contradictions caught | 34/95 | 39/95 | 42/95 | 49/95 |
| Correct gold Not Mentioned | 221/423 | 237/423 | 208/423 | 213/423 |
| False Not Mentioned | 244 | 200 | 201 | 171 |
| Confidently wrong at 0.95 | 405 | 334 | 366 | 337 |
| Rank-one inputs over 512 tokens | 195 | 303 | 250 | 247 |

Paired 95% contract-bootstrap intervals were +3.95 to +11.38 percentage
points for BM25 versus BGE (+7.62), −6.46 to −1.06 points for Hybrid
versus BM25 (−3.76), and +0.77 to +5.11 points for reranked versus
plain Hybrid (+2.89). Reranking fixed 93 plain Hybrid errors and introduced
63. BM25 leads on development accuracy. Reranked Hybrid leads on visible
rank-one evidence and contradiction recall. These results support carrying
both candidates into the multi-clause comparison. A reranker on the BM25
shortlist would test whether its benefit also holds without dense retrieval.
An additional paired replay of the saved verdicts put reranked Hybrid
0.87 points below BM25, with a 95% contract-bootstrap interval of
−3.86 to +2.12 points. The development point estimates favor different
methods depending on whether overall accuracy or contradiction recall
is the priority.

[Per-case rankings, verdicts, and paired comparisons](retrieval-reranking.json)

## Reproducing the offline runs

Run from the repository root with Python 3.11 and
`backend/requirements.txt`. The model-based runs use the team's
`backend/app/triage_model/checkpoints/current/best_model.pt` and the
configured Hugging Face model assets. The dataset goes in
`backend/data/contractnli/`. Run `python backend/app/data_setup.py`
if the dataset is missing. The text scorer uses the training and
development splits. These comparisons use no test labels or Groq keys.

| Module to run with `python -m` | Reads | Writes in this results folder |
|---|---|---|
| `backend.app.evaluate_oversized_clauses` | Development split and models | `cases.json` and the parser-only report |
| `backend.app.evaluate_evidence_visibility` | `cases.json`, development split, and DeBERTa | `evidence-visibility.json` |
| `backend.app.evaluate_top_three` | `evidence-visibility.json`, development split, and models | `top-three.json` |
| `backend.app.evaluate_strict_overrides` | `top-three.json` | `strict-overrides.json` |
| `backend.app.evaluate_evidence_verifier` | `top-three.json` | `evidence-verifier.json` |
| `backend.app.evaluate_text_evidence_scorer` | `top-three.json` and train/dev splits | `text-evidence-scorer.json` |
| `backend.app.evaluate_clause_references` | `top-three.json` and development split | `clause-reference-audit.json` |
| `backend.app.evaluate_hybrid_retrieval` | `top-three.json`, development split, and models | `hybrid-retrieval.json` |
| `backend.app.evaluate_retrieval_reranking` | `top-three.json`, `hybrid-retrieval.json`, development split, and models | `retrieval-reranking.json` |

Use `--out` with a separate directory when rerunning the parser evaluator
to preserve this report. To rebuild the dependent comparisons from scratch,
work in a separate checkout or preserve the saved artifacts before replacing
them. The top-three evaluator resumes from an existing `top-three.json`.
The [progress file](../../docs/EdgeLegal_Progress.md) records the current
decision and the multi-clause training handoff.

For later RAG experiments with fixed excerpt budgets and model-directed lookup, see the [current progress](../../docs/EdgeLegal_Progress.md#reference-lookup-and-agentic-rag).
