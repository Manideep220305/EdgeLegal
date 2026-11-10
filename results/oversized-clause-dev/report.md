# Oversized-clause parser comparison

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

**Decision.** Did not meet all three criteria; keep the baseline for the next experiment.

Evidence rates use cases with annotated evidence. Unmapped spans count as misses. Confidence uses the existing 0.95 threshold. These are development results and the parser option remains off by default.

[Per-case data](cases.json)

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
