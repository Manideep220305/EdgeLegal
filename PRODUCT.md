# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: demo audiences and evaluators — a thesis panel, paper reviewers,
and recruiters/engineers viewing it as a resume project. They watch or try a
live run and judge whether the mechanism is real and well built.

Legal reviewers checking NDAs are the product's story and intended future
user, not today's actual user.

## Product Purpose

EdgeLegal checks a contract for clause-level contradictions against a fixed
set of hypotheses. A user uploads a contract PDF, picks a hypothesis, and gets
a verdict (Entailment / Contradiction / NotMentioned) with the exact clauses
and pages it is based on, highlighted in the document.

Success: a viewer understands the verdict, trusts it because the evidence is
shown, and sees the confidence-gated mechanism working.

## Positioning

A small fine-tuned NLI classifier acts as a confidence-gated triage layer in
front of an LLM: confident cases are accepted directly, uncertain ones are
escalated. The goal is fewer LLM calls without losing accuracy, and only if
the classifier's confidence is actually calibrated.

The document classifier can run locally. Hosted deployment is a product
plan. Cloud review sends the question and selected excerpts to Groq.
An opt-in experimental mode lets the model request more evidence and
shows which PDF pages it looked up.

## Operating Context

- Input: contract PDFs (NDAs, ContractNLI-style).
- Flow in `/app`: upload -> choose hypothesis -> analysis -> report with
  clickable citations that jump to the source in the PDF viewer.
- `/` is the landing page that explains the product and links into `/app`.

## Capabilities and Constraints

- Dataset: ContractNLI — 607 NDAs, 17 fixed hypotheses, labels Entailment /
  Contradiction / NotMentioned.
- Triage classifier: DeBERTa-v3-base NLI cross-encoder, fine-tuned, with a
  frozen confidence threshold for escalation.
- Retrieval over parsed clauses, LLM reasoning for escalated cases.
- The default app uses confidence routing. The whole-contract model scored
  89.09% on the 103-contract benchmark cohort with zero LLM calls. The
  hard-case agent pilot is complete and the matched held-out routing
  comparison is pending. The agent remains experimental.
  See [current research progress](docs/EdgeLegal_Progress.md).

## Evidence on Hand

- Real: the dataset, the method, the trained checkpoint, the working app.
- Development and pilot metrics are available. Product accuracy, latency,
  and cost-saving claims still need a full held-out evaluation.
- No customers, testimonials, or deployment claims exist. Do not invent them.

## Product Principles

1. Every verdict shows its evidence — clause text and page, never a bare label.
2. Show the mechanism honestly: confidence and whether a result was escalated.
3. No fabricated numbers or claims; unfinished results are labeled pending.
4. Demo clarity first: a first-time viewer should follow the flow unaided.
