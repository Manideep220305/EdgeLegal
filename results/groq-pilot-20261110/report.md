# EdgeLegal Groq pilot report

**Run date** — 10 November 2026

The A/B/C pilot completed in **25 minutes 25 seconds** across **51 cases** from three ContractNLI development contracts (IDs 3, 44, and 405). One Groq key made 51 requests to `openai/gpt-oss-20b`. All 51 returned valid verdicts, with no retries or rate-limit errors.

## Results

| Config | Correct verdicts | Accuracy | Groq calls if run separately |
|---|---:|---:|---:|
| A · Groq for every case | 33/51 | 64.7% | 51 |
| B · Pretrained DeBERTa + Groq | 29/51 | 56.9% | 12 |
| C · Fine-tuned DeBERTa + Groq | 23/51 | 45.1% | 9 |

Each case used the same retrieved context and Groq prompt across configs, without a DeBERTa verdict hint. B and C reused A's saved answer when their classifier confidence was below 0.95. Reuse reduced actual requests from 72 to 51.

Config C skipped 42 of 51 calls (82.4%), but 24 of its 42 automatically accepted verdicts were wrong. A had the highest overall accuracy. B and C each caught three of four contradictions, while A caught two. The sample is too small to rank the configs on contradiction detection or draw conclusions about the full benchmark.

## Timing and quota

- Median Groq response time was **0.81 seconds**. Pacing against the token limit accounted for most of the elapsed time.
- Groq reported **89,509 tokens** (80,895 input and 8,614 output), averaging **1,755 per request**.
- Contracts 3 and 44 averaged **1,031** and **904** tokens per request. Contract 405 parsed into one long clause and averaged **3,330**.
- At the pilot average, 2,091 shared requests for the full test split would use about **3.67 million tokens**, or **19 daily quota windows** at 200,000 tokens per day. This is a projection from three contracts.

Before starting the full run, measure prompt lengths across the development split without calling Groq. That will show whether the long contract is typical and give a better estimate of the required quota.

The original per-case JSON remains in the companion EdgeLegal workspace at
C:/Users/katta/OneDrive/Documents/ChatGPT/EdgeLegal/groq-pilot-20261110.json.
This repository copy preserves the pilot summary.
