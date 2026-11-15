# EdgeLegal’s five-clause experiment improved development accuracy

**Follow-up.** The frozen whole-contract model reached 86.11% on the same development cases. The [paired results](results/document-nli/comparison.json) preserve this five-clause experiment as a baseline.

The five-clause logistic regression scored **631/1,037 (60.85%)**, up from **568/1,037 (54.77%)** for the one-clause logistic regression and **534/1,037 (51.49%)** for rank-one DeBERTa. The five-clause model exceeded the overall and Contradiction targets. It scored **203/423** on Not Mentioned, 17 below the target, so the experiment stopped at development evaluation.

## What the experiment added

The offline evaluator tests whether multiple individually scored clauses improve a verdict about the entire contract. It adds a second model after retrieval and clause scoring. The second model combines features from ranked clauses and predicts one of three verdicts for each contract and hypothesis.

- [`evaluate_clause_combiner.py`](../../backend/app/evaluate_clause_combiner.py) implements the pilot, feature generation, logistic regression training, and paired evaluation.
- [`evidence_visibility.py`](../../backend/app/evidence_visibility.py) holds the existing token-level evidence visibility check. The older [evidence evaluator](../../backend/app/evaluate_evidence_visibility.py) imports the shared implementation.
- The experiment used EdgeLegal’s existing parser, BGE embedding wrapper, ChromaDB retrieval wrapper, and fine-tuned DeBERTa checkpoint. It ran offline with zero Groq calls and zero live-app changes. The held-out test set stayed untouched.

## How the dataset was generated

1. Read the original ContractNLI training and development splits. Each contract has 17 hypotheses. The resulting datasets have **7,191 rows from 423 training contracts** and **1,037 rows from 61 development contracts**. One row represents one contract and one hypothesis. The target is the original whole-contract annotation, one of **Contradiction**, **Entailment**, or **Not Mentioned**.
2. Parse each contract into clauses using EdgeLegal’s original segmentation settings with oversized-clause splitting disabled. Encode the clauses and hypotheses with `BAAI/bge-small-en` using normalized embeddings. Index the clauses in a per-contract ChromaDB collection with cosine distance and retrieve up to five clauses in ranked order, before the similarity-gap filter.
3. Score each retrieved clause as its own premise–hypothesis pair with frozen fine-tuned DeBERTa. The model applies a 512-token limit to the pair. The evaluator converts its logits to three softmax probabilities and records whether the original pair exceeded that limit. Batches group individual pairs for inference.
4. Build a fixed-width row with five ranked slots. Each slot contributes six values in this exact order.

| Position within slot | Feature |
|---:|---|
| 1 | DeBERTa probability of Contradiction |
| 2 | DeBERTa probability of Entailment |
| 3 | DeBERTa probability of Not Mentioned |
| 4 | ChromaDB cosine distance |
| 5 | Truncation flag, 1 when the original pair exceeds 512 tokens |
| 6 | Presence flag, 1 when a clause occupies the slot |

A contract with fewer than five clauses gets trailing zero-filled slots, including presence 0. Each exported feature row has an ordered array of 30 numeric values. Contract and hypothesis IDs are metadata, and the verdict is the training target. Gold evidence text, clause text, and evidence visibility are available in the per-contract records for auditing. Model input is limited to the numeric feature array.

The evaluator saves each contract atomically after all 17 hypotheses are processed. Checkpoints carry the manifest hash, which identifies the input files, cached model revisions and hashes, evaluator source, software versions, and retrieval and inference settings. The same command can reuse completed checkpoints when those inputs and settings match. [Training feature table](train_features.json) · [Development feature table](dev_features.json) · [Run manifest](manifest.json)

## What the 10-contract pilot verified

The pilot sampled ten development contracts across the range of parsed clause counts, including the shortest and longest contracts. All **170** first-ranked clause IDs, texts, and DeBERTa verdicts matched the saved baseline. Across **765** present slots, the largest deviation from a probability sum of 1 was **1.32 × 10⁻⁷**. The contracts with one and four clauses produced **85** padded slots across their hypotheses.

The CPU pilot took **396.9 seconds** for the sample plus **17.1 seconds** of setup. Scaling the mean per-contract time across all 484 contracts gave an initial estimate of **5 hours 20 minutes**. The full evaluator reused the ten saved pilot contracts and checked rank-one parity against the saved baseline across all 1,037 development cases. [Pilot results](pilot.json) · [Saved rank-one baseline](../oversized-clause-dev/cases.json)

## Why logistic regression

Logistic regression is a simple way to learn how clause-level scores, retrieval distances, and flags relate to a contract-level verdict. It can learn different weights for different ranks and classes. Training two versions of the same model on identical rows makes the comparison interpretable. The change in input between the two logistic models is the addition of ranks two through five.

| Method | Inputs and prediction rule |
|---|---|
| Rank-one DeBERTa | Take the highest-probability verdict from the first retrieved clause. |
| Rank-one logistic regression | Fit a scaler and a three-class logistic regression on the six features from rank one. |
| Top-five logistic regression | Fit a scaler and a three-class logistic regression on all 30 features from the five ranked slots. |

Both logistic models fit their scaler and coefficients on the same 7,191 training rows. Both used `StandardScaler`, L2 regularization with fixed `C=1.0`, and the `lbfgs` solver with a 2,000-iteration cap. The run used fixed settings without hyperparameter tuning. DeBERTa supplied the features with frozen weights. The time-intensive work was generating the DeBERTa features. Fitting the two small regressions ran immediately after the feature tables were saved. [Rank-one model](rank_one_logistic.joblib) · [Top-five model](top_five_logistic.joblib)

## Results on the same 1,037 development cases

| Method | Correct | Accuracy | Macro F1 | Contradiction | Entailment | Not Mentioned |
|---|---:|---:|---:|---:|---:|---:|
| Rank-one DeBERTa | 534 | 51.49% | 0.4641 | 34/95 | 279/519 | 221/423 |
| Rank-one logistic regression | 568 | 54.77% | 0.4796 | 32/95 | 350/519 | 186/423 |
| Top-five logistic regression | 631 | 60.85% | 0.5579 | 40/95 | 388/519 | 203/423 |

Class columns show correct predictions divided by the number of gold cases in that class. Macro F1 averages the three class F1 scores equally.

Compared with rank-one logistic regression, top-five **fixed 131 cases** and **got 68 wrong that rank-one logistic got right**. The net gain was **63 correct verdicts**, or **6.08 percentage points** in accuracy. The extra clauses improved correct counts for all three classes relative to rank-one logistic regression. Rank-one DeBERTa still had the highest Not Mentioned recall.

| Target | Top-five result | Outcome |
|---|---:|---|
| At least 574/1,037 overall | 631/1,037 | Met by 57 |
| At least 39/95 Contradiction | 40/95 | Met by 1 |
| At least 220/423 Not Mentioned | 203/423 | Missed by 17 |

[Machine-readable comparison](comparison.json) · [Per-case development predictions](dev_predictions.json)

## What the evidence and errors showed

Evidence visibility required at least one complete verbatim gold span to survive tokenization in a retrieved clause’s premise. Of the 614 development cases with annotated evidence, **222 (36.2%)** had visible evidence at rank one, **391 (63.7%)** by rank three, and **484 (78.8%)** by rank five. Ranks two through five therefore exposed annotated evidence in **262 additional cases**.

| Diagnostic for the 406 top-five errors | Cases |
|---|---:|
| Gold Not Mentioned case predicted as Entailment or Contradiction | 220 |
| Gold evidence was absent from the visible part of all five scored clauses | 55 |
| Evidence was visible, but each DeBERTa verdict on those evidence-bearing clauses differed from the contract verdict | 61 |
| An evidence-bearing clause had the correct DeBERTa verdict, but the combiner chose a different verdict | 70 |

These diagnostics describe the observed inputs and outputs. A visible annotated span can still omit surrounding context needed for the whole-contract verdict. Cases without visible evidence can reflect retrieval rank, segmentation, or truncation.

The development decision was to hold off on a held-out evaluation because top-five missed the Not Mentioned target. The proposed next experiment would use the saved training features and whole-contract folds to tune how much confidence is required to predict a mention. Any selected rule would be locked before checking the development targets again.

## Reproduce the completed run

The command below was run from the Part 2 EdgeLegal checkout. The original ContractNLI JSON files and fine-tuned checkpoint were read from a separate local EdgeLegal folder.

```powershell
python -u -m backend.app.evaluate_clause_combiner --data C:/Users/koman/OneDrive/Desktop/edgelegal/backend/data/contractnli --checkpoint C:/Users/koman/OneDrive/Desktop/edgelegal/backend/app/triage_model/checkpoints/best_model.pt
```

On another computer, replace the two absolute input paths. The evaluator also needs the saved baseline, installed EdgeLegal Python dependencies, and local Hugging Face caches for BGE and the DeBERTa base model. Use `--pilot-only` to run only the ten-contract pilot. Use a fresh `--out` directory for a new run, or reuse the same paths and settings to resume. The verified SHA-256 of the fine-tuned `best_model.pt` is `7ea45d37671897f04267f5a9e187a28790631362c8231c99757db620da1cbc61`. [Run instructions](README.md) · [Run manifest](manifest.json)
