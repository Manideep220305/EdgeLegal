# EdgeLegal: Agentic RAG System for Privacy-Preserving Legal Contract Analysis and Contradiction Detection

> Original project plan. The document-model benchmark ran without LLM calls. The experimental agent sends selected passages and the question to Groq. See [current results and limitations](EdgeLegal_Progress.md).

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why This Problem Matters](#2-why-this-problem-matters)
3. [End Customers](#3-end-customers)
4. [Existing Solutions and Their Failures](#4-existing-solutions-and-their-failures)
5. [Research Gaps — Honest Assessment](#5-research-gaps--honest-assessment)
6. [Our Proposed System](#6-our-proposed-system)
7. [Features and Tech Stack](#7-features-and-tech-stack)
8. [Dataset — ContractNLI](#8-dataset--contractnli)
9. [How We Evaluate Without Training](#9-how-we-evaluate-without-training)
10. [Pipeline — Step by Step](#10-pipeline--step-by-step)
11. [Structure-Aware Chunking — How It Works](#11-structure-aware-chunking--how-it-works)
12. [The LangGraph Agent — How It Reasons](#12-the-langgraph-agent--how-it-reasons)
13. [Why This Is Not Just a RAG Chatbot](#13-why-this-is-not-just-a-rag-chatbot)
14. [Deployment and the Local LLM Question](#14-deployment-and-the-local-llm-question)
15. [Ablation Studies and Evaluation Plan](#15-ablation-studies-and-evaluation-plan)
16. [Research Scope and Paper Strategy](#16-research-scope-and-paper-strategy)
17. [Possible Improvements and Extensions](#17-possible-improvements-and-extensions)
18. [Honest Risk Assessment](#18-honest-risk-assessment)

---

## 1. Problem Statement

Every individual and organization regularly signs legal documents — employment contracts, NDAs, vendor agreements, service agreements, and terms and conditions. These documents are long, dense, and deliberately written in complex legal language. Understanding them fully requires a trained lawyer.

This creates three compounding problems:

**Problem A — Cost.** Hiring a lawyer to review every contract is expensive. Large corporations can afford it. Startups, small businesses, and individuals cannot. Most people sign contracts without understanding what they are agreeing to.

**Problem B — Privacy.** The obvious solution today is to paste the contract into ChatGPT or Claude and ask for a summary. But contracts contain sensitive information — salary details, company trade secrets, personal identification, financial terms, confidential business obligations. Uploading these to a third-party cloud AI is a serious data privacy risk. Many companies explicitly prohibit this in their own data handling policies. Banks, hospitals, law firms, and government organizations cannot use cloud AI for legal document review under any circumstances.

**Problem C — Contradiction Detection.** Even when lawyers review contracts, subtle contradictions between clauses are easy to miss — especially in documents with 30-50 pages. A clause on page 2 may directly contradict a clause on page 38. Current AI tools that perform legal Q&A are good at retrieval but poor at reasoning across multiple clauses simultaneously to detect conflicts.

**Our goal:** Build a fully local, privacy-preserving AI system that reads any legal document, detects contradictions between clauses, flags risky obligations, and answers questions about the contract — without sending a single byte to OpenAI, Anthropic, or any third-party API.

---

## 2. Why This Problem Matters

Consider a real example.

An employee receives a 42-page employment contract. They don't have a lawyer. They cannot paste it into ChatGPT because it contains their salary, designation, and company-confidential project details. They sign.

Six months later they try to leave the company. They discover:
- Clause 14: "Employee may resign with 30 days notice"
- Clause 31: "Employee agrees to a 6-month non-compete across all related industries in India"
- Clause 38: "Any invention created during employment is the exclusive property of the company, including work done outside office hours on personal devices"

None of these clauses were explained to them. Two of them contradict common employment law expectations. One of them (Clause 38) is potentially unenforceable but most people would never know.

This is not a rare edge case. This happens to every person who signs a contract without legal counsel.

The same problem exists at scale for:
- Startups reviewing vendor agreements
- Small businesses signing service contracts
- HR departments processing employment agreements
- Anyone accepting Terms and Conditions for software products

---

## 3. End Customers

| Customer | Use Case | Why They Can't Use Cloud AI |
|---|---|---|
| Individual employees | Review employment contracts before signing | Contains salary, personal details |
| Startups | Review vendor/investor agreements | Contains trade secrets, financial terms |
| Small businesses | NDA and service contract review | Cannot afford lawyers, privacy concerns |
| HR departments | Standardize employment contract review | Bulk processing of sensitive employee data |
| Law firms | First-pass review before lawyer deep-dive | Client confidentiality obligations |
| Healthcare organizations | Vendor and service agreements | HIPAA-equivalent data obligations |
| Government organizations | Procurement contracts | Cannot use foreign cloud services |

---

## 4. Existing Solutions and Their Failures

### 4.1 Current Legal AI Tools

Tools like Harvey AI, CoCounsel, Lexis AI, and Thomson Reuters AI exist for enterprise legal teams. They are:
- Expensive (enterprise pricing, not accessible to individuals or small businesses)
- Cloud-based (data leaves the organization)
- Focused on large law firms, not individuals or SMEs
- Not specifically designed for contradiction detection

### 4.2 Generic ChatGPT / Claude Approach

Most people who try to use AI for contract review paste the document into a chat interface. Problems:
- Hard privacy limit — data goes to a third-party server
- Context window limits on very long contracts
- No structured contradiction detection — the model summarizes but does not explicitly identify clause-level conflicts
- No evaluation framework — no way to know if the model missed something

### 4.3 Standard RAG Systems

Typical RAG pipelines for legal documents work like this:

```
Upload Contract
      ↓
Split into 500-token chunks (random)
      ↓
Embed chunks
      ↓
Store in vector DB
      ↓
User asks question
      ↓
Retrieve top-K chunks
      ↓
LLM answers
```

**Why this fails for contradiction detection:**

1. **Random chunking breaks clause boundaries.** A clause that spans 600 tokens gets split in the middle. Retrieved chunks are incomplete.

2. **Single-pass retrieval misses cross-clause conflicts.** If Clause 5 and Clause 38 contradict each other, a single retrieval step that fetches Clause 5 will not automatically fetch Clause 38. The contradiction is never surfaced.

3. **No structured output.** The system answers questions but does not produce a structured contradiction report.

4. **No evaluation.** There is no ground truth to compare against, so you cannot measure how many contradictions were missed.

---

## 5. Research Gaps — Honest Assessment

### Genuine Gaps (Where We Can Contribute)

**Gap 1 — Quantized Local LLMs for Legal Reasoning Not Benchmarked**
Most papers on legal NLP use GPT-4, Claude, or other frontier cloud models. Nobody has rigorously benchmarked quantized local models (Phi-3 Mini, Llama 3 8B at 4-bit quantization) specifically for contradiction detection on ContractNLI with proper ablations. This is a real gap.

**Gap 2 — Agentic Multi-Step Retrieval for Contradiction Detection**
There is no published work that uses a LangGraph-style agent with explicit tools (`retrieve_clause`, `cross_reference`, `flag_contradiction`) for legal contradiction detection specifically. The agentic approach — where the model decides what to retrieve next based on what it already found — has not been applied to cross-clause contradiction detection on a standardized legal benchmark.

**Gap 3 — Structure-Aware Chunking + Agentic Retrieval Combined**
Structure-aware chunking for legal documents has been studied (NLLP 2025 paper on SAC). Agentic RAG has been studied separately. Their combination specifically for contradiction detection on ContractNLI has not been evaluated.

### Partially Claimed Gaps (Be Careful)

**Claimed — Privacy-preserving local RAG for legal documents**
A 2026 paper did this for Indian legal law using Llama 3 + ChromaDB. Your differentiation must be contradiction detection specifically, not just "local RAG." Do not make privacy your primary research claim. Make it a system property, not the contribution.

**Claimed — Structure-aware chunking for legal docs**
NLLP 2025 paper proposed this. Do not claim you invented structure-aware chunking. Claim you are combining it with agentic retrieval and evaluating it on ContractNLI for contradiction detection specifically.

**Claimed — Multi-agent contract review**
A 2025 ACL paper did multi-agent local LLM for contract clause analysis. Your differentiation is the standardized evaluation on ContractNLI and the contradiction detection focus.

### The Narrow but Real Research Question

> *How does structure-aware clause retrieval combined with agentic multi-step reasoning affect contradiction detection F1 on ContractNLI under local compute constraints?*

That is your research contribution. Not broader, not narrower.

---

## 6. Our Proposed System

**Title:** EdgeLegal — Agentic RAG System for Privacy-Preserving Legal Contract Analysis

**Core Claim:** Structure-aware clause retrieval combined with agentic multi-step reasoning improves contradiction detection performance over single-pass RAG, while operating entirely on local compute without third-party API dependencies.

**What we are NOT training:** No model training. No fine-tuning. Pre-trained LLM weights, pre-trained embedding model, custom agent logic.

**What we ARE building:**
- Structure-aware clause parser
- LangGraph agent with legal reasoning tools
- Evaluation pipeline against ContractNLI
- FastAPI backend + React frontend for demo

---

## 7. Features and Tech Stack

| Feature | Description | Technology |
|---|---|---|
| Document Upload | Accept PDF legal documents | PyMuPDF |
| Structure-Aware Clause Extraction | Split by clause boundaries, not token count | Custom regex parser + heuristics |
| Embedding Generation | Convert clauses to dense vectors | BGE-small-en / E5-base |
| Vector Storage | Store and retrieve clause embeddings | ChromaDB (local) |
| Agentic Reasoning | Multi-step retrieval and reasoning | LangGraph |
| Contradiction Detection | Identify conflicting clause pairs | LangGraph agent + local LLM |
| Risk Flagging | Highlight dangerous obligations | LLM with structured output |
| Confidence Scoring | Score each detected contradiction by confidence | LLM calibration |
| Plain English Summary | Summarize what the user is agreeing to | LLM |
| Multi-turn Q&A | Ask questions about the contract | Agentic RAG pipeline |
| Fully Local Inference | No third-party LLM calls | Llama 3 8B (Ollama) / Phi-3 Mini |
| API Backend | Serve the system | FastAPI |
| Frontend | Upload, view, interact | React |
| Evaluation Pipeline | Batch evaluation on ContractNLI | Python script + sklearn metrics |

---

## 8. Dataset — ContractNLI

### What It Is

ContractNLI is a publicly available, peer-reviewed dataset specifically designed for contract-level natural language inference. It contains **607 real Non-Disclosure Agreements (NDAs)** collected from the SEC EDGAR database. It is split into:
- Train: ~400 contracts
- Dev: ~50 contracts
- Test: ~150 contracts

**We only use the test split for evaluation.**

### Structure of Each Sample

Each NDA is paired with **17 fixed hypotheses**. For example:

| Hypothesis | Ground Truth |
|---|---|
| "The confidentiality obligation survives termination" | Entailment |
| "The receiving party must delete data after 30 days" | Contradiction |
| "The agreement covers verbal disclosures" | NotMentioned |

Each hypothesis is labelled with one of three labels:
- **Entailment** — the contract supports the hypothesis
- **Contradiction** — the contract contradicts the hypothesis
- **NotMentioned** — the contract does not address this

### Why This Dataset

- Public and freely available
- Peer-reviewed and widely cited in legal NLP literature
- Provides objective ground truth labels for contradiction detection
- Results are directly comparable to other published papers
- Eliminates subjectivity from evaluation — contradiction is not a matter of opinion, it has a label

### Limitation to Acknowledge

ContractNLI is limited to NDAs. It does not cover employment contracts, vendor agreements, or T&Cs. This is a known limitation of the dataset and should be stated honestly in the paper. Generalization to other contract types is listed as future work.

---

## 9. How We Evaluate Without Training

This is the most commonly misunderstood part.

**We do not train the LLM.** The LLM comes with pre-trained weights. We use it as-is (or quantized).

**We do not feed all 607 NDAs simultaneously.** We write a batch evaluation script that processes one contract at a time, automatically.

### The Evaluation Loop

```python
for contract in contractnli_test_split:
    # Step 1: Load and chunk the contract
    clauses = structure_aware_chunker(contract.text)
    
    # Step 2: Embed and store in ChromaDB
    vectorstore.add(clauses)
    
    # Step 3: For each hypothesis, run the agent
    for hypothesis in contract.hypotheses:
        predicted_label = agent.evaluate(
            hypothesis=hypothesis.text,
            vectorstore=vectorstore
        )
        results.append({
            "predicted": predicted_label,
            "ground_truth": hypothesis.label
        })
    
    # Step 4: Clear ChromaDB for next contract
    vectorstore.clear()

# Step 5: Compute metrics
from sklearn.metrics import classification_report
print(classification_report(
    [r["ground_truth"] for r in results],
    [r["predicted"] for r in results]
))
```

### Forcing Structured Output From LLM

The LLM prompt ends with:

```
Based on the retrieved clauses, does this contract:
- Support the following statement (Entailment)
- Contradict the following statement (Contradiction)  
- Not mention this at all (NotMentioned)

Statement: {hypothesis}

Respond with exactly one word: Entailment, Contradiction, or NotMentioned.
```

Or using LangChain `with_structured_output`:

```python
class ContractNLIOutput(BaseModel):
    label: Literal["Entailment", "Contradiction", "NotMentioned"]
    evidence_clause_id: str
    confidence: float

llm_with_structure = llm.with_structured_output(ContractNLIOutput)
```

Then evaluation is a simple string comparison:
```python
predicted == ground_truth  # True or False
```

### Metrics Reported

- **Contradiction F1** — primary metric (what we claim to improve)
- **Overall Accuracy** — secondary metric
- **Precision and Recall per class** — full picture
- **Latency** — time per contract on local hardware

---

## 10. Pipeline — Step by Step

```
User uploads PDF contract
          ↓
[PyMuPDF] Extract raw text
          ↓
[Structure-Aware Parser] Split into clauses at clause boundaries
Metadata stored: clause_id, clause_number, page_number, section_header
          ↓
[BGE-small / E5-base] Generate clause embeddings
          ↓
[ChromaDB] Store clause vectors + metadata locally
          ↓
System runs automatic analysis OR user asks question
          ↓
[LangGraph Agent] Activated
          ↓
Agent calls tools based on query:

    retrieve_clause(query) 
    → semantic search in ChromaDB
    → returns top-K relevant clauses

    cross_reference(clause_id_A, clause_id_B)
    → retrieves both clauses
    → sends to LLM for comparison

    flag_contradiction(clause_A, clause_B)
    → LLM determines if conflict exists
    → returns structured output with confidence score

    summarize_risk(clauses)
    → LLM identifies risky obligations
    → returns structured risk report

          ↓
[Local LLM — Llama 3 8B / Phi-3] Reasons over retrieved clauses
          ↓
Output:
    - Contradiction report (clause pairs that conflict)
    - Risk report (dangerous obligations)
    - Confidence scores per finding
    - Plain English summary
    - Interactive Q&A capability
          ↓
[React Frontend] Display highlights, reports, chat interface
```

---

## 11. Structure-Aware Chunking — How It Works

### The Problem With Random Chunking

Standard RAG splits text every 500 tokens regardless of content:

```
...The receiving party shall maintain strict confidentiality
regarding all disclosed information. [CHUNK BREAK]
3. Payment Terms
3.1 All payments shall be made within...
```

Clause 2 is now split across two chunks. Any retrieval of the second chunk is meaningless without the first.

### Structure-Aware Approach

Legal contracts follow consistent structural patterns:

```
1. Confidentiality Obligations
   1.1 The receiving party shall...
   1.2 This obligation shall survive...

2. Payment Terms
   2.1 All payments due within 30 days...
   2.2 Late payment shall incur...

Article III — Termination
Section 3.1 Either party may terminate...
```

We parse these patterns using regex and heuristics:

```python
import re

CLAUSE_PATTERNS = [
    r'^\d+\.\s+[A-Z]',           # "1. Confidentiality"
    r'^\d+\.\d+\s+[A-Z]',        # "1.1 The receiving party"
    r'^Article\s+[IVXLC]+',       # "Article III"
    r'^Section\s+\d+\.\d+',       # "Section 3.1"
    r'^CLAUSE\s+\d+',             # "CLAUSE 5"
    r'^[A-Z][A-Z\s]{3,30}\n',     # "PAYMENT TERMS"
]

def structure_aware_chunk(text):
    clauses = []
    current_clause = []
    current_header = None
    
    for line in text.split('\n'):
        is_header = any(re.match(p, line) for p in CLAUSE_PATTERNS)
        if is_header and current_clause:
            clauses.append({
                'header': current_header,
                'text': '\n'.join(current_clause),
                'clause_id': generate_id()
            })
            current_clause = []
            current_header = line
        current_clause.append(line)
    
    return clauses
```

Each chunk is one complete clause with its ID stored in metadata. This means cross-referencing between clauses is now possible.

### Why This Matters for Contradiction Detection

When Clause 5 says "30 days payment" and Clause 18 says "immediate payment," both clauses need to be retrieved and compared together. With random chunking, neither clause is guaranteed to be retrieved completely. With structure-aware chunking, each clause is a self-contained unit with a unique ID, making cross-reference retrieval precise and reliable.

---

## 12. The LangGraph Agent — How It Reasons

### Why an Agent Instead of a Single LLM Call

A single LLM call for a 40-page contract does not work:
- Context window overflows
- No mechanism to go back and retrieve additional evidence
- Cannot self-correct if the first retrieval was insufficient

An agent using LangGraph can:
- Retrieve clause A
- Decide it needs clause B to confirm a suspicion
- Retrieve clause B
- Compare them
- Decide whether a contradiction exists
- Request more clauses if confidence is low

### Agent Tools

```python
from langchain.tools import tool

@tool
def retrieve_clause(query: str) -> list[dict]:
    """Search ChromaDB for clauses relevant to the query.
    Returns top-5 clauses with their IDs and text."""
    results = vectorstore.similarity_search(query, k=5)
    return [{"id": r.metadata["clause_id"], 
             "text": r.page_content} for r in results]

@tool  
def cross_reference(clause_id_a: str, clause_id_b: str) -> dict:
    """Retrieve two specific clauses by ID for direct comparison.
    Use when you suspect two clauses may conflict."""
    clause_a = vectorstore.get_by_id(clause_id_a)
    clause_b = vectorstore.get_by_id(clause_id_b)
    return {"clause_a": clause_a, "clause_b": clause_b}

@tool
def flag_contradiction(clause_a: str, clause_b: str) -> dict:
    """Send two clauses to the LLM to determine if they contradict.
    Returns label, explanation, and confidence score."""
    response = llm.with_structured_output(ContradictionOutput).invoke(
        CONTRADICTION_PROMPT.format(a=clause_a, b=clause_b)
    )
    return response

@tool
def summarize_risk(clauses: list[str]) -> dict:
    """Identify risky obligations in the provided clauses.
    Returns structured risk report with severity levels."""
    response = llm.invoke(RISK_PROMPT.format(clauses=clauses))
    return response
```

### Agent Decision Flow

```
User Query / Auto-Analysis Trigger
          ↓
Agent receives task: "Find contradictions in this contract"
          ↓
Step 1: retrieve_clause("payment terms")
→ Gets Clause 5 ("payment within 30 days")
          ↓
Step 2: retrieve_clause("payment due immediately")  
→ Gets Clause 18 ("payment due on invoice")
          ↓
Step 3: Agent suspects conflict
→ Calls cross_reference(clause_id_5, clause_id_18)
          ↓
Step 4: flag_contradiction(clause_5_text, clause_18_text)
→ LLM confirms: CONTRADICTION, confidence 0.91
          ↓
Step 5: Agent continues searching for more contradictions
→ Calls retrieve_clause("termination notice period")
          ↓
... continues until all major clause categories checked
          ↓
Final Output: Structured contradiction report
```

---

## 13. Why This Is Not Just a RAG Chatbot

Four things that separate this from a chatbot:

### 1. Structured Output Not Free Text
The system does not say "I found some issues with your contract." It produces:

```json
{
  "contradictions": [
    {
      "clause_a": {"id": "5", "text": "..."},
      "clause_b": {"id": "18", "text": "..."},
      "type": "PAYMENT_CONFLICT",
      "explanation": "Clause 5 requires payment within 30 days while Clause 18 requires immediate payment on invoice generation.",
      "confidence": 0.91,
      "severity": "HIGH"
    }
  ],
  "risk_flags": [...],
  "summary": "..."
}
```

### 2. Multi-Step Agentic Retrieval
The agent does not make one retrieval call and answer. It makes multiple retrieval calls, cross-references specific clauses by ID, and iteratively builds its understanding. This is reasoning, not retrieval.

### 3. Ground Truth Evaluation
The system is evaluated against ContractNLI labels. We report F1 score for contradiction detection. Not "look how good the output looks." Actual numbers against actual ground truth.

### 4. Ablation Studies
Three system variants are compared:
- Baseline: Random chunking + single-pass RAG + LLM
- Variant A: Structure-aware chunking + single-pass RAG + LLM
- Variant B: Structure-aware chunking + agentic multi-step retrieval + LLM

If Variant B does not outperform Baseline on contradiction F1, we report that honestly. This is what makes it research.

---

## 14. Deployment and the Local LLM Question

### The Privacy vs Practicality Tension

"Fully local" means different things:

| Mode | LLM runs on | Privacy | Feasibility |
|---|---|---|---|
| True local (desktop app) | User's machine | Maximum | Needs GPU or quantized model, slow on CPU |
| Self-hosted server | Your own server | High — no big tech | Practical for demo and deployment |
| Groq API | Groq's servers | Medium — not OpenAI/Anthropic | Fast, free tier, but data leaves device |

### Practical Stack for This Project

**Development and Demo:** Use Groq API (Llama 3 70B). Fast, free, already familiar from Smart IMS. The privacy argument is "no OpenAI/Anthropic/Google" which still holds.

**Local inference option:** Ollama with Llama 3 8B GGUF (4-bit quantized). Runs on CPU at ~3-5 tokens/second. Usable for demo. Showcase this as the privacy-first deployment option.

**ContractNLI Evaluation batch:** Run on Google Colab (free T4 GPU) overnight. Save results. This is a one-time run, not a product feature.

### For Your Guide
Say: "The system uses locally deployable open-source LLMs, with no dependency on proprietary APIs like OpenAI or Anthropic." This is accurate regardless of whether you use Groq for the demo.

---

## 15. Ablation Studies and Evaluation Plan

### Three System Variants

| Variant | Chunking | Retrieval | System |
|---|---|---|---|
| Baseline | Random (500 tokens) | Single-pass RAG | LLM answers once |
| Variant A | Structure-aware | Single-pass RAG | LLM answers once |
| Variant B (Ours) | Structure-aware | Agentic multi-step | Agent with tools |

### Metrics Per Variant on ContractNLI Test Split

| Metric | Baseline | Variant A | Variant B |
|---|---|---|---|
| Contradiction F1 | ? | ? | ? (expected highest) |
| Entailment F1 | ? | ? | ? |
| Overall Accuracy | ? | ? | ? |
| Avg clauses retrieved per query | ? | ? | ? |
| Avg latency per contract | ? | ? | ? |

You fill in the actual numbers after running the evaluation.

### Why This Ablation Structure Works for Publication

- Shows that structure-aware chunking alone improves over baseline (Variant A vs Baseline)
- Shows that agentic retrieval on top of that improves further (Variant B vs Variant A)
- Each improvement is isolated and measurable
- Reviewers cannot say "how do you know which component contributed?" — you've isolated them

### Additional Comparison: Local vs Cloud LLM

Run the same Variant B pipeline with:
- Phi-3 Mini (local, 3.8B)
- Llama 3 8B quantized (local, 8B)
- Llama 3 70B via Groq (cloud-equivalent)

Report accuracy vs latency vs RAM usage tradeoff. This becomes Paper 2.

---

## 16. Research Scope and Paper Strategy

### Paper 1 — Core System Paper

**Title:** Agentic RAG for Legal Contradiction Detection: A Structure-Aware Approach on ContractNLI

**Research Question:** Does structure-aware clause retrieval combined with agentic multi-step reasoning improve contradiction detection F1 over single-pass RAG on ContractNLI?

**Contribution:**
- Propose a structure-aware clause parser for legal documents
- Design a LangGraph agent with legal reasoning tools
- Evaluate three system variants on ContractNLI test split
- Report ablation results showing contribution of each component

**Target Journals:** MDPI Applied Sciences, Expert Systems with Applications, Information Processing and Management (all Scopus-indexed)

### Paper 2 — Local LLM Benchmarking Paper

**Title:** Benchmarking Quantized Local LLMs for Privacy-Preserving Legal Document Analysis

**Research Question:** How does inference quality degrade as model size decreases from 70B (cloud) to 8B to 3.8B (local) for legal NLP tasks, and what is the accuracy-latency-privacy tradeoff?

**Contribution:**
- Systematic comparison of local quantized models vs cloud models on legal reasoning
- RAM usage, latency, and accuracy benchmarks on consumer hardware
- Practical guidance for privacy-constrained deployments

**Target Journals:** IEEE Access, Computers (MDPI), Journal of Information Security and Applications (all Scopus-indexed)

---

## 17. Possible Improvements and Extensions

### Addition 1 — Confidence Scoring (Recommended)

The agent does not just say "contradiction found." It outputs a calibrated confidence score per finding.

**Research question:** Does confidence thresholding improve precision without significantly hurting recall for contradiction detection?

**Why this is original:** Nobody has applied confidence calibration specifically to legal contradiction detection. It also makes the system more trustworthy in practice — high confidence contradictions get flagged urgently, low confidence ones are flagged as "review recommended."

### Addition 2 — Cross-Document Comparison

Upload two versions of the same contract (original vs revised). System detects:
- New clauses added
- Clauses removed
- Whether new contradictions were introduced in the revision

**Practical use case:** Contract negotiation. You want to know what changed between drafts and whether the other party introduced anything problematic.

### Addition 3 — Retrieval Method Ablation

Dense retrieval (ChromaDB semantic search) vs sparse retrieval (BM25 keyword matching) vs hybrid (both combined).

**Why this matters for legal text:** Legal documents use precise terminology. "Termination" and "ending the agreement" mean the same thing to a human but a semantic search model may not connect them. BM25 would match "termination" exactly. Hybrid retrieval might outperform pure dense retrieval for legal text.

This is a clean, contained experiment that adds value without significantly increasing implementation complexity.

### Addition 4 — Domain Extension Beyond NDAs

Apply the same pipeline to:
- Employment contracts
- Terms and Conditions documents
- Vendor service agreements

Evaluate qualitatively since no ground truth labels exist for these (ContractNLI only covers NDAs). Show the system generalizes and discuss where it fails.

---

## 18. Honest Risk Assessment

### Risk 1 — It becomes a chatbot demo ❌
**Probability:** High if you spend 80% of time on the React frontend and 20% on evaluation.

**Mitigation:** Write the evaluation pipeline first, before the frontend. If the evaluation shows strong results, the frontend is just packaging. If you build the frontend first, you'll be tempted to call it done.

### Risk 2 — Contradiction F1 doesn't improve meaningfully ❌
**Probability:** Medium. If structure-aware chunking and agentic retrieval don't show clear improvement on ContractNLI, the core claim weakens.

**Mitigation:** Run a quick proof-of-concept evaluation on 20-30 contracts before committing to the full system. If the numbers are flat, revisit the agent design before building the full pipeline.

### Risk 3 — Research gap is challenged by reviewers ❌
**Probability:** Low-Medium. The specific combination of agentic retrieval + structure-aware chunking + ContractNLI evaluation is not fully claimed.

**Mitigation:** Be precise in how you frame the contribution. Do not say "we built a local legal AI." Say "we evaluate how structure-aware agentic retrieval affects contradiction detection F1 on ContractNLI under local compute constraints." Specificity protects you.

### Risk 4 — Local LLM quality is too poor for legal reasoning ❌
**Probability:** Medium for very small models (Phi-3 Mini). Low for Llama 3 8B with good prompting.

**Mitigation:** Test Llama 3 8B first. If quality is unacceptable, use Groq for the main experiments and present local inference as a deployment option with documented accuracy trade-offs. This is honest and still publishable.

### Risk 5 — ContractNLI only covers NDAs ❌
**Probability:** Guaranteed — this is a known dataset limitation, not a risk.

**Mitigation:** State it explicitly as a limitation in your paper. "Results are evaluated on NDA contracts; generalization to other contract types is left for future work." Reviewers accept honest scope limitations. They do not accept hidden ones.

---

## Summary

| Property | Value |
|---|---|
| Project Type | Research + System |
| Training Required | None |
| Primary Dataset | ContractNLI (607 NDAs, test split ~150) |
| Core Tech | LangGraph, ChromaDB, BGE embeddings, Llama 3 8B, FastAPI, React |
| Research Contribution | Agentic retrieval for contradiction detection under local compute constraints |
| Evaluation Method | F1/Precision/Recall on ContractNLI contradiction class |
| Number of Papers | 2 (system paper + local LLM benchmarking paper) |
| Implementation Difficulty | 7/10 (code is easy, evaluation rigor is hard) |
| Research Viability | High if executed with evaluation-first mindset |
| Biggest Risk | Becoming a demo instead of a research project |

---

*This document was prepared as an internal project reference. The research gap and novelty claims must be verified against the latest literature before submission.*
