# Agentic / tool-use retrieval papers (grounding for the cross-reference retrieval feature)

The opt-in lookup agent can follow a numbered reference, read adjacent
clauses, or issue another keyword search. Its [development comparison](../../docs/EdgeLegal_Progress.md#reference-lookup-and-agentic-rag)
uses ordinary RAG and a deterministic lookup policy as controls.

## ReAct
Shunyu Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023.
- arXiv: https://arxiv.org/abs/2210.03629

The foundational "reasoning + tool-calling interleaved" agent pattern. Standard citation
for any LLM that autonomously decides to take an action (like an extra retrieval call)
mid-reasoning rather than following a fixed pipeline.

## Self-RAG
Akari Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through
Self-Reflection." ICLR 2024.
- arXiv: https://arxiv.org/abs/2310.11511

Trains a model to decide *when* retrieval is needed and to critique its own generations
against retrieved evidence. Relevant precedent for "the model decides whether it has
enough context," which is exactly the judgment call your cross-reference feature would
need to make.

## Adaptive-RAG
Already covered in note 02 (LLM cascades) — also directly relevant here as the
small-classifier-decides-retrieval-strategy pattern.

## How to use these
Cite ReAct for interleaving model decisions and tool calls, and Self-RAG
for deciding when more evidence is needed. EdgeLegal uses an existing LLM
in a bounded lookup loop. Use the matched pilot to assess its contribution.
