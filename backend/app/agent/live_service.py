"""
agent/live_service.py
──────────────────────
Wires the LangGraph triage agent (agent/graph.py) into the live FastAPI app.

This is a separate, additive path. It does not touch rag_engine.py,
document_processor.py, or the "legal_documents" ChromaDB collection those
use, nor experiment_runner.py's ephemeral "experiment_doc_*" collections —
the actual Config A/B/C measurement pipeline. Isolation from that pipeline
is deliberate: a bug in the live app must never be able to change a
thesis number. main.py no longer imports rag_engine.py at all; agent/
nodes.py, agent/state.py, and agent/graph.py are used exactly as they
already existed for the research pipeline — nothing in this file changes
their behavior or return shapes.

Public API:
  index_document(doc_id: str, pdf_path: str) -> int   # clauses indexed
  run_agent(doc_id: str, hypothesis: str) -> dict      # -> ReportResult shape
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

# agent/nodes.py and triage_model/*.py import as `backend.app.X` (written to
# be run via `python -m backend.app.X` from the project root). The live app
# runs as `app.X` via `uvicorn app.main:app` with cwd=backend/. Putting the
# project root on sys.path lets both import conventions resolve in the same
# process without changing either existing convention.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.app.agent.graph import build_graph  # noqa: E402
from backend.app.embeddings.embedder import Embedder  # noqa: E402
from backend.app.parsing.clause_parser import Clause, PageBlock, segment_clauses  # noqa: E402
from backend.app.vectorstore.chroma_client import (  # noqa: E402
    add_clauses,
    get_persistent_client,
    init_collection,
)
from app.logging_config import get_logger, log_exception, log_step  # noqa: E402

logger = get_logger(__name__)

# Kept in sync with backend/app/experiment_runner.py's FROZEN_THRESHOLD —
# duplicated as a plain constant rather than imported, since that module is
# a CLI script for offline experiments, not meant to be imported by a
# long-running web server process.
FROZEN_THRESHOLD = 0.95
LIVE_CHROMA_DIR = os.getenv(
    "EDGELEGAL_CHROMA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "live_chroma_db")
)


class NoIndexedClausesError(Exception):
    """The document has no indexed clauses available for analysis."""


class ReviewProviderError(Exception):
    """The escalation provider could not return a usable verdict."""


_setup_lock = threading.Lock()
_embedder: Optional[Embedder] = None
_graph = None
_chroma_client = None
_llm_client = None
_document_reviewer = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _setup_lock:
            if _embedder is None:
                e = Embedder()
                e.load()
                _embedder = e
    return _embedder


def _get_graph():
    global _graph
    if _graph is None:
        with _setup_lock:
            if _graph is None:
                _graph = build_graph()
    return _graph


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        with _setup_lock:
            if _chroma_client is None:
                _chroma_client = get_persistent_client(LIVE_CHROMA_DIR)
    return _chroma_client


def _get_llm_client() -> "GroqAgentClient":
    global _llm_client
    if _llm_client is None:
        with _setup_lock:
            if _llm_client is None:
                _llm_client = GroqAgentClient()
    return _llm_client


class GroqAgentClient:
    """
    Adapts the project's Groq client to the `.query_llm(clauses, hypothesis,
    triage_label="") -> LLMResult` interface call_llm_node expects — the
    same interface OllamaClient exposes. Reuses the exact same prompt
    builder and response parser as the Ollama path (llm/prompts.py,
    llm/ollama_client.py's _parse_llm_response) so only the HTTP transport
    differs; only the Ollama path is used for the paper's own local/private
    evaluation numbers.
    """

    def query_llm(self, clauses: list, hypothesis: str, triage_label: str = ""):
        from app.llm.groq_client import groq_client
        from backend.app.llm.ollama_client import _parse_llm_response
        from backend.app.llm.prompts import VALID_LABELS, build_reasoning_prompt

        prompt = build_reasoning_prompt(clauses, hypothesis, triage_label)
        try:
            if not groq_client.has_keys:
                raise RuntimeError("No Groq API key configured.")
            raw = groq_client.chat(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            parsed = json.loads(raw)
            label = parsed.get("label") if isinstance(parsed, dict) else None
            if not isinstance(label, str) or label.strip().casefold() not in {v.casefold() for v in VALID_LABELS}:
                raise ValueError("Invalid provider verdict.")
        except Exception as exc:
            raise ReviewProviderError("The review provider is unavailable or returned an invalid verdict. Try again.") from exc
        return _parse_llm_response(raw)


def _collection_for(doc_id: str):
    return init_collection(f"live_{doc_id}", client=_get_chroma_client())


def _uploaded_pdf(doc_id: str) -> Path:
    from uuid import UUID

    UUID(doc_id)
    upload_dir = Path(os.getenv("EDGELEGAL_UPLOAD_DIR", Path(__file__).resolve().parents[2] / "uploads"))
    return upload_dir / f"{doc_id}.pdf"


def _review_document(doc_id: str, hypothesis: str, checkpoint: str) -> dict:
    import pymupdf
    from backend.app.triage_model.document_infer import DocumentReviewer

    global _document_reviewer
    if _document_reviewer is None:
        with _setup_lock:
            if _document_reviewer is None:
                _document_reviewer = DocumentReviewer(Path(checkpoint))
    with pymupdf.open(_uploaded_pdf(doc_id)) as pdf:
        return _document_reviewer.predict([page.get_text() for page in pdf], hypothesis)


def index_document(doc_id: str, pdf_path: str) -> int:
    """
    Parse a PDF into clause-level units (parsing/clause_parser.py — the
    same structure-aware segmentation the research pipeline uses, not
    document_processor.py's naive block chunking) and index them for the
    live agent. Returns the number of clauses indexed (0 if the PDF had no
    extractable text, e.g. a scanned image with no text layer).
    """
    import pymupdf as fitz

    log_step(logger, "index_document.start", doc_id=doc_id, pdf_path=pdf_path)

    try:
        pages: list[PageBlock] = []
        # A failed path-based open can keep the file locked on Windows.
        with open(pdf_path, "rb") as upload:
            pdf_bytes = upload.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for i, page in enumerate(pdf):
                pages.append(PageBlock(text=page.get_text(), page_num=i + 1))
    except Exception:
        log_exception(logger, "PDF parsing failed", doc_id=doc_id, pdf_path=pdf_path)
        raise

    clauses = segment_clauses(pages)
    if not clauses:
        logger.warning("index_document: 0 clauses for doc_id=%s (%d pages) - "
                       "likely a scanned/image-only PDF.", doc_id, len(pages))
        return 0

    try:
        embedder = _get_embedder()
        collection = _collection_for(doc_id)
        add_clauses(collection, clauses, embedder.embed([c.text for c in clauses]))
    except Exception:
        log_exception(logger, "Indexing (embed/add_clauses) failed", doc_id=doc_id,
                      n_clauses=len(clauses))
        raise

    log_step(logger, "index_document.done", doc_id=doc_id, n_pages=len(pages),
             n_clauses=len(clauses))
    return len(clauses)


def _clause_citations(retrieved: list[Clause], excerpts: Optional[list] = None) -> list[dict]:
    """
    Build ReportCitation-shaped dicts {id, text, page, bbox} from the
    agent's retrieved clauses.

    - excerpts is None (classifier-accept path): cite the single rank-1
      clause the classifier actually saw, matching accept_node's own
      behavior of assembling the result from that clause.
    - excerpts is a list[str] (LLM path): the LLM's own verbatim-quoted
      excerpts (per the prompt's instructions) are more precise than "every
      retrieved clause", so match each back to its source clause for a page
      number. Excerpts without a unique source page are omitted.

    Bounding boxes are resolved from the source PDF after these citations
    are assembled.
    """
    if not retrieved:
        return []

    if excerpts is None:
        top = retrieved[0]
        return [{"id": top.id, "text": top.text, "page": top.page_num, "bbox": None}]

    def _norm(s: str) -> str:
        return " ".join(s.split())

    citations, seen_ids = [], set()
    for excerpt in excerpts:
        excerpt = excerpt.strip() if isinstance(excerpt, str) else ""
        if not excerpt:
            continue
        norm_excerpt = _norm(excerpt)
        matches = [c for c in retrieved if norm_excerpt in _norm(c.text)]
        if matches and len({c.page_num for c in matches}) == 1:
            match = matches[0]
            if match.id in seen_ids:
                continue
            seen_ids.add(match.id)
            citations.append({"id": match.id, "text": excerpt, "page": match.page_num, "bbox": None})
    return citations


def _locate_citations(doc_id, citations):
    """Use the source PDF to place a highlight for unambiguous matches."""
    import pymupdf as fitz

    def highlight(page, phrase, rectangles):
        if len(rectangles) == 1:
            return list(rectangles[0])
        page_text = " ".join(page.get_text().split())
        if page_text.casefold().count(phrase.casefold()) != 1:
            return None
        bounds = rectangles[0]
        for rectangle in rectangles[1:]:
            bounds |= rectangle
        return list(bounds)

    try:
        path = _uploaded_pdf(doc_id)
    except ValueError:
        return citations
    if not citations or not path.is_file():
        return citations
    try:
        pdf = fitz.open(path)
    except (FileNotFoundError, fitz.FileDataError):
        return citations
    with pdf:
        for citation in citations:
            phrase = " ".join(citation["text"].split())
            if not phrase:
                continue
            page_index = citation["page"] - 1
            if not 0 <= page_index < pdf.page_count:
                continue
            page = pdf[page_index]
            matches = page.search_for(phrase)
            if matches:
                citation["bbox"] = highlight(page, phrase, matches)
                continue
            matching_pages = [(i, page, matches) for i, page in enumerate(pdf)
                              if i != page_index and (matches := page.search_for(phrase))]
            if len(matching_pages) == 1:
                index, page, matches = matching_pages[0]
                bounds = highlight(page, phrase, matches)
                if bounds is not None:
                    citation.update(page=index + 1, bbox=bounds)
    return citations


def _run_reference_review(doc_id: str, hypothesis: str) -> dict:
    import pymupdf
    from groq import APIError, Groq
    from backend.app.agent.reference_review import ContractLookup, review
    from app.llm.groq_client import groq_client

    with pymupdf.open(_uploaded_pdf(doc_id)) as pdf:
        pages = [PageBlock(page.get_text(), i + 1) for i, page in enumerate(pdf)]
    lookup = ContractLookup(segment_clauses(pages, split_oversized=True))
    ranking = lookup.search(hypothesis) or list(lookup.by_id)
    if not ranking:
        raise NoIndexedClausesError("No readable text is available in the uploaded PDF.")
    if not groq_client.has_keys:
        raise ReviewProviderError("The review provider is unavailable. Try again.")
    with Groq(api_key=groq_client.get_current_key(), max_retries=0, timeout=45) as client:
        def generate(prompt):
            response = client.chat.completions.create(
                model=groq_client.default_model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            payload = json.loads(response.choices[0].message.content or "")
            return payload if isinstance(payload, dict) else {}, {}

        try:
            result = review(lookup, ranking, hypothesis, "agent", generate)
        except (APIError, ValueError, IndexError) as exc:
            raise ReviewProviderError("The review provider could not complete the review. Try again.") from exc
    if result["label"] is None:
        raise ReviewProviderError("The agent did not find enough usable evidence. Try a narrower statement.")
    citations = [{"id": item["id"], "text": item["quote"],
                  "page": lookup.by_id[item["id"]].page_num, "bbox": None}
                 for item in result["citations"]]
    actions = {"follow_reference": "Opened referenced clauses", "read_neighbors": "Read nearby clauses",
               "search": "Searched again"}
    steps = []
    for event in result["trace"]:
        if event.get("found"):
            pages = ", ".join(str(page) for page in sorted({
                lookup.by_id[cid].page_num for cid in event["found"]
            }))
            steps.append(f"{actions[event['action']['tool']]} on PDF page(s) {pages}.")
    reasoning = " ".join(["The agent reviewed the retrieved contract passages.", *steps])
    return {"label": result["label"], "reasoning": reasoning, "source": "llm",
            "cited_clauses": _locate_citations(doc_id, citations)}


def run_agent(doc_id: str, hypothesis: str) -> dict:
    """
    Run the confidence-routed triage agent for one hypothesis against an
    already-indexed document. Returns a dict matching schemas.ReportResult.
    """
    log_step(logger, "run_agent.start", doc_id=doc_id, hypothesis=hypothesis[:80])

    collection = _collection_for(doc_id)
    if collection.count() == 0:
        logger.warning("run_agent: doc_id=%s has no indexed clauses.", doc_id)
        raise NoIndexedClausesError(
            "No indexed text is available for this document. Upload a PDF with selectable text and try again."
        )

    if os.getenv("EDGELEGAL_AGENTIC_RAG") == "1":
        return _run_reference_review(doc_id, hypothesis)

    try:
        retrieval_mode = os.getenv("EDGELEGAL_RETRIEVAL_MODE", "bge").strip().lower()
        if retrieval_mode not in {"bge", "bm25"}:
            raise ValueError("EDGELEGAL_RETRIEVAL_MODE must be bge or bm25.")
        document_checkpoint = os.getenv("EDGELEGAL_DOCUMENT_CHECKPOINT")
        threshold = float(os.environ["EDGELEGAL_DOCUMENT_THRESHOLD"]) if document_checkpoint else FROZEN_THRESHOLD
        if not 0 <= threshold <= 1:
            raise ValueError("EDGELEGAL_DOCUMENT_THRESHOLD must be between zero and one.")
        state = {
            "document_id": doc_id,
            "hypothesis": hypothesis,
            "collection": collection,
            "embedder": _get_embedder() if retrieval_mode == "bge" else None,
            "retrieval_mode": retrieval_mode,
            "llm_client": _get_llm_client(),
            "threshold": threshold,
            "skip_triage": False,
        }
        if document_checkpoint:
            state["document_prediction"] = _review_document(doc_id, hypothesis, document_checkpoint)
        final_state = _get_graph().invoke(state)
    except Exception:
        log_exception(logger, "Agent graph invoke() failed", doc_id=doc_id,
                      hypothesis=hypothesis[:80])
        raise

    retrieved = final_state.get("retrieved_clauses") or []
    source = final_state.get("source_engine")
    label = final_state.get("final_label") or "NotMentioned"
    triage_confidence = final_state.get("triage_confidence")
    routing_decision = final_state.get("routing_decision")

    if source == "classifier":
        pct = f"{triage_confidence * 100:.0f}%" if triage_confidence is not None else "high"
        reasoning = (
            f"Resolved directly by the triage classifier ({pct} confidence, at or "
            f"above the {threshold * 100:.0f}% threshold) — no LLM call needed."
        )
        citations = (state["document_prediction"]["citations"] if document_checkpoint
                     else _clause_citations(retrieved))
    else:
        reasoning = final_state.get("reasoning_text") or "No reasoning returned."
        citations = _clause_citations(retrieved, final_state.get("cited_clauses"))

    triage_detail = None
    if routing_decision is not None:
        triage_detail = {
            "label": final_state.get("triage_label"),
            "confidence": triage_confidence,
            "threshold": threshold,
            "decision": routing_decision,
        }

    log_step(logger, "run_agent.done", doc_id=doc_id, label=label, source=source)
    return {
        "label": label,
        "reasoning": reasoning,
        "cited_clauses": _locate_citations(doc_id, citations),
        "source": source,
        # Only meaningful on the classifier path — None here (rather than the
        # sub-threshold triage score) matches ReportResult's documented
        # contract and what the frontend's confidence bar expects to gate on.
        # The pre-routing triage score is still available either way, via
        # `triage`, for the "how this was decided" panel.
        "confidence": triage_confidence if source == "classifier" else None,
        "triage": triage_detail,
    }
