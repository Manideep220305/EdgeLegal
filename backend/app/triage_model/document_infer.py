"""Inference for a frozen document-level DeBERTa checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..parsing.clause_parser import Clause, PageBlock, segment_clauses
from ..vectorstore.bm25 import BM25, terms
from .model import ID2LABEL, TriageModel, _best_device
from .train_document import DocumentNLI, encode_document


class DocumentReviewer:
    def __init__(self, path: Path):
        manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
        self.settings = manifest["settings"]
        if self.settings["smoke"]:
            raise ValueError("A smoke-test checkpoint cannot be used for document review.")
        self.device = _best_device()
        wrapper = TriageModel().load(device=self.device)
        self.tokenizer = wrapper._tokenizer
        self.model = DocumentNLI(wrapper._model, self.settings["window_batch"]).to(self.device)
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, pages: list[str], hypothesis: str) -> dict:
        document = "\n\n".join(pages)
        windows, bounds = encode_document(self.tokenizer, document, hypothesis,
                                          self.settings["max_length"], self.settings["stride"])
        # Older tokenizers can report overflow but silently discard later windows.
        if bounds[-1][1] != len(document.rstrip()):
            offsets = self.tokenizer(
                document, add_special_tokens=False, return_offsets_mapping=True, verbose=False,
            )["offset_mapping"]
            if offsets and bounds[-1][1] != offsets[-1][1]:
                raise RuntimeError(
                    "Tokenizer windows did not cover the whole contract. Install the pinned backend dependencies."
                )
        logits, evidence = self.model(windows, self.tokenizer, self.device)
        score, label = logits.softmax(dim=-1).max(dim=-1)
        verdict = ID2LABEL[label.item()]
        start, end = bounds[evidence.argmax().item()]
        citations = []
        if verdict == "NotMentioned":
            return {"label": verdict, "confidence": score.item(), "citations": citations}
        candidates = []
        offset = 0
        for page, text in enumerate(pages, 1):
            left, right = max(start, offset), min(end, offset + len(text))
            excerpt = document[left:right].strip()
            if left < right and excerpt:
                for clause in segment_clauses([PageBlock(excerpt, page)]):
                    candidates.append(Clause(f"window_{len(candidates)}", clause.text, page, clause.section_heading))
            offset += len(text) + 2
        if candidates:
            positions = {clause.id: index for index, clause in enumerate(candidates)}
            ranked = BM25(candidates).rank(terms(hypothesis), positions)
            best = min(candidates, key=lambda clause: ranked.get(clause.id, (float("inf"), 0))[0])
            citations.append({"id": best.id, "text": best.text, "page": best.page_num, "bbox": None})
        return {"label": verdict, "confidence": score.item(), "citations": citations}
