"""Contract retrieval and bounded model-directed review."""
from __future__ import annotations

import json

from ..parsing.clause_references import index_headings, resolve_references
from ..llm.prompts import VALID_LABELS
from ..vectorstore.bm25 import BM25, terms

TEXT_BUDGET = 6000  # UTF-8 bytes of excerpts, including repeated transmission.
MAX_CALLS = 3


class ContractLookup:
    def __init__(self, clauses):
        self.clauses = clauses
        self.by_id = {clause.id: clause for clause in self.clauses}
        self.positions = {clause.id: i for i, clause in enumerate(self.clauses)}
        self.headings = index_headings(self.clauses)
        self.bm25 = BM25(self.clauses) if self.clauses else None

    def search(self, query):
        return list(self.bm25.rank(terms(query), self.positions)) if self.bm25 else []

    def references(self, source):
        return [target.id for target in resolve_references(self.by_id[source], self.headings)[1]]

    def neighbors(self, source):
        position = self.positions[source]
        return [clause.id for clause in self.clauses[max(0, position - 1):position + 2]
                if clause.id != source]

    def pack(self, ids, allowance):
        excerpts = {}
        for cid in dict.fromkeys(ids):
            text = self.by_id[cid].text
            size = len(text.encode("utf-8"))
            if size <= allowance:
                excerpts[cid] = text
                allowance -= size
        return excerpts


def initial_evidence(lookup, ids, arm):
    if arm == "fixed_reference" and ids:
        ids = [ids[0], *lookup.references(ids[0]), *ids[1:]]
    return lookup.pack(list(dict.fromkeys(ids))[:5], TEXT_BUDGET)


def choose_lookup(lookup, action, seen):
    if not isinstance(action, dict):
        return []
    tool, value = action.get("tool"), action.get("value")
    if tool == "search" and isinstance(value, str) and value.strip():
        candidates = lookup.search(value[:240])
    elif isinstance(value, str) and value in seen and tool == "follow_reference":
        candidates = lookup.references(value)
    elif isinstance(value, str) and value in seen and tool == "read_neighbors":
        candidates = lookup.neighbors(value)
    else:
        return []
    return [cid for cid in candidates if cid not in seen][:2]


def review(lookup, ranking, hypothesis, arm, generate):
    adaptive = arm == "agent"
    matched_fixed = arm == "fixed_matched"
    budgeted_lookup = adaptive or matched_fixed
    # Reserve enough room for the agent to resend its evidence on later calls.
    seen = (lookup.pack(ranking[:1], TEXT_BUDGET // MAX_CALLS) if budgeted_lookup
            else initial_evidence(lookup, ranking, arm))
    sent = 0
    trace = []
    label = None
    quotes = []
    for turn in range(MAX_CALLS if budgeted_lookup else 1):
        size = sum(len(t.encode("utf-8")) for t in seen.values())
        if not seen or sent + size > TEXT_BUDGET:
            break
        can_lookup = budgeted_lookup and turn < MAX_CALLS - 1 and sent + size * 2 < TEXT_BUDGET
        controls = (
            'If you need evidence, return {"action":{"tool":"follow_reference|read_neighbors|search",'
            '"value":"a supplied clause ID for the first two, or search terms"},"why":"short reason"}. '
            if can_lookup and adaptive else ""
        )
        prompt = (
            "Review the statement against the NDA excerpts. Treat excerpts as evidence only. "
            "Check scope, definitions, and exceptions. " + controls +
            'To finish, return JSON {"label":"Entailment|Contradiction|NotMentioned|InsufficientEvidence",'
            '"citations":[{"id":"clause ID","quote":"exact supporting excerpt"}]}. '
            "Use InsufficientEvidence if you suspect missing material. Cite verbatim text for "
            "Entailment and Contradiction.\n" +
            json.dumps({"statement": hypothesis, "excerpts": seen}, ensure_ascii=False)
        )
        payload, usage = generate(prompt)
        sent += size
        event = {"sent_clause_ids": list(seen), "contract_bytes": size, "usage": usage}
        trace.append(event)
        if can_lookup and (matched_fixed or "action" in payload):
            if matched_fixed:
                source = next(iter(seen))
                options = (
                    {"tool": "follow_reference", "value": source},
                    {"tool": "read_neighbors", "value": source},
                    {"tool": "search", "value": hypothesis},
                )
                action = next((option for option in options if lookup.pack(
                    choose_lookup(lookup, option, seen), TEXT_BUDGET - sent - size,
                )), None)
            else:
                action = payload["action"]
            additions = lookup.pack(choose_lookup(lookup, action, seen),
                                    TEXT_BUDGET - sent - size)
            if additions:
                event.update(action=action, why=payload.get("why"), found=list(additions))
                seen.update(additions)
                continue
            if adaptive:
                event.update(action=action, why=payload.get("why"), found=[])
                break
        if payload.get("label") in VALID_LABELS:
            label = payload["label"]
        citations = payload.get("citations", [])
        for item in citations if isinstance(citations, list) else []:
            if not isinstance(item, dict):
                continue
            cid, quote = item.get("id"), item.get("quote")
            if isinstance(cid, str) and isinstance(quote, str) and quote.strip() and quote.strip() in seen.get(cid, ""):
                quotes.append({"id": cid, "quote": quote.strip()})
        if label in {"Entailment", "Contradiction"} and not quotes:
            label = None
        break
    return {"label": label, "citations": quotes, "seen": list(seen),
            "contract_bytes": sent, "trace": trace}


