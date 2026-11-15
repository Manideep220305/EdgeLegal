"""Compare fixed retrieval and model-directed contract lookup on development data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .agent.reference_review import ContractLookup, MAX_CALLS, TEXT_BUDGET, initial_evidence, review
from .parsing.clause_parser import PageBlock, segment_clauses

ROOT = Path(__file__).resolve().parents[2]
MODEL = "openai/gpt-oss-20b"
ARMS = ("top_k", "fixed_reference", "agent")


def cases():
    dataset = json.loads((ROOT / "backend/data/contractnli/dev.json").read_bytes())
    for doc in dataset["documents"]:
        lookup = ContractLookup(segment_clauses([PageBlock(doc["text"], 0)], split_oversized=True))
        for hid, gold in doc["annotation_sets"][0]["annotations"].items():
            hypothesis = dataset["labels"][hid]["hypothesis"]
            ranking = lookup.search(hypothesis) or list(lookup.by_id)
            spans = [doc["text"][start:end].strip() for start, end in
                     (doc["spans"][index] for index in gold["spans"])]
            yield lookup, ranking, spans, {
                "doc_id": doc["id"], "hyp_id": hid, "gold": gold["choice"],
                "hypothesis": hypothesis, "has_reference": bool(lookup.references(ranking[0])),
            }


def evidence_hits(excerpts, spans):
    return sum(any(span and span in text for text in excerpts) for span in spans)


def audit_hard_cases(items, local):
    audit = []
    candidates = []
    for lookup, ranking, spans, item in items:
        row = dict(item)
        key = row["doc_id"], row["hyp_id"]
        start = lookup.pack(ranking[:1], TEXT_BUDGET // MAX_CALLS)
        row["start_ids"] = list(start)
        row["start_hits"] = evidence_hits(start.values(), spans)
        row["span_count"] = len(spans)
        row["deberta"] = local[key]
        allowance = TEXT_BUDGET - 2 * sum(len(text.encode("utf-8")) for text in start.values())
        first = ranking[0]
        groups = {"reference": lookup.references(first), "neighbor": lookup.neighbors(first),
                  "search": ranking[1:]}
        row["missing_spans_by_lookup"] = {
            group: [n for n, span in enumerate(spans)
                    if evidence_hits(start.values(), [span]) == 0 and evidence_hits(
                        lookup.pack(ids[:2], allowance).values(), [span])]
            for group, ids in groups.items()
        } if start else {}
        if any(row["missing_spans_by_lookup"].get(tool) for tool in ("reference", "neighbor")):
            row["group"] = "structural_lookup"
        elif row["missing_spans_by_lookup"].get("search"):
            row["group"] = "search_lookup"
        elif start and (row["start_hits"] == len(spans)):
            row["group"] = "control"
        else:
            row["group"] = "other"
        audit.append(row)
        if row["group"] != "other":
            candidates.append((lookup, ranking, spans, row))
    return audit, candidates


def select_hard_cases(candidates, per_group):
    # Spread cases across contracts while retaining scarce contradiction examples.
    counters = Counter()
    used = set()
    selected = []
    for item in sorted(candidates, key=lambda x: (
        x[3]["group"] == "control", x[3]["gold"] != "Contradiction",
        not x[3]["missing_spans_by_lookup"].get("reference"),
        hashlib.sha256(f'42:{x[3]["doc_id"]}:{x[3]["hyp_id"]}'.encode()).digest(),
    )):
        row = item[3]
        key = row["group"], row["gold"]
        if counters[key] < per_group and row["doc_id"] not in used:
            counters[key] += 1
            used.add(row["doc_id"])
            selected.append(item)
    return selected


def evaluate_fixed_lookup(items, audit):
    """Measure evidence reached by the fixed policy with no verdict generation."""
    totals = {}
    for (lookup, ranking, spans, _), row in zip(items, audit, strict=True):
        result = review(lookup, ranking, row["hypothesis"], "fixed_matched",
                        lambda _: ({}, {}))
        # Count only passages actually exposed on a simulated call.
        seen = dict.fromkeys(cid for turn in result["trace"] for cid in turn["sent_clause_ids"])
        hits = evidence_hits([lookup.by_id[cid].text for cid in seen], spans)
        row["fixed_replay"] = {
            "hits": hits, "sent_clause_ids": list(seen),
            "calls": len(result["trace"]), "contract_bytes": result["contract_bytes"],
        }
        for group in ("all", row["group"], f'{row["group"]}/{row["gold"]}'):
            count = totals.setdefault(group, Counter())
            count.update({
                "cases": 1,
                "annotated": bool(spans),
                "first_any": row["start_hits"] > 0,
                "fixed_any": hits > 0,
                "fixed_all": bool(spans) and hits == len(spans),
                "added_evidence": hits > row["start_hits"],
                "added_evidence_on_deberta_error": hits > row["start_hits"] and row["deberta"] != row["gold"],
                "simulated_calls": len(result["trace"]),
                "excerpt_bytes": result["contract_bytes"],
            })
    return {group: dict(count) for group, count in totals.items()}


def summarize(rows):
    result = {}
    for arm in rows[0]["arms"]:
        pairs = [(r, r["arms"][arm]) for r in rows if arm in r["arms"]]
        result[arm] = {
            "cases": len(pairs),
            "annotated": sum(r["span_count"] > 0 for r, _ in pairs),
            "any_evidence": sum(o["evidence_hits"] > 0 for _, o in pairs),
            "all_evidence": sum(r["span_count"] > 0 and r["span_count"] == o["evidence_hits"] for r, o in pairs),
        }
        if "label" in pairs[0][1]:
            result[arm].update(
                correct=sum(r["gold"] == o["label"] for r, o in pairs),
                unresolved=sum(o["label"] is None for _, o in pairs),
                contradictions_correct=sum(r["gold"] == o["label"] == "Contradiction" for r, o in pairs),
                contradiction_cases=sum(r["gold"] == "Contradiction" for r, _ in pairs),
                calls=sum(len(o["trace"]) for _, o in pairs),
                contract_bytes=sum(o["contract_bytes"] for _, o in pairs),
                calls_with_lookup=sum("action" in t for _, o in pairs for t in o["trace"]),
                citation_cases_with_span_overlap=sum(o["citation_span_hits"] > 0 for _, o in pairs),
                false_mentions=sum(r["gold"] == "NotMentioned" and o["label"] in
                                   {"Entailment", "Contradiction"} for r, o in pairs),
            )
    return result


def provider(cache_path):
    from groq import BadRequestError, Groq, RateLimitError
    from .llm.groq_client import groq_client

    keys = list(dict.fromkeys(groq_client.keys))
    configured = os.getenv("EDGELEGAL_GROQ_ORGANIZATIONS", "").strip()
    groups = [group.strip() for group in configured.split(",")] if configured else []
    if groups and (len(groups) != len(keys) or not all(groups)):
        raise SystemExit("Provide one organization label per distinct Groq key in configuration order.")
    if not keys:
        raise SystemExit("Configure a Groq key before running the live pilot.")
    # Use one representative key per organization so shared capacity is counted once.
    representatives = {}
    for group, api_key in zip(groups or ["default"], keys):
        representatives.setdefault(group, api_key)
    clients = {group: Groq(api_key=api_key, max_retries=0, timeout=45)
               for group, api_key in representatives.items()}
    cache = (dict((r["key"], r) for r in map(json.loads, cache_path.read_bytes().splitlines()))
             if cache_path.exists() else {})
    quota_path = cache_path.with_name("groq_quota.json")
    stored = json.loads(quota_path.read_bytes()) if quota_path.exists() else {}
    first_group = next(iter(clients))
    states = stored.get("organizations", {first_group: stored})
    previous_first_group = stored.get("first_group", "default")
    if first_group != previous_first_group and previous_first_group in states:
        if first_group not in states:
            states[first_group] = states.pop(previous_first_group)

    def save_quota():
        write(quota_path, {"first_group": first_group, "organizations": states})

    def reset_seconds(headers, header):
        value = headers.get(header, "")
        parts = re.findall(r"(\d+(?:\.\d+)?)(ms|[hms])", value)
        return sum(float(amount) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
                   for amount, unit in parts) if parts else 60

    def available_at(quota, prompt):
        # Leave room for the chat template and a reasoning completion.
        reserve = len(prompt.encode("utf-8")) // 2 + 1024
        resume = quota.get("retry_at", 0)
        if quota.get("tokens_left", 8000) < reserve:
            resume = max(resume, quota.get("tokens_reset_at", 0))
        if quota.get("requests_left", 1000) < 1:
            resume = max(resume, quota.get("requests_reset_at", 0))
        return resume

    def generate(prompt):
        key = hashlib.sha256(f"{MODEL}\n{prompt}".encode()).hexdigest()
        if key not in cache:
            failure = None
            for attempt in range(len(clients) + 1):
                group = min(clients, key=lambda name: available_at(states.get(name, {}), prompt))
                quota = states.setdefault(group, {})
                resume = available_at(quota, prompt)
                delay = max(0, resume - time.time())
                if delay > 600:
                    raise SystemExit("Groq quota paused the pilot. Resume with the same output directory.")
                while delay > 0:
                    print(f"Waiting {delay:.0f}s for Groq budget.", flush=True)
                    time.sleep(min(delay, 30))
                    delay = resume - time.time()
                try:
                    raw = clients[group].chat.completions.with_raw_response.create(
                        model=MODEL, temperature=0, response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": prompt}],
                    )
                    break
                except RateLimitError as exc:
                    cooldown = groq_client._retry_after_seconds(exc, default=300)
                    quota["retry_at"] = time.time() + cooldown + 2
                    save_quota()
                    print(f"Groq reported a quota cooldown of {cooldown:.0f}s. Cache saved.", flush=True)
                    if attempt == len(clients):
                        raise SystemExit("Groq quota paused the pilot. Resume with the same output directory.") from None
                except BadRequestError as exc:
                    details = exc.body if isinstance(exc.body, dict) else {}
                    if details.get("error", details).get("code") != "json_validate_failed":
                        raise
                    failure = exc.response
                    break
            response = raw.parse() if failure is None else None
            headers = raw.headers if failure is None else failure.headers
            now = time.time()
            quota.clear()
            for resource in ("tokens", "requests"):
                remaining = headers.get(f"x-ratelimit-remaining-{resource}")
                if remaining is not None and remaining.isdecimal():
                    quota[f"{resource}_left"] = int(remaining)
                    quota[f"{resource}_reset_at"] = now + reset_seconds(
                        headers, f"x-ratelimit-reset-{resource}"
                    ) + 1
            payload = {}
            try:
                if response is not None:
                    payload = json.loads(response.choices[0].message.content or "")
            except (ValueError, IndexError):
                payload = {}
            record = {"key": key, "payload": payload if isinstance(payload, dict) else {},
                      "usage": response.usage.model_dump() if response is not None and response.usage else {}}
            if failure is not None:
                record["error"] = "json_validate_failed"
            with cache_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            cache[key] = record
            save_quota()
            if "tokens_left" not in quota:
                time.sleep(16)
        return cache[key]["payload"], cache[key]["usage"]

    return generate


def write(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/reference-rag-dev")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--hard", action="store_true", help="Audit or pilot cases with recoverable missing evidence")
    parser.add_argument("--per-group", type=int, default=2, help="Pilot cases per label and reference group")
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error("--per-group must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    items = list(cases())
    if args.live or args.hard:
        development = json.loads((ROOT / "results/document-nli/dev_predictions.json").read_bytes())
        local = {(row["document_id"], row["hypothesis_id"]): row["predicted"]
                 for row in development["rows"]}
    if args.hard:
        audit, candidates = audit_hard_cases(items, local)
        replay = evaluate_fixed_lookup(items, audit)
        items = select_hard_cases(candidates, args.per_group)
        counts = Counter(f'{row["group"]}/{row["gold"]}' for row in audit)
        write(args.out / "hard_audit.json", {
            "counts": dict(counts), "rows": audit, "selected": [item[3] for item in items],
            "fixed_replay": replay,
            "split": "development", "seed": 42, "cases_per_group_and_label": args.per_group,
            "contract_byte_cap": TEXT_BUDGET, "call_cap": MAX_CALLS,
        })
        if not args.live:
            print(json.dumps({"counts": dict(counts), "selected": len(items)}, indent=2))
            return
    elif args.live:
        counts = Counter()
        used = set()
        selected = []
        for item in sorted(items, key=lambda i: hashlib.sha256(
                f'42:{i[3]["doc_id"]}:{i[3]["hyp_id"]}'.encode()).digest()):
            row = item[3]
            group = row["has_reference"], row["gold"]
            if counts[group] < args.per_group and row["doc_id"] not in used:
                selected.append(item)
                counts[group] += 1
                used.add(row["doc_id"])
        items = selected
    if args.live:
        generate = provider(args.out / ("hard_responses.jsonl" if args.hard else "responses.jsonl"))
    rows = []
    path = args.out / ("hard_live.json" if args.hard else "live.json" if args.live else "retrieval.json")
    arms = ("top_k", "fixed_matched", "agent") if args.hard else ARMS
    for lookup, ranking, spans, row in items:
        row["span_count"] = len(spans)
        row["arms"] = {}
        if args.live:
            row["deberta"] = local[row["doc_id"], row["hyp_id"]]
        for arm in arms if args.live else arms[:2]:
            if args.live:
                output = review(lookup, ranking, row["hypothesis"], arm, generate)
                excerpts = [lookup.by_id[cid].text for cid in output["seen"]]
                output["citation_span_hits"] = sum(any(
                    span and (span in c["quote"] or c["quote"] in span) for c in output["citations"]
                ) for span in spans)
            else:
                packed = initial_evidence(lookup, ranking, arm)
                excerpts, output = list(packed.values()), {"seen": list(packed)}
            output["evidence_hits"] = evidence_hits(excerpts, spans)
            row["arms"][arm] = output
        rows.append(row)
        if args.live:
            write(path, {"model": MODEL, "contract_byte_cap": TEXT_BUDGET, "call_cap": MAX_CALLS,
                         "planned_cases": len(items), "summary": summarize(rows),
                         "deberta_correct": sum(r["deberta"] == r["gold"] for r in rows),
                         "rows": rows})
            print(f"Completed {len(rows)}/{len(items)} pilot cases.", flush=True)
    summary = summarize(rows)
    if not args.live:
        write(path, {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
