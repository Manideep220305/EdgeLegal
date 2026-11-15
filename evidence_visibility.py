"""Token-level visibility of annotated evidence in a clause pair."""

INPUT_LIMIT = 512


def premise_offsets(tokenizer, clause, hypothesis, *, truncate):
    encoded = tokenizer(
        clause, hypothesis, truncation=truncate, max_length=INPUT_LIMIT,
        return_offsets_mapping=True, verbose=False,
    )
    return [
        (start, end) for (start, end), sequence in
        zip(encoded["offset_mapping"], encoded.sequence_ids())
        if sequence == 0 and end > start
    ]


def evidence_state(tokenizer, row, spans):
    """Check tokens overlapping any complete, verbatim annotated span."""
    clause = row["top_clause"]
    ranges = []
    for span in spans:
        span = span.strip()
        if not span:
            continue
        start = clause.find(span)
        while start >= 0:
            ranges.append((start, start + len(span)))
            start = clause.find(span, start + 1)
    if not ranges:
        return "elsewhere" if row["evidence_mapped"] else "unmapped"

    if not row["truncated"]:
        return "visible"

    full = premise_offsets(tokenizer, clause, row["hypothesis"], truncate=False)
    retained = set(premise_offsets(tokenizer, clause, row["hypothesis"], truncate=True))
    partial = False
    for left, right in ranges:
        tokens = {offset for offset in full if offset[1] > left and offset[0] < right}
        if tokens and tokens <= retained:
            return "visible"
        partial |= bool(tokens & retained)
    return "partial" if partial else "truncated"


