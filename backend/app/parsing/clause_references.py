"""Resolve an internal clause reference against unique numbered headings."""

import re
from collections import defaultdict

REFERENCE = re.compile(
    r"\b(?P<kind>section|article|paragraph|clause)\s+"
    r"(?P<number>\d+[a-z]?(?:\.\d+[a-z]?)*(?:\([a-z0-9]+\))*)"
    r"(?![\w(]|\.\d)", re.IGNORECASE,
)
NUMBERED_HEADING = re.compile(r"\d+(?:\.\d+)*\.?")
OF_PHRASE = re.compile(r"\s+of\s+", re.IGNORECASE)
INTERNAL = re.compile(r"\s+of\s+(?:this|the)\s+agreement\b", re.IGNORECASE)
SELF = re.compile(r"\bthis\s+$", re.IGNORECASE)


def index_headings(clauses):
    headings = defaultdict(list)
    for clause in clauses:
        if NUMBERED_HEADING.fullmatch(clause.section_heading):
            headings[clause.section_heading.rstrip(".")].append(clause)
    return headings


def resolve_references(source, headings):
    references = []
    targets = {}
    for match in REFERENCE.finditer(source.text):
        number = match["number"].lower()
        suffix = source.text[match.end():]
        prefix = source.text[max(0, match.start() - 12):match.start()]
        matches = headings.get(number, [])
        if OF_PHRASE.match(suffix) and not INTERNAL.match(suffix):
            status = "qualified_reference_skipped"
        elif SELF.search(prefix):
            status = "self"
        elif not matches:
            status = "no_exact_heading"
        elif len(matches) > 1:
            status = "ambiguous_heading"
        elif matches[0].id == source.id:
            status = "self"
        else:
            status = "resolved"
            targets[matches[0].id] = matches[0]
        references.append({"text": match.group(), "status": status,
                           "heading_matches": [clause.id for clause in matches]})
    return references, list(targets.values())


