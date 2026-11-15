"""Per-contract lexical ranking shared by live and offline retrieval."""
import math
import re
from collections import Counter

BM25_K1 = 1.2
BM25_B = 0.75
TOKENS = re.compile(r"[^\W_]+")


def terms(text):
    return Counter(TOKENS.findall(text.casefold()))


class BM25:
    def __init__(self, clauses):
        self.frequencies = {clause.id: terms(clause.text) for clause in clauses}
        self.lengths = {clause_id: sum(counts.values()) for clause_id, counts in self.frequencies.items()}
        mean_length = sum(self.lengths.values()) / len(self.lengths) or 1
        self.normalizers = {
            clause_id: BM25_K1 * (1 - BM25_B + BM25_B * length / mean_length)
            for clause_id, length in self.lengths.items()
        }
        doc_frequency = Counter(term for counts in self.frequencies.values() for term in counts)
        self.idf = {
            term: math.log1p((len(clauses) - count + 0.5) / (count + 0.5))
            for term, count in doc_frequency.items()
        }

    def rank(self, query_terms, tie_positions):
        scores = {}
        for clause_id, counts in self.frequencies.items():
            scores[clause_id] = sum(
                self.idf[term] * counts[term] * (BM25_K1 + 1)
                / (counts[term] + self.normalizers[clause_id])
                for term in query_terms if term in counts
            )
        # Zero-overlap clauses receive no lexical rank.
        ranked = sorted((cid for cid, score in scores.items() if score > 0),
                        key=lambda cid: (-scores[cid], tie_positions[cid]))
        return {cid: (rank, scores[cid]) for rank, cid in enumerate(ranked, 1)}
