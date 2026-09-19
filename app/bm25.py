"""
Minimal, dependency-free BM25 implementation.

Written from scratch (rather than pulling in a third-party BM25 package) so the
core ranking logic is fully auditable in a small demo -- there's no black box
between "query in" and "score out". Standard Robertson/Sparck-Jones BM25 with
the usual k1/b smoothing parameters.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenizer. Deliberately simple: no
    stemming, no stopword removal -- BM25's IDF term already down-weights
    very common words, and keeping this transparent matters more than
    squeezing out marginal gains for a demo this size."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """BM25 over a fixed corpus of pre-tokenized documents."""

    def __init__(self, tokenized_docs: Sequence[list[str]], k1: float = 1.5, b: float = 0.75):
        if not tokenized_docs:
            raise ValueError("BM25 requires at least one document")

        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_docs)
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / self.n_docs

        self.term_freqs: list[Counter[str]] = [Counter(doc) for doc in tokenized_docs]

        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())

        # BM25 IDF with +1 smoothing so idf is always non-negative.
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score_doc(self, query_tokens: Sequence[str], doc_index: int) -> float:
        tf = self.term_freqs[doc_index]
        doc_len = self.doc_lens[doc_index]
        score = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self.idf.get(term, 0.0)
            numerator = f * (self.k1 + 1.0)
            denominator = f + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (numerator / denominator)
        return score

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        """Raw BM25 score for every document, in corpus order."""
        return [self.score_doc(query_tokens, i) for i in range(self.n_docs)]
