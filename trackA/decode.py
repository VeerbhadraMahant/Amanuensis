"""CTC decoding: greedy, then a small prefix beam search (no language model)."""
import math
from collections import defaultdict

import numpy as np

from trackA.text import BLANK


def greedy(log_probs: np.ndarray) -> list[int]:
    """(T, V) log-probabilities -> best label per frame, collapse repeats, drop blanks."""
    best = log_probs.argmax(axis=1)
    out, prev = [], BLANK
    for b in best:
        if b != prev and b != BLANK:
            out.append(int(b))
        prev = b
    return out


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search(log_probs: np.ndarray, beam: int = 8) -> list[int]:
    """CTC prefix beam search. Tracks, per prefix, the probability of ending in a blank and in a non-blank,
    so different alignments of the same text are merged instead of competing."""
    neg_inf = -math.inf
    beams = {(): (0.0, neg_inf)}  # prefix -> (log p ending in blank, log p ending in non-blank)
    for t in range(log_probs.shape[0]):
        nxt: dict[tuple, list[float]] = defaultdict(lambda: [neg_inf, neg_inf])
        for prefix, (p_b, p_nb) in beams.items():
            p_total = _logsumexp(p_b, p_nb)
            for c in np.argsort(log_probs[t])[::-1][:beam]:  # only the most likely symbols can matter
                p = float(log_probs[t, c])
                if c == BLANK:
                    nxt[prefix][0] = _logsumexp(nxt[prefix][0], p_total + p)
                    continue
                new = prefix + (int(c),)
                if prefix and prefix[-1] == c:
                    # a repeat only extends the prefix if a blank separated the two; otherwise it stays the same prefix
                    nxt[new][1] = _logsumexp(nxt[new][1], p_b + p)
                    nxt[prefix][1] = _logsumexp(nxt[prefix][1], p_nb + p)
                else:
                    nxt[new][1] = _logsumexp(nxt[new][1], p_total + p)
        beams = {k: tuple(v) for k, v in sorted(nxt.items(), key=lambda kv: _logsumexp(*kv[1]), reverse=True)[:beam]}
    return list(max(beams.items(), key=lambda kv: _logsumexp(*kv[1]))[0])
