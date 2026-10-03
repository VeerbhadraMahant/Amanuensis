import jiwer

from amanuensis.text.normalizer import normalize


def wer(refs: list[str], hyps: list[str]) -> float:
    return jiwer.wer(refs, hyps)


def cer(refs: list[str], hyps: list[str]) -> float:
    return jiwer.cer(refs, hyps)


def normalized_wer(refs: list[str], hyps: list[str], variants: dict[str, str] | None = None) -> float:
    return wer([normalize(r, variants) for r in refs], [normalize(h, variants) for h in hyps])


def term_accuracy(refs: list[str], hyps: list[str], terms: list[str]) -> float | None:
    """Share of lexicon-term occurrences in the references that appear in the matching hypothesis.

    Returns None when no term occurs in the references.
    """
    terms_l = {t.lower() for t in terms}
    hit = total = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        hyp_words = normalize(hyp).split(" ")
        for w in normalize(ref).split(" "):
            if w in terms_l:
                total += 1
                if w in hyp_words:
                    hyp_words.remove(w)
                    hit += 1
    return hit / total if total else None
