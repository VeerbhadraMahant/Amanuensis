"""Deterministic text normalizer. Spelling variants and lexicon come from
docs/romanization.md; nothing is hardcoded here."""
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(text: str, variants: dict[str, str] | None = None) -> str:
    """Lowercase, strip punctuation, collapse whitespace, map variants to canonical forms.

    `variants` maps a lowercase variant to its canonical (lowercase) spelling.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT.sub(" ", text)
    words = _SPACE.sub(" ", text).strip().split(" ")
    if variants:
        words = [variants.get(w, w) for w in words]
    return " ".join(w for w in words if w)


def apply_lexicon_casing(text: str, canonical_terms: list[str]) -> str:
    """Restore canonical casing for lexicon terms (e.g. 'pccoe' -> 'PCCOE')."""
    by_lower = {t.lower(): t for t in canonical_terms}
    return " ".join(by_lower.get(w.lower(), w) for w in text.split(" "))


_WORD = re.compile(r"[\w']+", re.UNICODE)


def canonicalize(text: str, variants: dict[str, str] | None = None, terms: list[str] | None = None) -> str:
    """Display form: map known variants to canonical spellings and apply lexicon casing,
    keeping the original punctuation and spacing. Identity when given no variants or terms."""
    variants = variants or {}
    casing = {t.lower(): t for t in terms or []}

    def fix(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        out = variants.get(lw, w)
        return casing.get(out.lower(), out)

    return _WORD.sub(fix, text)
