"""Spelling check against docs/romanization.md (via its derived variant map)."""
import re
from dataclasses import dataclass

_WORD = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True)
class SpellingFlag:
    word: str
    suggestion: str
    start: int
    end: int


def check_spelling(text: str, variants: dict[str, str]) -> list[SpellingFlag]:
    """Flag words that are known non-canonical variants. Unknown words are never flagged."""
    flags = []
    for m in _WORD.finditer(text):
        canonical = variants.get(m.group(0).lower())
        if canonical is not None and canonical != m.group(0):  # casing is part of the canonical form
            flags.append(SpellingFlag(m.group(0), canonical, m.start(), m.end()))
    return flags
