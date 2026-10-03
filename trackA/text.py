"""Character vocabulary for CTC. Index 0 is the blank."""
VOCAB = ["<blank>", " ", "'"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
_INDEX = {c: i for i, c in enumerate(VOCAB)}
BLANK = 0


def encode(text: str) -> list[int]:
    """Lowercase letters, space and apostrophe only; anything else is dropped."""
    return [_INDEX[c] for c in text.lower() if c in _INDEX and c != "<blank>"]


def decode(ids: list[int]) -> str:
    return "".join(VOCAB[i] for i in ids if i != BLANK)
