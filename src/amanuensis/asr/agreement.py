"""LocalAgreement-2 (Machacek et al., 2023): commit the prefix two consecutive decodes agree on."""
from amanuensis.asr.types import Word


def _key(w: Word) -> str:
    return w.text.lower().strip(" .,!?;:")


class LocalAgreement:
    def __init__(self) -> None:
        self.committed_end = 0.0
        self._prev: list[Word] = []

    def update(self, words: list[Word]) -> list[Word]:
        """Takes the latest full decode (absolute times); returns newly committed words."""
        new = [w for w in words if w.start >= self.committed_end - 0.1]
        n = 0
        while n < min(len(new), len(self._prev)) and _key(new[n]) == _key(self._prev[n]):
            n += 1
        committed = new[:n]
        if committed:
            self.committed_end = committed[-1].end
        self._prev = new[n:]
        return committed

    @property
    def tentative(self) -> list[Word]:
        return self._prev

    def flush(self) -> list[Word]:
        """Commit everything still tentative (end of utterance) and reset for the next one."""
        rest = self._prev
        self._prev = []
        self.committed_end = 0.0
        return rest
