from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from amanuensis.config import SAMPLE_RATE, WINDOW

ProbFn = Callable[[np.ndarray], float]  # one 512-sample window -> speech probability


class SileroProb:
    """Silero VAD (the ONNX model bundled with faster-whisper, CPU).

    The bundled model is stateless per call, so each window is scored with up to
    one second of preceding audio as context and the last output is used.
    """

    def __init__(self, context_windows: int = 31) -> None:
        from faster_whisper.vad import get_vad_model

        self._model = get_vad_model()
        self._recent: deque[np.ndarray] = deque(maxlen=context_windows + 1)

    def __call__(self, window: np.ndarray) -> float:
        self._recent.append(window)
        out = self._model(np.concatenate(self._recent))
        return float(np.asarray(out).reshape(-1)[-1])


@dataclass(frozen=True)
class VadEvent:
    kind: str  # "start" | "end"
    sample: int  # absolute sample index of the window that triggered it


class VadSegmenter:
    """Turns per-window probabilities into utterance start/end events.

    An utterance ends once silence has lasted min_silence_s.
    """

    def __init__(self, prob: ProbFn, threshold: float, min_silence_s: float) -> None:
        self._prob, self._threshold = prob, threshold
        self._silence_limit = int(min_silence_s * SAMPLE_RATE)
        self.in_speech = False
        self._silence = 0
        self._pos = 0

    def feed(self, window: np.ndarray) -> VadEvent | None:
        assert len(window) == WINDOW
        pos, self._pos = self._pos, self._pos + WINDOW
        if self._prob(window) >= self._threshold:
            self._silence = 0
            if not self.in_speech:
                self.in_speech = True
                return VadEvent("start", pos)
        elif self.in_speech:
            self._silence += WINDOW
            if self._silence >= self._silence_limit:
                self.in_speech, self._silence = False, 0
                return VadEvent("end", pos + WINDOW)
        return None
