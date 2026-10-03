import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from amanuensis.asr.agreement import LocalAgreement
from amanuensis.asr.latency import LatencyStats
from amanuensis.asr.types import Engine, Word
from amanuensis.audio.vad import VadSegmenter
from amanuensis.config import SAMPLE_RATE, WINDOW, StreamingConfig
from amanuensis.logging import log

PREROLL = 16 * WINDOW  # ~0.5 s kept before speech onset
SENTENCE_END = (".", "?", "!")


@dataclass(frozen=True)
class Update:
    committed: str  # newly committed text (empty if none)
    tentative: str  # current uncommitted text, overlay only
    final: bool  # True when an utterance just closed


def _text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


class Streamer:
    """Feed 512-sample windows; emits Updates through `sink`. Time is the audio clock, plus
    real decode time, so replay mode and live mode measure latency the same way."""

    def __init__(
        self,
        engine: Engine,
        segmenter: VadSegmenter,
        cfg: StreamingConfig,
        sink: Callable[[Update], None],
        stats: LatencyStats | None = None,
        on_utterance: Callable[[np.ndarray, str, float | None], None] | None = None,
    ) -> None:
        self._engine, self._seg, self._cfg, self._sink = engine, segmenter, cfg, sink
        self._on_utterance = on_utterance  # (audio, raw text, median commit latency ms) per closed utterance
        self.stats = stats or LatencyStats()
        self.utterances: list[str] = []
        self._agree = LocalAgreement()
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0  # absolute sample index of _buf[0]
        self._total = 0  # samples fed
        self._active = False
        self._speech_start = 0
        self._last_decode = 0
        self._interval = int(cfg.redecode_interval_s * SAMPLE_RATE)
        self._behind = 0
        self._context = ""  # committed text that already left the buffer: the decoding prompt
        self._buf_words: list[Word] = []  # committed words whose audio is still in the buffer
        self._utt_words: list[Word] = []
        self._first_partial_done = False
        self._utt_audio: list[np.ndarray] = []
        self._utt_lat: list[float] = []

    def feed(self, window: np.ndarray) -> None:
        event = self._seg.feed(window)
        self._buf = np.concatenate([self._buf, window])
        self._total += WINDOW
        if event and event.kind == "start":
            self._active, self._speech_start = True, event.sample
            self._last_decode, self._first_partial_done = self._total, False
            self._utt_audio, self._utt_lat = [self._buf.copy()], []  # buffer holds the pre-roll + onset
        elif self._active:
            self._utt_audio.append(window)
        if self._active:
            if event and event.kind == "end":
                self._finalize()
            elif self._total - self._last_decode >= self._interval:
                self._cycle()
        elif len(self._buf) > PREROLL:
            self._drop_before(self._total - PREROLL)

    def close(self) -> None:
        if self._active:
            self._finalize()

    def _decode(self) -> tuple[list[Word], float]:
        t0 = time.perf_counter()
        words = self._engine.transcribe(self._buf, self._context[-self._cfg.prompt_chars :] or None)
        dt = time.perf_counter() - t0
        self.stats.decode_s.append(dt)
        offset = self._buf_start / SAMPLE_RATE
        return [Word(w.text, w.start + offset, w.end + offset) for w in words], dt

    def _cycle(self) -> None:
        words, dt = self._decode()
        self._last_decode = self._total
        self._track_lag(dt)
        now = self._total / SAMPLE_RATE
        committed = self._agree.update(words)
        self._record(committed, now, dt)
        if words and not self._first_partial_done:
            self._first_partial_done = True
            self.stats.first_partial_s.append(now - self._speech_start / SAMPLE_RATE + dt)
        self._sink(Update(_text(committed), _text(self._agree.tentative), final=False))
        self._trim(committed)

    def _finalize(self) -> None:
        words, dt = self._decode()
        now = self._total / SAMPLE_RATE
        rest = self._agree.update(words)
        rest += self._agree.flush()
        self._record(rest, now, dt)
        text = _text(self._utt_words)
        log("utterance", start_s=self._speech_start / SAMPLE_RATE, end_s=now, text=text)
        if text:
            self.utterances.append(text)
            if self._on_utterance:  # empty hypotheses (noise) are not logged
                lat = float(np.median(self._utt_lat)) * 1000 if self._utt_lat else None
                self._on_utterance(np.concatenate(self._utt_audio), text, lat)
        self._sink(Update(_text(rest), "", final=True))
        self._utt_words, self._active = [], False
        self._drop_before(self._total - PREROLL)
        self._retire(len(self._buf_words))  # utterance over: all its text becomes prompt context

    def _record(self, committed: list[Word], now: float, dt: float) -> None:
        for w in committed:
            self.stats.commit_s.append(now - w.end + dt)
            self._utt_lat.append(now - w.end + dt)
        self._utt_words += committed
        self._buf_words += committed

    def _trim(self, committed: list[Word]) -> None:
        too_long = len(self._buf) > self._cfg.max_buffer_s * SAMPLE_RATE
        sentence = bool(committed) and committed[-1].text.endswith(SENTENCE_END)
        if (too_long or sentence) and self._agree.committed_end:
            self._drop_before(int(self._agree.committed_end * SAMPLE_RATE))

    def _drop_before(self, abs_sample: int) -> None:
        n = max(0, min(abs_sample - self._buf_start, len(self._buf)))
        self._buf, self._buf_start = self._buf[n:], self._buf_start + n
        gone = sum(1 for w in self._buf_words if w.end <= self._buf_start / SAMPLE_RATE)
        self._retire(gone)

    def _retire(self, n: int) -> None:
        """Words whose audio is gone become context. Text still in the buffer must NOT be in
        the prompt, or Whisper treats it as already said and skips it."""
        retired, self._buf_words = self._buf_words[:n], self._buf_words[n:]
        self._context = (self._context + " " + _text(retired)).strip()[-self._cfg.prompt_chars :]

    def _track_lag(self, decode_s: float) -> None:
        self._behind = self._behind + 1 if decode_s > self._interval / SAMPLE_RATE else 0
        if self._behind >= 3:
            self._interval = min(int(self._interval * 1.5), 2 * SAMPLE_RATE)
            self._behind = 0
            log("decode_behind_realtime", new_interval_s=self._interval / SAMPLE_RATE)
