"""Streaming pipeline tests with a scripted engine and VAD (no models, no mic).

Audio is a ramp whose sample value encodes its absolute index, so the fake engine and
fake VAD can tell which part of the stream they were handed even after buffer trimming.
"""
import numpy as np
import pytest

from amanuensis.asr.agreement import LocalAgreement
from amanuensis.asr.replay import replay_audio
from amanuensis.asr.streaming import Streamer, Update
from amanuensis.asr.types import Word
from amanuensis.audio.ring_buffer import RingBuffer
from amanuensis.audio.vad import VadSegmenter
from amanuensis.config import SAMPLE_RATE, WINDOW, StreamingConfig

SCALE = 1e7


def cfg(**kw) -> StreamingConfig:
    base = dict(
        model="x", device="cpu", compute_type="int8", cpu_fallback_model="x", language="en",
        vad_threshold=0.5, min_silence_s=0.6, redecode_interval_s=0.7, max_buffer_s=20.0, prompt_chars=200,
    )
    return StreamingConfig(**{**base, **kw})


def abs_index(x: np.ndarray) -> int:
    return round(float(x[0]) * SCALE)


class FakeEngine:
    """Hears every scripted word that ends inside the audio. Words ending in the last
    0.5 s are unstable (wrong text) so only agreement can commit the right ones."""

    def __init__(self, words: list[Word]) -> None:
        self.words, self.prompts, self.calls = words, [], []

    def transcribe(self, audio, prompt):
        self.prompts.append(prompt)
        a0 = abs_index(audio) / SAMPLE_RATE
        self.calls.append((a0, prompt))
        a1 = a0 + len(audio) / SAMPLE_RATE
        out = []
        for w in self.words:
            if w.start >= a0 and w.end <= a1:
                text = "tmp" if w.end > a1 - 0.5 else w.text
                out.append(Word(text, w.start - a0, w.end - a0))
        return out


def fake_vad(words: list[Word]):
    def prob(window):
        t = abs_index(window) / SAMPLE_RATE
        return 1.0 if any(w.start - 0.05 <= t <= w.end for w in words) else 0.0

    return prob


def run(words: list[Word], seconds: float, c: StreamingConfig | None = None):
    c = c or cfg()
    engine = FakeEngine(words)
    updates: list[Update] = []
    streamer = Streamer(engine, VadSegmenter(fake_vad(words), c.vad_threshold, c.min_silence_s), c, updates.append)
    audio = (np.arange(int(seconds * SAMPLE_RATE) // WINDOW * WINDOW) / SCALE).astype(np.float32)
    max_buf = 0
    for i in range(0, len(audio), WINDOW):
        streamer.feed(audio[i : i + WINDOW])
        max_buf = max(max_buf, len(streamer._buf))
    streamer.close()
    return streamer, engine, updates, max_buf


def speech(texts: list[str], t0: float, step: float = 0.4) -> list[Word]:
    return [Word(t, t0 + i * step, t0 + i * step + 0.3) for i, t in enumerate(texts)]


def test_agreement_commits_only_stable_prefix():
    la = LocalAgreement()
    w = lambda *ts: [Word(t, i * 0.5, i * 0.5 + 0.4) for i, t in enumerate(ts)]
    assert la.update(w("hello", "wor")) == []
    assert [x.text for x in la.update(w("hello", "world", "x"))] == ["hello"]
    assert [x.text for x in la.tentative] == ["world", "x"]
    assert [x.text for x in la.update(w("hello", "world", "x"))] == ["world", "x"]


def test_agreement_flush_commits_rest_and_resets():
    la = LocalAgreement()
    la.update([Word("a", 0, 0.3)])
    assert [x.text for x in la.flush()] == ["a"]
    assert la.committed_end == 0.0 and la.tentative == []


def test_single_utterance_text_is_the_true_words():
    words = speech(["hello", "world", "how", "are", "you"], 1.0)
    s, _, updates, _ = run(words, 5.0)
    assert s.utterances == ["hello world how are you"]
    assert updates[-1].final
    assert "tmp" not in " ".join(u.committed for u in updates)


def test_two_utterances_split_by_silence():
    words = speech(["one", "two"], 1.0) + speech(["three", "four"], 4.0)
    s, _, _, _ = run(words, 7.0)
    assert s.utterances == ["one two", "three four"]


def test_latency_recorded_and_bounded_in_audio_time():
    words = speech(["a", "b", "c", "d", "e", "f"], 1.0)
    s, _, _, _ = run(words, 6.0)
    summ = s.stats.summary()
    assert summ["first_partial_s"]["n"] == 1 and summ["commit_s"]["n"] == 6
    assert summ["first_partial_s"]["p50"] < 2.0
    assert summ["decode_s"]["n"] > 0


def test_prompt_carries_recent_committed_text():
    words = speech(["one", "two"], 1.0) + speech(["three", "four"], 4.0)
    _, engine, _, _ = run(words, 7.0)
    assert engine.prompts[0] is None
    assert any(p and "one two" in p for p in engine.prompts)


def test_prompt_never_contains_text_still_in_the_buffer():
    words = speech([f"w{i}" for i in range(40)], 1.0)
    _, engine, _, _ = run(words, 19.5, cfg(max_buffer_s=4.0))
    for a0, prompt in engine.calls:
        in_buffer = {w.text for w in words if w.end > a0}
        assert not (set((prompt or "").split()) & in_buffer)
    assert any(p for _, p in engine.calls)  # trimmed text does reach the prompt


def test_buffer_trimmed_during_long_speech():
    words = speech([f"w{i}" for i in range(40)], 1.0)  # 16 s of speech
    s, _, _, max_buf = run(words, 19.5, cfg(max_buffer_s=4.0))
    assert s.utterances == [" ".join(f"w{i}" for i in range(40))]
    assert max_buf < 8 * SAMPLE_RATE


def test_no_speech_no_decode_and_bounded_buffer():
    _, engine, updates, max_buf = run([], 5.0)
    assert engine.prompts == [] and updates == []
    assert max_buf <= 16 * WINDOW + WINDOW


def test_slow_decoding_widens_interval(monkeypatch):
    words = speech([f"w{i}" for i in range(30)], 1.0)
    c = cfg(redecode_interval_s=0.7)
    engine = FakeEngine(words)
    seg = VadSegmenter(fake_vad(words), c.vad_threshold, c.min_silence_s)
    s = Streamer(engine, seg, c, lambda u: None)
    start = s._interval
    for _ in range(3):
        s._track_lag(5.0)
    assert s._interval > start


def test_ring_buffer_roundtrip_and_overrun():
    rb = RingBuffer(10)
    rb.write(np.arange(6, dtype=np.float32))
    assert rb.read(7) is None
    assert rb.read(4).tolist() == [0, 1, 2, 3]
    rb.write(np.arange(100, 108, dtype=np.float32))  # 2 held + 8 = 10: wraps, no drop
    assert rb.overruns == 0
    rb.write(np.arange(200, 203, dtype=np.float32))  # 10 held + 3 -> oldest 3 dropped
    assert rb.overruns == 3
    assert rb.read(10).tolist() == [101, 102, 103, 104, 105, 106, 107, 200, 201, 202]


@pytest.mark.parametrize("silence_s,ends", [(0.3, 0), (0.7, 1)])
def test_vad_segmenter_needs_min_silence_to_end(silence_s, ends):
    probs = [1.0] * 5 + [0.0] * int(silence_s * SAMPLE_RATE / WINDOW)
    it = iter(probs)
    seg = VadSegmenter(lambda w: next(it), 0.5, 0.6)
    events = [seg.feed(np.zeros(WINDOW, np.float32)) for _ in probs]
    kinds = [e.kind for e in events if e]
    assert kinds.count("start") == 1 and kinds.count("end") == ends


def test_replay_audio_closes_trailing_utterance_and_reports_latency():
    words = speech(["alpha", "beta", "gamma"], 1.0)
    audio = (np.arange(int(3.0 * SAMPLE_RATE)) / SCALE).astype(np.float32)  # speech runs to the clip end
    text, latency = replay_audio(audio, FakeEngine(words), cfg(), prob=fake_vad(words))
    assert text == "alpha beta gamma"
    assert latency["commit_s"]["n"] == 3


def _run_with_hook(words, seconds, c, engine=None):
    got = []
    engine = engine or FakeEngine(words)
    seg = VadSegmenter(fake_vad(words), c.vad_threshold, c.min_silence_s)
    s = Streamer(engine, seg, c, lambda u: None, on_utterance=lambda *a: got.append(a))
    audio = (np.arange(int(seconds * SAMPLE_RATE)) / SCALE).astype(np.float32)
    audio = audio[: len(audio) // WINDOW * WINDOW]
    for i in range(0, len(audio), WINDOW):
        s.feed(audio[i : i + WINDOW])
    s.close()
    return got


def test_on_utterance_gets_full_untrimmed_audio_and_text():
    words = speech([f"w{i}" for i in range(40)], 1.0)  # long speech: the decode buffer is trimmed meanwhile
    got = _run_with_hook(words, 19.5, cfg(max_buffer_s=4.0))
    assert len(got) == 1
    a, text, lat = got[0]
    assert text == " ".join(f"w{i}" for i in range(40))
    assert len(a) > 16 * SAMPLE_RATE  # whole utterance, not just the last trimmed buffer
    assert np.all(np.diff(a) >= 0)  # contiguous ramp: no gaps or duplicates
    assert lat is not None and lat > 0


def test_noise_utterance_with_empty_text_is_not_logged():
    class Mute:
        def transcribe(self, audio, prompt):
            return []

    assert _run_with_hook(speech(["x"], 1.0), 4.0, cfg(), engine=Mute()) == []
