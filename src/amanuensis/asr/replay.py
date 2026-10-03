"""Replay mode: feed a WAV through the streaming pipeline as if live.

Usage: uv run python -m amanuensis.asr.replay FILE.wav [--ref "reference text"]
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from amanuensis.asr.engine import FasterWhisperEngine
from amanuensis.asr.streaming import Streamer
from amanuensis.asr.types import Engine
from amanuensis.audio.vad import SileroProb, VadSegmenter
from amanuensis.config import SAMPLE_RATE, WINDOW, StreamingConfig, load_streaming
from amanuensis.eval.metrics import normalized_wer


@dataclass(frozen=True)
class ReplayResult:
    streaming_text: str
    offline_text: str
    latency: dict


def replay_audio(
    audio: np.ndarray, engine: Engine, cfg: StreamingConfig, prob=None, on_utterance=None
) -> tuple[str, dict]:
    tail = np.zeros(int((cfg.min_silence_s + 0.3) * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([audio.astype(np.float32), tail])
    audio = audio[: len(audio) // WINDOW * WINDOW]
    seg = VadSegmenter(prob or SileroProb(), cfg.vad_threshold, cfg.min_silence_s)
    streamer = Streamer(engine, seg, cfg, lambda u: None, on_utterance=on_utterance)
    for i in range(0, len(audio), WINDOW):
        streamer.feed(audio[i : i + WINDOW])
    streamer.close()
    return " ".join(streamer.utterances), streamer.stats.summary()


def replay_file(path: str, engine: Engine, cfg: StreamingConfig) -> ReplayResult:
    from faster_whisper.audio import decode_audio

    audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
    text, latency = replay_audio(audio, engine, cfg)
    offline = " ".join(w.text for w in engine.transcribe(audio, None))
    return ReplayResult(text, offline, latency)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--ref", help="reference transcript; WER is reported against it")
    p.add_argument("--config", default="configs/streaming.yaml")
    args = p.parse_args()
    cfg = load_streaming(Path(args.config))
    r = replay_file(args.wav, FasterWhisperEngine(cfg), cfg)
    out = {"latency": r.latency, "streaming_text": r.streaming_text, "offline_text": r.offline_text}
    out["wer_streaming_vs_offline"] = normalized_wer([r.offline_text], [r.streaming_text])
    if args.ref:
        out["wer_streaming_vs_ref"] = normalized_wer([args.ref], [r.streaming_text])
        out["wer_offline_vs_ref"] = normalized_wer([args.ref], [r.offline_text])
    print(json.dumps(out, indent=2, ensure_ascii=False))
