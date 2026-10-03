"""Whisper on the same LibriSpeech dev utterances the from-scratch model is scored on.

Usage: uv run python -m trackA.compare_whisper [--models tiny base small] [--cache data/trackA_cache]
Public data only. The owner's English eval slice is a separate comparison (trackA.evaluate).
"""
import argparse
import json
from pathlib import Path

import jiwer

from amanuensis.text.normalizer import normalize
from trackA.evaluate import vocab_normalize


def whisper_wer(model_name: str, index: list[dict], device: str = "cuda", compute_type: str = "int8") -> float:
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    refs, hyps = [], []
    for e in index:
        segs, _ = model.transcribe(decode_audio(e["audio"], sampling_rate=16000), language="en", beam_size=1, temperature=0.0)
        hyps.append(vocab_normalize(normalize(" ".join(s.text for s in segs))))
        refs.append(vocab_normalize(e["text"]))
    return jiwer.wer(refs, hyps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    p.add_argument("--cache", type=Path, default=Path("data/trackA_cache"))
    p.add_argument("--out", type=Path, default=Path("models/trackA/whisper_dev.json"))
    a = p.parse_args()
    index = json.loads((a.cache / "dev" / "index.json").read_text())
    result = {"dev_utterances": len(index), "wer": {m: whisper_wer(m, index) for m in a.models}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
