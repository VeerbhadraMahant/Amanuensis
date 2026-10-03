from dataclasses import dataclass, fields
from pathlib import Path

import yaml

SAMPLE_RATE = 16000
WINDOW = 512  # samples per VAD window (32 ms)


@dataclass(frozen=True)
class StreamingConfig:
    model: str
    device: str
    compute_type: str
    cpu_fallback_model: str
    language: str
    vad_threshold: float
    min_silence_s: float
    redecode_interval_s: float
    max_buffer_s: float
    prompt_chars: int


def load_streaming(path: Path = Path("configs/streaming.yaml")) -> StreamingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return StreamingConfig(**{f.name: raw[f.name] for f in fields(StreamingConfig)})
