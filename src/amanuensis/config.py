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
    max_prompt_terms: int  # lexicon terms put in the decoding prompt; too many invites hallucination


def load_streaming(path: Path = Path("configs/streaming.yaml")) -> StreamingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return StreamingConfig(**{f.name: raw[f.name] for f in fields(StreamingConfig)})


@dataclass(frozen=True)
class PathsConfig:
    db_path: Path
    audio_dir: Path
    lock_file: Path  # present while dictation runs; the trainer refuses to start
    variants_file: Path  # owner-derived spelling variants; optional, absent until romanization.md exists


def load_paths(path: Path = Path("configs/paths.yaml")) -> PathsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PathsConfig(**{f.name: Path(raw[f.name]) for f in fields(PathsConfig)})


def load_language_tags(path: Path = Path("configs/review.yaml")) -> list[str]:
    return list(yaml.safe_load(path.read_text(encoding="utf-8"))["language_tags"])


def load_variants(path: Path) -> dict[str, str]:
    """Owner-derived spelling variants (lowercase variant -> canonical). Empty until docs/romanization.md exists."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
