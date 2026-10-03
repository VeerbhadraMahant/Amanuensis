"""Standalone data validators. Each returns None when fine, or a short reason string."""
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from amanuensis.text.spelling import check_spelling


@dataclass(frozen=True)
class ValidationConfig:
    min_duration_s: float
    max_duration_s: float
    max_clipping_fraction: float  # share of samples at full scale
    min_chars_per_s: float
    max_chars_per_s: float


def load_validation(path: Path = Path("configs/validation.yaml")) -> ValidationConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ValidationConfig(**{f.name: raw[f.name] for f in fields(ValidationConfig)})


def check_duration(duration_s: float, cfg: ValidationConfig) -> str | None:
    if duration_s < cfg.min_duration_s:
        return f"too short ({duration_s:.2f}s)"
    if duration_s > cfg.max_duration_s:
        return f"too long ({duration_s:.2f}s)"
    return None


def check_text_nonempty(text: str | None) -> str | None:
    return None if text and text.strip() else "empty text"


def check_clipping(audio: np.ndarray, cfg: ValidationConfig) -> str | None:
    if len(audio) == 0:
        return "empty audio"
    frac = float(np.mean(np.abs(audio) >= 0.999))
    return f"clipping ({frac:.1%} of samples)" if frac > cfg.max_clipping_fraction else None


def check_text_rate(text: str, duration_s: float, cfg: ValidationConfig) -> str | None:
    """Characters per second: far too few or too many means the text does not match the audio."""
    rate = len(text.strip()) / duration_s if duration_s > 0 else float("inf")
    if rate < cfg.min_chars_per_s:
        return f"text too short for audio ({rate:.1f} chars/s)"
    if rate > cfg.max_chars_per_s:
        return f"text too long for audio ({rate:.1f} chars/s)"
    return None


def check_spelling_compliance(text: str, variants: dict[str, str]) -> str | None:
    flags = check_spelling(text, variants)
    return f"non-canonical spelling: {', '.join(f.word for f in flags)}" if flags else None


def validate(text: str | None, audio: np.ndarray, duration_s: float, cfg: ValidationConfig, variants: dict[str, str]) -> list[str]:
    """All problems found; an empty list means the utterance may enter a dataset."""
    problems = [check_text_nonempty(text), check_duration(duration_s, cfg), check_clipping(audio, cfg)]
    if text and text.strip():
        problems += [check_text_rate(text, duration_s, cfg), check_spelling_compliance(text, variants)]
    return [p for p in problems if p]
