from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # seconds
    end: float


class Engine(Protocol):
    def transcribe(self, audio: np.ndarray, prompt: str | None) -> list[Word]:
        """Words with times relative to the start of `audio`."""
        ...
