"""Pre-flight checks. Training and dictation cannot share an 8 GB GPU."""
from collections.abc import Callable
from pathlib import Path

from amanuensis import process_lock


class TrainingBlocked(RuntimeError):
    pass


def free_vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        raise TrainingBlocked("CUDA is not available")
    return torch.cuda.mem_get_info()[0] / 1024**3


def ensure_can_train(lock_path: Path, min_free_vram_gb: float, free_vram: Callable[[], float] = free_vram_gb) -> None:
    if process_lock.is_active(lock_path):
        raise TrainingBlocked("dictation is running (lock file names a live process); stop it before training")
    free = free_vram()
    if free < min_free_vram_gb:
        raise TrainingBlocked(f"only {free:.1f} GB VRAM free, need {min_free_vram_gb:.1f} GB")
