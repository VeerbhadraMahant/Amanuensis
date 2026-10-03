"""Lock file marking that dictation is running, so training can refuse to start (8 GB VRAM).

psutil is used because os.kill(pid, 0) terminates the process on Windows.
"""
import os
from pathlib import Path

import psutil


def acquire(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()))


def release(path: Path) -> None:
    path.unlink(missing_ok=True)


def is_active(path: Path) -> bool:
    """True if the lock names a live process. A stale lock from a crash does not count."""
    try:
        pid = int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return False
    return psutil.pid_exists(pid)
