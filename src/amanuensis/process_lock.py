"""Lock files that keep dictation and training off the 8 GB GPU at the same time, in both directions.

A lock records "pid:process-create-time", so a recycled PID is not mistaken for the original owner.
psutil is used because os.kill(pid, 0) terminates the process on Windows.
"""
import os
from pathlib import Path

import psutil


class LockHeld(RuntimeError):
    pass


def training_lock_path(dictation_lock: Path) -> Path:
    """Convention: the training lock sits next to the dictation lock."""
    return dictation_lock.with_name("training.lock")


def _owner(path: Path) -> tuple[int, float | None] | None:
    try:
        pid_s, _, started = path.read_text().strip().partition(":")
        return int(pid_s), float(started) if started else None
    except (FileNotFoundError, ValueError):
        return None


def is_active(path: Path) -> bool:
    """True if the lock names a live process. A stale lock (crash, or a recycled PID) does not count."""
    owner = _owner(path)
    if owner is None or not psutil.pid_exists(owner[0]):
        return False
    pid, started = owner
    if started is None:
        return True
    try:
        return abs(psutil.Process(pid).create_time() - started) < 2.0
    except psutil.Error:
        return False


def acquire(path: Path) -> None:
    owner = _owner(path)
    if is_active(path) and owner and owner[0] != os.getpid():
        raise LockHeld(f"{path.name} is held by running process {owner[0]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}:{psutil.Process().create_time()}")


def release(path: Path) -> None:
    path.unlink(missing_ok=True)
