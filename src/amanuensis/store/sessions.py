"""Session and utterance logging. Every closed VAD segment is stored with status 'raw'."""
import json
import sqlite3
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from amanuensis.config import SAMPLE_RATE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def start_session(conn: sqlite3.Connection, model_version: str, mode: str, notes: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO sessions (started_at, model_version, mode, notes) VALUES (?, ?, ?, ?)",
        (_now(), model_version, mode, notes),
    )
    conn.commit()
    return cur.lastrowid


def end_session(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), session_id))
    conn.commit()


def write_wav(path: Path, audio: np.ndarray) -> None:
    """16 kHz mono 16-bit WAV (ADR-005)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def log_utterance(
    conn: sqlite3.Connection,
    audio_dir: Path,
    session_id: int,
    audio: np.ndarray,
    hypothesis_raw: str,
    hypothesis_normalized: str,
    latency_ms: float | None,
) -> int:
    """Store audio + hypotheses. audio_path is stored relative to audio_dir."""
    created = _now()
    rel = f"s{session_id}/{uuid.uuid4().hex}.wav"  # timestamps collide: Windows clock ticks are ~15 ms
    write_wav(audio_dir / rel, audio)
    cur = conn.execute(
        "INSERT INTO utterances (session_id, audio_path, duration_s, hypothesis_raw, hypothesis_normalized,"
        " latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, rel, len(audio) / SAMPLE_RATE, hypothesis_raw, hypothesis_normalized, latency_ms, created),
    )
    conn.commit()
    return cur.lastrowid


def tags_json(tags: list[str]) -> str:
    return json.dumps(sorted(set(tags)))
