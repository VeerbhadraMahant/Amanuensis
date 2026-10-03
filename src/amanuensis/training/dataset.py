"""Dataset builder: reviewed utterances -> versioned, hash-recorded manifest.

Golden rules enforced here:
  2. Only 'corrected' / 'approved_as_is' utterances are ever selected (SQL filter).
  1. Any utterance whose audio matches the frozen eval set aborts the build (EvalLeakError).
"""
import hashlib
import json
import random
import sqlite3
import wave
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from amanuensis.text.normalizer import normalize
from amanuensis.training.validators import ValidationConfig, validate

REVIEWED_SQL = "status IN ('corrected', 'approved_as_is') AND final_text IS NOT NULL"


class EvalLeakError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetConfig:
    datasets_dir: Path
    max_hours_per_group: float  # cap per language-tag combination, so no group swamps the rest
    val_fraction: float
    seed: int
    replay_min_fraction: float  # warn when English+German share of the data falls below this


def load_dataset_config(path: Path = Path("configs/dataset.yaml")) -> DatasetConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetConfig(**{f.name: Path(raw[f.name]) if f.name == "datasets_dir" else raw[f.name] for f in fields(DatasetConfig)})


@dataclass
class BuildResult:
    version: str
    manifest_path: Path
    content_hash: str
    utterance_count: int
    hours_by_group: dict[str, float]
    rejected: list[tuple[int, list[str]]] = field(default_factory=list)  # (utterance id, reasons)
    deduped: int = 0
    capped: int = 0
    warnings: list[str] = field(default_factory=list)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eval_audio_hashes(eval_manifest: Path, eval_root: Path) -> set[str]:
    return {sha256_file(eval_root / e["audio"]) for e in json.loads(eval_manifest.read_text(encoding="utf-8"))}


def eval_texts(eval_manifest: Path) -> set[str]:
    """Normalized eval transcripts. A training utterance with the same text is excluded: a re-recording of an
    eval sentence would leak it even though the audio bytes differ."""
    return {normalize(e["text"]) for e in json.loads(eval_manifest.read_text(encoding="utf-8"))}


def verify_registered(conn: sqlite3.Connection, manifest: Path) -> None:
    """A manifest may only be trained on if the builder recorded it and it is unchanged since."""
    row = conn.execute("SELECT content_hash FROM dataset_versions WHERE manifest_path = ?", (str(manifest),)).fetchone()
    if row is None:
        raise ValueError(f"{manifest} is not a registered dataset version; build it with build_dataset")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != row["content_hash"]:
        raise ValueError(f"{manifest} changed after it was built (hash mismatch)")


def _group(tags_json: str) -> str:
    return "+".join(sorted(json.loads(tags_json))) or "untagged"


def build_dataset(
    conn: sqlite3.Connection,
    audio_dir: Path,
    cfg: DatasetConfig,
    vcfg: ValidationConfig,
    variants: dict[str, str],
    eval_hashes: set[str],
    eval_text_set: frozenset[str] | set[str] = frozenset(),
) -> BuildResult:
    rows = conn.execute(f"SELECT * FROM utterances WHERE {REVIEWED_SQL} ORDER BY id").fetchall()
    result = BuildResult("", Path(), "", 0, {})
    kept, seen_audio, seen_text = [], set(), set()
    for r in rows:
        path = audio_dir / r["audio_path"]
        try:
            digest = sha256_file(path)
            if digest in eval_hashes:
                raise EvalLeakError(f"utterance {r['id']} has the same audio as an item in the frozen eval set")
            audio = read_wav(path)
        except (OSError, EOFError, wave.Error) as e:  # missing or corrupt file: reject this utterance only
            result.rejected.append((r["id"], [f"unreadable audio ({type(e).__name__})"]))
            continue
        if normalize(r["final_text"]) in eval_text_set:
            result.rejected.append((r["id"], ["text matches an item in the frozen eval set (excluded)"]))
            continue
        problems = validate(r["final_text"], audio, r["duration_s"], vcfg, variants)
        if problems:
            result.rejected.append((r["id"], problems))
            continue
        key = (r["final_text"].strip().lower(), round(r["duration_s"], 1))
        if digest in seen_audio or key in seen_text:
            result.deduped += 1
            continue
        seen_audio.add(digest)
        seen_text.add(key)
        kept.append({"row": r, "sha256": digest, "group": _group(r["language_tags"])})

    rng = random.Random(cfg.seed)
    rng.shuffle(kept)  # deterministic: the cap keeps a seeded random subset, not just the oldest
    hours: dict[str, float] = {}
    selected = []
    for k in kept:
        g = k["group"]
        if hours.get(g, 0.0) + k["row"]["duration_s"] / 3600 > cfg.max_hours_per_group:
            result.capped += 1
            continue
        hours[g] = hours.get(g, 0.0) + k["row"]["duration_s"] / 3600
        selected.append(k)
    selected.sort(key=lambda k: k["row"]["id"])

    total = sum(hours.values())
    replay = sum(h for g, h in hours.items() if set(g.split("+")) & {"en", "de"} and not set(g.split("+")) - {"en", "de"})
    if total and replay / total < cfg.replay_min_fraction:
        result.warnings.append(
            f"English/German replay share is {replay / total:.0%}, below {cfg.replay_min_fraction:.0%}: "
            "risk of forgetting; review more English/German speech"
        )

    previous = conn.execute("SELECT version FROM dataset_versions ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    n = conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0] + 1
    result.version = f"ds-{n:03d}"
    out_dir = cfg.datasets_dir / result.version
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for k in selected:
        r = k["row"]
        split = "val" if int(k["sha256"][:8], 16) / 0xFFFFFFFF < cfg.val_fraction else "train"
        lines.append(json.dumps({
            "id": r["id"], "audio": r["audio_path"], "text": r["final_text"].strip(), "tags": json.loads(r["language_tags"]),
            "duration_s": r["duration_s"], "sha256": k["sha256"], "split": split,
        }, ensure_ascii=False))
    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    result.manifest_path = manifest
    result.content_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    result.utterance_count = len(selected)
    result.hours_by_group = hours
    conn.execute(
        "INSERT INTO dataset_versions (version, created_at, manifest_path, utterance_count, hours_by_language,"
        " content_hash, parent_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (result.version, datetime.now(timezone.utc).isoformat(timespec="milliseconds"), str(manifest), len(selected),
         json.dumps(hours), result.content_hash, previous["version"] if previous else None),
    )
    conn.commit()
    return result


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
