"""Hash-verified frozen eval manifest (golden rule 1, ADR-006)."""
import hashlib
import json
from pathlib import Path


class ManifestHashError(RuntimeError):
    pass


def content_hash(manifest_path: Path, audio_root: Path) -> str:
    """SHA-256 over the manifest bytes plus every referenced audio file, in manifest order."""
    h = hashlib.sha256(manifest_path.read_bytes())
    for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
        h.update((audio_root / entry["audio"]).read_bytes())
    return h.hexdigest()


def verify(manifest_path: Path, audio_root: Path, expected_hash: str) -> None:
    actual = content_hash(manifest_path, audio_root)
    if actual != expected_hash:
        raise ManifestHashError(f"eval manifest hash mismatch: expected {expected_hash}, got {actual}")
