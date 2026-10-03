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


def freeze(eval_dir: Path) -> str:
    """Record the hash of eval_dir/manifest.json (and its audio) in eval_dir/manifest.sha256.

    Refuses to overwrite an existing hash: the frozen set is frozen (golden rule 1). To change it, ask the owner.
    """
    hash_file = eval_dir / "manifest.sha256"
    if hash_file.exists():
        raise FileExistsError(f"{hash_file} already exists; the eval set is frozen and must not be re-hashed")
    digest = content_hash(eval_dir / "manifest.json", eval_dir)
    hash_file.write_text(digest)
    return digest


if __name__ == "__main__":
    import sys

    print(freeze(Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval_frozen")))
