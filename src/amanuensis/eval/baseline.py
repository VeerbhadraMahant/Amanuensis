"""Phase 0 baseline: run Whisper sizes on the frozen eval set.

Usage: uv run python -m amanuensis.eval.baseline [configs/eval.yaml]
"""
import json
import sys
import time
from pathlib import Path

import yaml

from amanuensis.eval import manifest
from amanuensis.eval.report import Utterance, per_language_report


def run(cfg_path: Path) -> dict:
    from faster_whisper import WhisperModel  # heavy import, kept local

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mpath, root = Path(cfg["manifest"]), Path(cfg["eval_dir"])
    manifest.verify(mpath, root, Path(cfg["manifest_hash_file"]).read_text().strip())
    entries = json.loads(mpath.read_text(encoding="utf-8"))
    variants = None
    if cfg["variants_file"]:
        variants = yaml.safe_load(Path(cfg["variants_file"]).read_text(encoding="utf-8"))

    results = {}
    for name in cfg["models"]:
        model = WhisperModel(name, device="cuda", compute_type=cfg["compute_type"])
        utts, audio_s, t0 = [], 0.0, time.perf_counter()
        for e in entries:
            segs, info = model.transcribe(str(root / e["audio"]), language=cfg["language"])
            utts.append(Utterance(e["language"], e["text"], " ".join(s.text for s in segs)))
            audio_s += info.duration
        results[name] = {
            "per_language": per_language_report(utts, variants),
            "real_time_factor": (time.perf_counter() - t0) / audio_s,
        }
        del model
    return results


if __name__ == "__main__":
    out = run(Path(sys.argv[1] if len(sys.argv) > 1 else "configs/eval.yaml"))
    print(json.dumps(out, indent=2))
