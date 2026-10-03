"""Run a model (HF name or CT2 directory) on the frozen eval set and write a report for the promotion gate.

The hash of the frozen manifest is verified first (golden rule 1). Latency is measured in replay mode
through the same streaming pipeline that dictation uses.

Usage: uv run python -m amanuensis.eval.runner MODEL REPORT.json [configs/eval.yaml]
"""
import dataclasses
import json
import statistics
import sys
from pathlib import Path

import yaml

from amanuensis.asr.replay import replay_audio
from amanuensis.config import SAMPLE_RATE, StreamingConfig
from amanuensis.eval import manifest
from amanuensis.eval.report import Utterance, build_report


def evaluate_model(
    model: str,
    eval_cfg: dict,
    sc: StreamingConfig,
    variants: dict[str, str] | None = None,
    terms: list[str] | None = None,
    latency_clips: int = 10,
    engine=None,
) -> dict:
    from faster_whisper.audio import decode_audio

    from amanuensis.asr.engine import FasterWhisperEngine

    mpath, root = Path(eval_cfg["manifest"]), Path(eval_cfg["eval_dir"])
    manifest.verify(mpath, root, Path(eval_cfg["manifest_hash_file"]).read_text().strip())
    entries = json.loads(mpath.read_text(encoding="utf-8"))
    engine = engine or FasterWhisperEngine(dataclasses.replace(sc, model=model))
    utts, p50s = [], []
    for i, e in enumerate(entries):
        audio = decode_audio(str(root / e["audio"]), sampling_rate=SAMPLE_RATE)
        utts.append(Utterance(e["language"], e["text"], " ".join(w.text for w in engine.transcribe(audio, None))))
        if i < latency_clips:
            p50 = replay_audio(audio, engine, sc)[1]["commit_s"]["p50"]
            if p50 is not None:
                p50s.append(p50)
    report = build_report(utts, variants, terms, statistics.median(p50s) if p50s else float("nan"))
    report["model"] = model
    return report


if __name__ == "__main__":
    from amanuensis.config import load_paths, load_streaming, load_variants
    from amanuensis.store import db, lexicon

    cfg = yaml.safe_load(Path(sys.argv[3] if len(sys.argv) > 3 else "configs/eval.yaml").read_text(encoding="utf-8"))
    paths = load_paths()
    terms = lexicon.approved_terms(db.connect(paths.db_path))
    out = evaluate_model(sys.argv[1], cfg, load_streaming(), load_variants(paths.variants_file), terms)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["overall"], indent=2))
