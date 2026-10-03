"""Phase 3: measure lexicon term accuracy and WER with and without prompt biasing.

Usage: uv run python -m amanuensis.eval.biasing [configs/eval.yaml]
Reads terms from the approved lexicon in the DB; verifies the frozen manifest hash first.
"""
import json
import sys
from collections.abc import Callable
from pathlib import Path

from amanuensis.eval.metrics import normalized_wer, term_accuracy
from amanuensis.eval.report import Utterance, per_language_report

Transcribe = Callable[[str, str | None], str]  # (audio path, prompt) -> text


def bias_prompt(terms: list[str], max_terms: int) -> str | None:
    return ", ".join(terms[:max_terms]) + "." if terms[:max_terms] else None


def _run(transcribe: Transcribe, entries: list[dict], root: Path, prompt: str | None, variants, terms) -> dict:
    utts = [Utterance(e["language"], e["text"], transcribe(str(root / e["audio"]), prompt)) for e in entries]
    refs, hyps = [u.ref for u in utts], [u.hyp for u in utts]
    return {
        "per_language": per_language_report(utts, variants, terms),
        "overall": {"normalized_wer": normalized_wer(refs, hyps, variants), "term_accuracy": term_accuracy(refs, hyps, terms)},
    }


def compare_biasing(
    transcribe: Transcribe, entries: list[dict], root: Path, terms: list[str], max_terms: int, variants=None
) -> dict:
    """Run the eval set unbiased and biased. Negative wer delta / positive term delta = biasing helped."""
    base = _run(transcribe, entries, root, None, variants, terms)
    biased = _run(transcribe, entries, root, bias_prompt(terms, max_terms), variants, terms)
    b, d = base["overall"], biased["overall"]
    ta = None if b["term_accuracy"] is None else d["term_accuracy"] - b["term_accuracy"]
    return {
        "baseline": base,
        "biased": biased,
        "delta": {"normalized_wer": d["normalized_wer"] - b["normalized_wer"], "term_accuracy": ta},
        "terms_in_prompt": min(len(terms), max_terms),
    }


if __name__ == "__main__":
    import yaml
    from faster_whisper import WhisperModel

    from amanuensis.config import load_paths, load_streaming, load_variants
    from amanuensis.eval import manifest
    from amanuensis.store import db, lexicon

    cfg = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1 else "configs/eval.yaml").read_text(encoding="utf-8"))
    sc, paths = load_streaming(), load_paths()
    root, mpath = Path(cfg["eval_dir"]), Path(cfg["manifest"])
    manifest.verify(mpath, root, Path(cfg["manifest_hash_file"]).read_text().strip())
    model = WhisperModel(sc.model, device="cuda", compute_type=sc.compute_type)

    def transcribe(path: str, prompt: str | None) -> str:
        segs, _ = model.transcribe(path, language=sc.language, initial_prompt=prompt, temperature=0.0)
        return " ".join(s.text for s in segs)

    terms = lexicon.approved_terms(db.connect(paths.db_path))
    out = compare_biasing(transcribe, json.loads(mpath.read_text(encoding="utf-8")), root, terms,
                          sc.max_prompt_terms, load_variants(paths.variants_file))
    print(json.dumps(out, indent=2, ensure_ascii=False))
