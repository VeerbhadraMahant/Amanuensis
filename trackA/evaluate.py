"""Evaluate a Track A checkpoint on the owner's English eval slice and compare it with Whisper reports.

Usage: uv run python -m trackA.evaluate CKPT.pt [--eval-config configs/eval.yaml] [--whisper REPORT.json ...]
Reads (never writes) the frozen eval set, after verifying its hash. Only the 'en' slice is used: the model
knows no other language.
"""
import argparse
import json
from pathlib import Path

import jiwer
import numpy as np
import torch
import yaml

from trackA.decode import beam_search, greedy
from trackA.features import log_mel
from trackA.model import CtcModel
from trackA.text import decode, encode


def vocab_normalize(text: str) -> str:
    """Reference text reduced to what this model can output (letters, space, apostrophe), so the comparison is fair."""
    return " ".join(decode(encode(text)).split())


def load_model(ckpt: Path, device: str = "cpu") -> CtcModel:
    saved = torch.load(ckpt, map_location=device)
    model = CtcModel(arch=saved["arch"]).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


@torch.no_grad()
def transcribe_audio(model: CtcModel, audio: np.ndarray, beam: int = 0, device: str = "cpu") -> str:
    feats = torch.from_numpy(log_mel(audio)).unsqueeze(0).to(device)
    logp, olens = model(feats, torch.tensor([feats.shape[1]], device=device))
    lp = logp[0, : int(olens[0])].cpu().numpy()
    return " ".join(decode(beam_search(lp, beam) if beam else greedy(lp)).split())


def evaluate_english(model: CtcModel, entries: list[dict], root: Path, device: str = "cpu", beam: int = 8) -> dict:
    from faster_whisper.audio import decode_audio

    en = [e for e in entries if e["language"] == "en"]
    if not en:
        raise ValueError("the eval set has no 'en' slice")
    audio = [decode_audio(str(root / e["audio"]), sampling_rate=16000) for e in en]
    refs = [vocab_normalize(e["text"]) for e in en]
    result = {"n": len(en)}
    for name, b in (("wer_greedy", 0), (f"wer_beam{beam}", beam)):
        hyps = [transcribe_audio(model, a, b, device) for a in audio]
        result[name] = jiwer.wer(refs, hyps)
    return result


if __name__ == "__main__":
    from amanuensis.eval import manifest

    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--eval-config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--whisper", type=Path, nargs="*", default=[], help="eval reports from amanuensis.eval.runner")
    a = p.parse_args()
    cfg = yaml.safe_load(a.eval_config.read_text(encoding="utf-8"))
    root, mpath = Path(cfg["eval_dir"]), Path(cfg["manifest"])
    manifest.verify(mpath, root, Path(cfg["manifest_hash_file"]).read_text().strip())
    out = {"trackA": evaluate_english(load_model(a.ckpt), json.loads(mpath.read_text(encoding="utf-8")), root)}
    for report in a.whisper:
        r = json.loads(report.read_text(encoding="utf-8"))
        out[f"whisper:{r.get('model', report.stem)}"] = {"en_normalized_wer": r["per_language"]["en"]["normalized_wer"]}
    print(json.dumps(out, indent=2))
