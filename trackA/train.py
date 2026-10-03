"""Train the from-scratch CTC model on a LibriSpeech subset and report dev WER.

Usage:
  uv run python -m trackA.train --train DIR_train-clean-100 --dev DIR_dev-clean --hours 10 --epochs 30 --out models/trackA/h10
Independent of the product: it never touches the owner's data or the frozen eval set.
"""
import argparse
import json
import time
from pathlib import Path

import jiwer
import numpy as np
import torch

from trackA import data
from trackA.decode import beam_search, greedy
from trackA.model import CtcModel
from trackA.text import decode


def spec_augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Mask up to two frequency bands and two time spans per batch (zero is the mean after normalization)."""
    x = x.clone()
    _, t, f = x.shape
    for _ in range(2):
        w = int(rng.integers(0, 15))
        f0 = int(rng.integers(0, max(1, f - w)))
        x[:, :, f0 : f0 + w] = 0
        w = int(rng.integers(0, min(40, max(1, t // 5))))
        t0 = int(rng.integers(0, max(1, t - w)))
        x[:, t0 : t0 + w, :] = 0
    return x


@torch.no_grad()
def transcribe(model: CtcModel, cache_dir: Path, entries: list[dict], device: str, beam: int = 0, batch_frames: int = 20000) -> list[str]:
    model.eval()
    out: dict[int, str] = {}
    pos = {id(e): i for i, e in enumerate(entries)}
    for batch in data.length_buckets(entries, batch_frames, shuffle=False):
        feats, lens, _, _ = data.collate(cache_dir, batch)
        logp, olens = model(feats.to(device), lens.to(device))
        for e, lp, n in zip(batch, logp.cpu().numpy(), olens.tolist()):
            ids = beam_search(lp[:n], beam) if beam else greedy(lp[:n])
            out[pos[id(e)]] = decode(ids)
    return [out[i] for i in range(len(entries))]


def dev_wer(model, cache_dir, entries, device, beam: int = 0) -> float:
    hyps = transcribe(model, cache_dir, entries, device, beam)
    return jiwer.wer([e["text"].lower() for e in entries], hyps)


def train(train_dir: Path, dev_dir: Path, hours: float, epochs: int, out: Path, cache_root: Path, device: str = "cuda",
          dev_utts: int = 300, max_frames: int = 24000, lr: float = 1e-3, seed: int = 0, arch: str = "gru") -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_index = data.build_cache(data.list_items(train_dir), cache_root / "train", max_hours=hours)
    dev_items = data.list_items(dev_dir)
    dev_items = dev_items[:: max(1, len(dev_items) // dev_utts)][:dev_utts]  # spread evenly over speakers
    dev_index = data.build_cache(dev_items, cache_root / "dev")
    subset = data.select_hours(train_index, hours)
    actual_hours = sum(e["frames"] for e in subset) * 0.01 / 3600
    out.mkdir(parents=True, exist_ok=True)

    model = CtcModel(arch=arch).to(device)
    params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(data.length_buckets(subset, max_frames, False))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.15)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    history, best, t0 = [], float("inf"), time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in data.length_buckets(subset, max_frames, shuffle=True, seed=seed + epoch):
            feats, lens, targets, tlens = data.collate(cache_root / "train", batch)
            feats = spec_augment(feats, rng).to(device)
            logp, olens = model(feats, lens.to(device))
            loss = ctc(logp.transpose(0, 1), targets.to(device), olens, tlens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        wer = dev_wer(model, cache_root / "dev", dev_index, device)
        rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "dev_wer": wer, "seconds": round(time.time() - t0)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if wer < best:
            best = wer
            torch.save({"arch": arch, "state_dict": model.state_dict()}, out / "best.pt")
    result = {"arch": arch, "hours_requested": hours, "hours_used": round(actual_hours, 2), "epochs": epochs, "params": params,
              "best_dev_wer": best, "dev_utterances": len(dev_index), "history": history}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--hours", type=float, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("data/trackA_cache"))
    p.add_argument("--arch", choices=["gru", "transformer"], default="gru")
    a = p.parse_args()
    train(a.train, a.dev, a.hours, a.epochs, a.out, a.cache, arch=a.arch)
