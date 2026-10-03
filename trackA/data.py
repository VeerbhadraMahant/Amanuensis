"""LibriSpeech loading with a feature cache. Subsets are nested: the first N hours of the speaker-sorted corpus,
so the 10 h set is contained in the 25 h set, and so on (a clean data-scaling comparison)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from trackA.features import log_mel
from trackA.text import encode


@dataclass(frozen=True)
class Item:
    audio: Path
    text: str


def list_items(root: Path) -> list[Item]:
    """Every utterance under a LibriSpeech split directory, in deterministic speaker/chapter order."""
    items = []
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, _, text = line.partition(" ")
            audio = trans.parent / f"{uid}.flac"
            if audio.exists():  # a partially extracted corpus lists transcripts whose audio was not extracted
                items.append(Item(audio, text))
    return items


def build_cache(items: list[Item], cache_dir: Path, max_hours: float | None = None) -> list[dict]:
    """Compute log-mel features once and store them. Stops after max_hours of audio. Returns the index entries."""
    from faster_whisper.audio import decode_audio

    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    done = {e["audio"] for e in index}
    hours = sum(e["frames"] for e in index) * 0.01 / 3600
    for item in items:
        if max_hours is not None and hours >= max_hours:
            break
        if str(item.audio) in done:
            continue
        feats = log_mel(decode_audio(str(item.audio), sampling_rate=16000))
        path = cache_dir / f"{len(index):06d}.npy"
        np.save(path, feats.astype(np.float16))
        index.append({"audio": str(item.audio), "npy": path.name, "frames": len(feats), "ids": encode(item.text), "text": item.text})
        hours += len(feats) * 0.01 / 3600
    index_path.write_text(json.dumps(index))
    return index


def select_hours(index: list[dict], hours: float) -> list[dict]:
    """The first `hours` of audio (10 ms per frame), keeping the cache order so subsets are nested."""
    out, total = [], 0.0
    for e in index:
        if total >= hours:
            break
        out.append(e)
        total += e["frames"] * 0.01 / 3600
    return out


def collate(cache_dir: Path, entries: list[dict]):
    feats = [torch.from_numpy(np.load(cache_dir / e["npy"]).astype(np.float32)) for e in entries]
    lens = torch.tensor([len(f) for f in feats])
    padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    targets = [torch.tensor(e["ids"]) for e in entries]
    return padded, lens, torch.cat(targets), torch.tensor([len(t) for t in targets])


def length_buckets(entries: list[dict], max_frames: int, shuffle: bool, seed: int = 0) -> list[list[dict]]:
    """Group similar-length utterances so padding is small; each batch holds at most max_frames padded frames."""
    order = sorted(entries, key=lambda e: e["frames"])
    batches, cur = [], []
    for e in order:
        if cur and (len(cur) + 1) * e["frames"] > max_frames:
            batches.append(cur)
            cur = []
        cur.append(e)
    if cur:
        batches.append(cur)
    if shuffle:
        np.random.default_rng(seed).shuffle(batches)
    return batches
