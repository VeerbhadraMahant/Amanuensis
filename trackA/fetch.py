"""Download LibriSpeech splits from OpenSLR, extracting only as much as needed.

train-clean-100 is 6.3 GB; the data-scaling study starts with its first hours, so the tarball is streamed and
extraction stops after --max-files utterances (about 12.7 s each on average: 10 h ~ 2800 files).

Usage: uv run python -m trackA.fetch train-clean-100 data/librispeech --max-files 2800
       uv run python -m trackA.fetch dev-clean data/librispeech
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.openslr.org/resources/12/{split}.tar.gz"


def fetch(split: str, out: Path, max_files: int | None = None) -> int:
    """Returns the number of audio files extracted. Safe to re-run: existing files are skipped."""
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with urllib.request.urlopen(URL.format(split=split)) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name.endswith(".flac"):
                if max_files is not None and n >= max_files:
                    break
                n += 1
            if not (out / member.name).exists():
                tar.extract(member, out, filter="data")
    return n


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("split")
    p.add_argument("out", type=Path)
    p.add_argument("--max-files", type=int, default=None)
    a = p.parse_args()
    print(f"extracted {fetch(a.split, a.out, a.max_files)} audio files")
