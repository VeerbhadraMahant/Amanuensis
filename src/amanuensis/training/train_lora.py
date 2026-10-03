"""LoRA fine-tuning of Whisper from a dataset manifest. Machine-agnostic: device and paths come from config.

Usage: uv run python -m amanuensis.training.train_lora RUN_NAME MANIFEST.jsonl [--config configs/training.yaml]
"""
import argparse
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from amanuensis import process_lock
from amanuensis.logging import log
from amanuensis.training.dataset import load_manifest, read_wav
from amanuensis.training.guard import ensure_can_train


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    device: str
    load_in_8bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    learning_rate: float
    batch_size: int
    grad_accum: int
    epochs: int
    max_steps: int
    gradient_checkpointing: bool
    language: str
    task: str
    seed: int
    min_free_vram_gb: float
    runs_dir: Path
    ct2_quantization: str
    parity_max_wer: float


def load_train_config(path: Path = Path("configs/training.yaml")) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig(**{f.name: Path(raw[f.name]) if f.name == "runs_dir" else raw[f.name] for f in fields(TrainConfig)})


def _label_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text).input_ids
    # The model prepends <|startoftranscript|> itself when it shifts labels right.
    return ids[1:] if ids and ids[0] == tokenizer.convert_tokens_to_ids("<|startoftranscript|>") else ids


def make_batches(items: list[dict], audio_dir: Path, processor, batch_size: int, shuffle: bool, rng: random.Random):
    import torch

    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [items[j] for j in order[i : i + batch_size]]
        feats = processor.feature_extractor(
            [read_wav(audio_dir / e["audio"]) for e in chunk], sampling_rate=16000, return_tensors="pt"
        ).input_features
        labels = [_label_ids(processor.tokenizer, e["text"]) for e in chunk]
        width = max(len(x) for x in labels)
        padded = torch.full((len(labels), width), -100, dtype=torch.long)
        for r, x in enumerate(labels):
            padded[r, : len(x)] = torch.tensor(x)
        yield feats, padded


def train(cfg: TrainConfig, manifest: Path, audio_dir: Path, run_name: str, lock_file: Path, config_path: Path) -> Path:
    """lock_file is the dictation lock; training holds its own lock next to it so dictation cannot start mid-run."""
    ensure_can_train(lock_file, cfg.min_free_vram_gb)
    mine = process_lock.training_lock_path(lock_file)
    process_lock.acquire(mine)
    try:
        return _train(cfg, manifest, audio_dir, run_name, config_path)
    finally:
        process_lock.release(mine)


def _train(cfg: TrainConfig, manifest: Path, audio_dir: Path, run_name: str, config_path: Path) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, WhisperForConditionalGeneration, WhisperProcessor

    run_dir = cfg.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a previous run
    shutil.copy(config_path, run_dir / "train_config.yaml")  # every run's config is logged
    items = load_manifest(manifest)
    train_items = [e for e in items if e["split"] == "train"]
    val_items = [e for e in items if e["split"] == "val"]
    if not train_items:
        raise ValueError("manifest has no training utterances")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    processor = WhisperProcessor.from_pretrained(cfg.base_model, language=cfg.language, task=cfg.task)
    kwargs = {}
    if cfg.load_in_8bit:
        kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": {"": 0}}
    model = WhisperForConditionalGeneration.from_pretrained(cfg.base_model, **kwargs)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    if cfg.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.to(cfg.device)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, bias="none",
    ))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
    device = next(model.parameters()).device

    def run_batch(feats, labels):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return model(input_features=feats.to(device), labels=labels.to(device)).loss

    def val_loss() -> float | None:
        if not val_items:
            return None
        model.eval()
        losses = []
        with torch.no_grad():
            for feats, labels in make_batches(val_items, audio_dir, processor, cfg.batch_size, False, rng):
                losses.append(run_batch(feats, labels).item())
        model.train()
        return sum(losses) / len(losses)

    history, step, t0 = [], 0, time.time()
    status = "ok"
    model.train()
    try:
        done = False
        for epoch in range(cfg.epochs):
            micro = 0
            for feats, labels in make_batches(train_items, audio_dir, processor, cfg.batch_size, True, rng):
                loss = run_batch(feats, labels)
                (loss / cfg.grad_accum).backward()
                micro += 1
                if micro % cfg.grad_accum == 0:
                    opt.step()
                    opt.zero_grad()
                    step += 1
                    rec = {"epoch": epoch, "step": step, "loss": loss.item()}
                    if device.type == "cuda":
                        rec["vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                    history.append(rec)
                    log("train_step", **rec)
                    if cfg.max_steps and step >= cfg.max_steps:
                        done = True
                        break
            else:
                if micro % cfg.grad_accum:  # flush a partial accumulation at epoch end
                    opt.step()
                    opt.zero_grad()
            history.append({"epoch": epoch, "val_loss": val_loss()})
            if done:
                break
    except torch.OutOfMemoryError:
        status = "failed: out of memory (retry with a smaller batch_size and larger grad_accum)"
        log("train_oom", run=run_name)
    except Exception as e:
        status = f"failed: {e!r}"
        raise
    finally:
        if status == "ok":
            model.save_pretrained(run_dir / "adapter")
        import peft
        import transformers

        (run_dir / "run.json").write_text(json.dumps({
            "status": status, "run_name": run_name, "manifest": str(manifest), "config": asdict(cfg) | {"runs_dir": str(cfg.runs_dir)},
            "steps": step, "seconds": round(time.time() - t0, 1), "history": history,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device.type == "cuda" else None,
        }, indent=2), encoding="utf-8")
    if status != "ok":
        raise RuntimeError(status)
    return run_dir


if __name__ == "__main__":
    from amanuensis.config import load_paths

    p = argparse.ArgumentParser()
    p.add_argument("run_name")
    p.add_argument("manifest", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    a = p.parse_args()
    paths = load_paths()
    from amanuensis.store import db
    from amanuensis.training.dataset import verify_registered

    verify_registered(db.connect(paths.db_path), a.manifest)  # refuse hand-made or edited manifests
    print(train(load_train_config(a.config), a.manifest, paths.audio_dir, a.run_name, paths.lock_file, a.config))
